import os
import numpy as np
from imu_calib.utils.io import parse_imu_line
from imu_calib.constants import WE, FAI, RAD_TO_DEG, DEG_TO_RAD, CIRCLE_DEG, READ_TIME, INCREMENT_THRESHOLD

def calc_angle_integration(file_path, axis_idx):
    """
    读取单轴旋转数据，对陀螺仪三轴数据进行梯形积分，获得三轴总旋转角度和用时。
    :param file_path: 数据文件路径
    :param axis_idx: 当前旋转主轴索引 (0:X, 1:Y, 2:Z)
    :return: angle (3维numpy数组), total_time (浮点数)
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到 {file_path}")
        return None, None

    total_time = 0.0
    angle = np.zeros(3)
    is_integrating = False
    
    prev_time = None
    prev_gyr = np.zeros(3)

    with open(file_path, 'r') as f:
        for line in f:
            epoch = parse_imu_line(line)
            if epoch is None:
                continue
                
            time_sec = epoch['time']
            # 使用 rad/s 单位
            gyr = epoch['gyr']
            
            # 梯形积分计算增量 (转化为度)
            if prev_time is not None:
                dt = time_sec - prev_time
                if dt > 0.0:
                    # 三轴角度积分增量 (度)
                    add = 0.5 * (gyr + prev_gyr) * dt * RAD_TO_DEG
                    
                    # 判断主轴增量是否达到阈值，若达到则开启三轴同步积分
                    if not is_integrating and abs(add[axis_idx]) >= INCREMENT_THRESHOLD:
                        is_integrating = True
                        
                    # 在指定时间 readtime 内执行积分
                    if is_integrating and (READ_TIME - total_time) > 1e-3:
                        angle += add
                        total_time += dt
                        
            prev_time = time_sec
            prev_gyr = gyr
            
            # 若达到设定读取时长，结束循环
            if total_time >= READ_TIME:
                break
                
    return angle, total_time

class GyroscopeCalibrator:
    """
    角位置法陀螺仪标定器
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.axes_files = [
            ("X轴", 0, "x+360.ASC", "x-360.ASC"),
            ("Y轴", 1, "y+360.ASC", "y-360.ASC"),
            ("Z轴", 2, "z+360.ASC", "z-360.ASC")
        ]
        
        # 标定结果
        self.bias = np.zeros(3)          # 零偏 (deg/h)
        self.S_g = np.zeros((3, 3))      # 比例因子误差矩阵 (对角)
        self.N_g = np.zeros((3, 3))      # 交轴耦合误差矩阵 (非对角)

    def run_calibration(self):
        """
        运行角位置法陀螺标定
        """
        print("开始读取陀螺仪动态旋转数据并进行多轴积分计算...\n")
        
        for axis_name, axis_idx, pos_file, neg_file in self.axes_files:
            path_pos = os.path.join(self.data_dir, pos_file)
            path_neg = os.path.join(self.data_dir, neg_file)
            
            # 1. 积分得到正反转时的三轴角度 (Deg) 与总时间 (s)
            angle_pos, t_pos = calc_angle_integration(path_pos, axis_idx)
            angle_neg, t_neg = calc_angle_integration(path_neg, axis_idx)
            
            if angle_pos is None or angle_neg is None:
                print(f"警告: {axis_name} 标定失败，数据文件损坏或缺失。")
                continue
                
            # 2. 根据公式计算当前旋转轴带来的各项参数
            avg_t = (t_pos + t_neg) / 2.0
            
            # (A) 零偏 bg 仅由主轴计算 = (α1 + α2)/(2*t) - ωe*sin(φ)
            bias_deg_s = (angle_pos[axis_idx] + angle_neg[axis_idx]) / (2.0 * avg_t) - WE * np.sin(FAI)
            self.bias[axis_idx] = bias_deg_s * 3600.0  # 转换为 deg/h
            
            # (B) 比例因子误差 S_g (主轴对角线元素) = |α1 - α2|/(2*360) - 1
            self.S_g[axis_idx, axis_idx] = abs(angle_pos[axis_idx] - angle_neg[axis_idx]) / (2.0 * CIRCLE_DEG) - 1.0
            
            # (C) 交轴耦合误差 N_g (提取非主轴的寄生积分角度)
            for i in range(3):
                if i != axis_idx:
                    # 正转和反转做差可以消除对应轴零偏在相同时间段内的累积误差
                    self.N_g[i, axis_idx] = (angle_pos[i] - angle_neg[i]) / (2.0 * CIRCLE_DEG)
            
            print(f"[{axis_name} 旋转标定]")
            print(f"  正转主轴角度: {angle_pos[axis_idx]:8.4f}° | 耗时: {t_pos:.3f}s")
            print(f"  反转主轴角度: {angle_neg[axis_idx]:8.4f}° | 耗时: {t_neg:.3f}s")
            print(f"  --> 计算得主轴零偏: {self.bias[axis_idx]:.6f} deg/h")
            print(f"  --> 计算得比例因子: {self.S_g[axis_idx, axis_idx]:.8f}")
            print(f"  --> 提取交轴系数:   N_g[:, {axis_idx}] 已更新\n")
            
        return self.bias.reshape(3, 1), self.S_g, self.N_g

    def save_results(self, output_path):
        """
        保存结果到文件
        """
        b_g = self.bias.reshape(3, 1)
        with open(output_path, 'w') as out_f:
            out_f.write("=== GYROSCOPE CALIBRATION MATRICES ===\n\n")
            out_f.write("1. Bias Vector b_g (deg/h):\n")
            out_f.write(np.array2string(b_g, separator=', ') + "\n\n")
            out_f.write("2. Scale Factor Error Matrix S_g:\n")
            out_f.write(np.array2string(self.S_g, separator=', ') + "\n\n")
            out_f.write("3. Cross-axis Coupling Matrix N_g:\n")
            out_f.write(np.array2string(self.N_g, separator=', ') + "\n")
            
        print(f"陀螺仪结果矩阵已保存至: {output_path}")
