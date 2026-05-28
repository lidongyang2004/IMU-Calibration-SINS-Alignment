import os
import numpy as np

# -----------------------------------------------
#     常数定义 (严格对齐 C++ IMU_Structs.h)
# -----------------------------------------------
GS_XWGI = 1.0850694444e-7      # XW-GI7681 陀螺仪数据转换比例因子
r_XWGI = 100                   # XW-GI7681 采样率100Hz
readtime = 40.00               # 陀螺旋转读取时间 (s)
increment = 0.00018            # 角度积分增量阈值 (度)
PAI = 3.141592653589           # 圆周率
Deg = 180.0 / PAI              # 弧度转化为度
Rad = PAI / 180.0              # 度转化为弧度
we = 7.292115e-5 * Deg         # 地球自转角速度 (度每秒)
fai = 30.531651244 * Rad       # 转台实验室纬度 (rad)
circle = 360.0                 # 一圈360度

def calc_angle(file_path, axis_idx):
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
            line = line.strip()
            if not line or ';' not in line:
                continue
            try:
                _, data_part = line.split(';', 1)
                if '*' in data_part:
                    data_part = data_part.split('*', 1)[0]
                
                fields = data_part.split(',')
                
                # 读取 GPS 时间
                week = int(fields[0])
                second = float(fields[1])
                time_sec = week * 604800 + second
                
                # 解析陀螺仪角速度 (rad/s)
                raw_z = float(fields[6])
                raw_y = float(fields[7])
                raw_x = float(fields[8])
                
                # 比例因子转换，注意 Y 轴取反逻辑 (与 C++ 保持一致)
                val_z = raw_z * GS_XWGI * r_XWGI
                val_y = -raw_y * GS_XWGI * r_XWGI 
                val_x = raw_x * GS_XWGI * r_XWGI
                gyr = np.array([val_x, val_y, val_z])
                
                # 梯形积分计算增量
                if prev_time is not None:
                    dt = time_sec - prev_time
                    if dt > 0.0:
                        # 当前历元三轴角度增量 (度)
                        add = 0.5 * (gyr + prev_gyr) * dt * Deg
                        
                        # 判断主轴增量是否达到阈值，若达到则开启三轴同步积分
                        if not is_integrating and abs(add[axis_idx]) >= increment:
                            is_integrating = True
                            
                        # 在指定时间 readtime 内执行积分
                        if is_integrating and (readtime - total_time) > 1e-3:
                            angle += add
                            total_time += dt
                            
                prev_time = time_sec
                prev_gyr = gyr
                
                # 若达到设定读取时长，结束循环
                if total_time >= readtime:
                    break
                    
            except (ValueError, IndexError):
                continue
                
    # 返回完整的三轴积分角度
    return angle, total_time


def main():
    data_dir = "data/Calibration"
    
    # 待求参数矩阵
    bias = np.zeros(3)          # 零偏 (1D Array)
    S_g = np.zeros((3, 3))      # 比例因子误差矩阵 (主对角线)
    N_g = np.zeros((3, 3))      # 交轴耦合误差矩阵 (非对角线)
    
    axes_files = [
        ("X轴", 0, "x+360.ASC", "x-360.ASC"),
        ("Y轴", 1, "y+360.ASC", "y-360.ASC"),
        ("Z轴", 2, "z+360.ASC", "z-360.ASC")
    ]
    
    print("开始读取陀螺仪动态旋转数据并进行多轴积分计算...\n")
    
    for axis_name, axis_idx, pos_file, neg_file in axes_files:
        path_pos = os.path.join(data_dir, pos_file)
        path_neg = os.path.join(data_dir, neg_file)
        
        # 1. 积分得到正反转时的三轴角度 (Deg) 与总时间 (s)
        angle_pos, t_pos = calc_angle(path_pos, axis_idx)
        angle_neg, t_neg = calc_angle(path_neg, axis_idx)
        
        if angle_pos is None or angle_neg is None:
            continue
            
        # 2. 根据公式计算当前旋转轴带来的各项参数
        avg_t = (t_pos + t_neg) / 2.0
        
        # (A) 零偏 bg 仅由主轴计算 = (α1 + α2)/(2*t) - ωe*sin(φ)
        bias_deg_s = (angle_pos[axis_idx] + angle_neg[axis_idx]) / (2.0 * avg_t) - we * np.sin(fai)
        bias[axis_idx] = bias_deg_s * 3600.0  # 转换为 deg/h
        
        # (B) 比例因子误差 S_g (主轴对角线元素) = |α1 - α2|/(2*360) - 1
        S_g[axis_idx, axis_idx] = abs(angle_pos[axis_idx] - angle_neg[axis_idx]) / (2.0 * circle) - 1.0
        
        # (C) 交轴耦合误差 N_g (提取非主轴的寄生积分角度)
        for i in range(3):
            if i != axis_idx:
                # 正转和反转做差可以消除对应轴零偏在相同时间段内的累积误差
                N_g[i, axis_idx] = (angle_pos[i] - angle_neg[i]) / (2.0 * circle)
        
        print(f"[{axis_name} 旋转标定]")
        print(f"  正转主轴角度: {angle_pos[axis_idx]:8.4f}° | 耗时: {t_pos:.3f}s")
        print(f"  反转主轴角度: {angle_neg[axis_idx]:8.4f}° | 耗时: {t_neg:.3f}s")
        print(f"  --> 计算得主轴零偏: {bias[axis_idx]:.6f} deg/h")
        print(f"  --> 计算得比例因子: {S_g[axis_idx, axis_idx]:.8f}")
        print(f"  --> 提取交轴系数:   N_g[:, {axis_idx}] 已更新\n")

    # 3. 组织成标准的矩阵形式 bg
    b_g = bias.reshape(3, 1)

    # 4. 打印标准矩阵结果
    np.set_printoptions(suppress=True, precision=8)
    print("="*55)
    print(" 陀螺仪全参数角位置法标定成功！标准误差矩阵形式输出：")
    print("="*55)
    
    print("1. 陀螺仪零偏向量 b_g (deg/h):")
    print(b_g)
    print("\n2. 比例因子误差矩阵 S_g:")
    print(S_g)
    print("\n3. 交轴耦合误差矩阵 N_g:")
    print(N_g)
    
    # 5. 保存结果至文件
    output_path = "GyrCaliResult_Matrix.txt"
    with open(output_path, 'w') as out_f:
        out_f.write("=== GYROSCOPE CALIBRATION MATRICES ===\n\n")
        out_f.write("1. Bias Vector b_g (deg/h):\n")
        out_f.write(np.array2string(b_g, separator=', ') + "\n\n")
        out_f.write("2. Scale Factor Error Matrix S_g:\n")
        out_f.write(np.array2string(S_g, separator=', ') + "\n\n")
        out_f.write("3. Cross-axis Coupling Matrix N_g:\n")
        out_f.write(np.array2string(N_g, separator=', ') + "\n")
        
    print(f"\n结果矩阵已成功保存在: {output_path}")

if __name__ == "__main__":
    main()