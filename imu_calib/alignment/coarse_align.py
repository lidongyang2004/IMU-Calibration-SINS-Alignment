import numpy as np
from imu_calib.constants import GRAVITY, WE, FAI, RAD_TO_DEG, DEG_TO_RAD

def adjust_axes_to_frd(raw_vec):
    """
    将右-前-上 (RFU) 坐标系下的向量调整到前-右-下 (FRD) 载体坐标系
    RFU: [x_rfu, y_rfu, z_rfu]
    FRD: [y_rfu, x_rfu, -z_rfu]
    """
    return np.array([raw_vec[1], raw_vec[0], -raw_vec[2]])

def solve_attitude_from_vectors(f_b_raw, w_b_raw):
    """
    通过重力比力和地球自转矢量进行双矢量解析定姿计算姿态角
    :param f_b_raw: 比力输入向量 [ax, ay, az] (RFU)
    :param w_b_raw: 角速度输入向量 [gx, gy, gz] (RFU)
    :return: pitch, roll, yaw (度), C_b_n (3x3 旋转矩阵)
    """
    # 1. 调整到前-右-下 (FRD) 载体坐标系
    f_b = adjust_axes_to_frd(f_b_raw)
    w_b = adjust_axes_to_frd(w_b_raw)
    
    # 2. 导航坐标系 (n系, 北-东-地 NED) 下的已知矢量
    # 静态下比力 f_n = [0, 0, -g]^T
    f_n = np.array([0.0, 0.0, -GRAVITY])
    # 地球自转 w_ie_n = [we*cos(fai), 0, -we*sin(fai)]^T
    we_rad_s = WE * DEG_TO_RAD
    w_n = np.array([we_rad_s * np.cos(FAI), 0.0, -we_rad_s * np.sin(FAI)])
    
    # 3. 构造 n 系下的正交基 [u1_n, u2_n, u3_n]
    u1_n = f_n / np.linalg.norm(f_n)
    cross_n = np.cross(f_n, w_n)
    u2_n = cross_n / np.linalg.norm(cross_n)
    u3_n = np.cross(u1_n, u2_n)
    
    # 4. 构造 b 系下的正交基 [u1_b, u2_b, u3_b]
    # 为防止分母为0，添加微小扰动
    norm_f_b = np.linalg.norm(f_b)
    if norm_f_b < 1e-6:
        norm_f_b = 1e-6
    u1_b = f_b / norm_f_b
    
    cross_b = np.cross(f_b, w_b)
    norm_cross_b = np.linalg.norm(cross_b)
    if norm_cross_b < 1e-12:
        norm_cross_b = 1e-12
    u2_b = cross_b / norm_cross_b
    
    u3_b = np.cross(u1_b, u2_b)
    
    # 5. 组装正交基矩阵，求解 C_b_n = M_n * M_b^T
    M_n = np.column_stack((u1_n, u2_n, u3_n))
    M_b = np.column_stack((u1_b, u2_b, u3_b))
    C_b_n = M_n @ M_b.T
    
    # 6. 从 C_b_n 中提取欧拉角 (Pitch, Roll, Yaw)
    # C_b_n = [ C11 C12 C13
    #           C21 C22 C23
    #           C31 C32 C33 ]
    # pitch = arcsin(-C31)
    # roll = arctan2(C32, C33)
    # yaw = arctan2(C21, C11)
    
    C31 = np.clip(C_b_n[2, 0], -1.0, 1.0)
    pitch_rad = np.arcsin(-C31)
    roll_rad = np.arctan2(C_b_n[2, 1], C_b_n[2, 2])
    yaw_rad = np.arctan2(C_b_n[1, 0], C_b_n[0, 0])
    
    # 弧度转为角度
    pitch = pitch_rad * RAD_TO_DEG
    roll = roll_rad * RAD_TO_DEG
    yaw = yaw_rad * RAD_TO_DEG
    
    # 航向角映射到 [0, 360] 度
    if yaw < 0:
        yaw += 360.0
        
    return pitch, roll, yaw, C_b_n

class CoarseAligner:
    """
    静态解析粗对准器
    """
    def __init__(self, imu_data):
        self.imu_data = imu_data  # 包含 'time', 'acc', 'gyr' 的字典

    def align_whole_average(self):
        """
        方式1：整段静态数据平均后计算一次姿态角
        """
        mean_acc = np.mean(self.imu_data['acc'], axis=0)
        mean_gyr = np.mean(self.imu_data['gyr'], axis=0)
        
        pitch, roll, yaw, C_b_n = solve_attitude_from_vectors(mean_acc, mean_gyr)
        
        print("\n=== 整段数据平均对准结果 ===")
        print(f"俯仰角 (Pitch): {pitch:.6f}°")
        print(f"横滚角 (Roll) : {roll:.6f}°")
        print(f"航向角 (Yaw)  : {yaw:.6f}°")
        print("C_b_n 姿态矩阵:")
        print(C_b_n)
        return pitch, roll, yaw, C_b_n

    def align_per_second(self):
        """
        方式2：每秒平均值计算姿态角
        返回: times, pitches, rolls, yaws
        """
        times_all = self.imu_data['time']
        acc_all = self.imu_data['acc']
        gyr_all = self.imu_data['gyr']
        
        # 按照秒进行分组 (每一秒 100 个历元)
        epoch_per_sec = 100
        N = len(times_all)
        num_secs = N // epoch_per_sec
        
        times = []
        pitches = []
        rolls = []
        yaws = []
        
        for i in range(num_secs):
            idx_start = i * epoch_per_sec
            idx_end = (i + 1) * epoch_per_sec
            
            mean_acc = np.mean(acc_all[idx_start:idx_end], axis=0)
            mean_gyr = np.mean(gyr_all[idx_start:idx_end], axis=0)
            
            p, r, y, _ = solve_attitude_from_vectors(mean_acc, mean_gyr)
            
            times.append(times_all[idx_start])
            pitches.append(p)
            rolls.append(r)
            yaws.append(y)
            
        return np.array(times), np.array(pitches), np.array(rolls), np.array(yaws)

    def align_per_epoch(self):
        """
        方式3：每历元计算姿态角
        返回: times, pitches, rolls, yaws
        """
        times_all = self.imu_data['time']
        acc_all = self.imu_data['acc']
        gyr_all = self.imu_data['gyr']
        
        N = len(times_all)
        pitches = np.zeros(N)
        rolls = np.zeros(N)
        yaws = np.zeros(N)
        
        for i in range(N):
            p, r, y, _ = solve_attitude_from_vectors(acc_all[i], gyr_all[i])
            pitches[i] = p
            rolls[i] = r
            yaws[i] = y
            
        return times_all, pitches, rolls, yaws
