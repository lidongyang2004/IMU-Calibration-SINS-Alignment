import os
import numpy as np
from imu_calib.constants import GS_XWGI, AS_XWGI, SAMPLE_RATE

def parse_imu_line(line):
    """
    解析单行 IMU 数据
    格式为: %RAWIMUSA,week,sec;week,sec,status,acc_z,acc_y,acc_x,gyr_z,gyr_y,gyr_x*checksum
    返回: dict containing raw values and physical values (acc in m/s^2, gyr in rad/s)
    """
    line = line.strip()
    if not line or ';' not in line:
        return None
    try:
        _, data_part = line.split(';', 1)
        if '*' in data_part:
            data_part = data_part.split('*', 1)[0]
        
        fields = data_part.split(',')
        
        # 解析时间和状态
        week = int(fields[0])
        second = float(fields[1])
        time_sec = week * 604800 + second
        status = int(fields[2], 16) if fields[2].startswith('0x') else int(fields[2])
        
        # 解析原始输出数据 (注意对应的顺序: Z, Y, X)
        raw_az = float(fields[3])
        raw_ay = float(fields[4])
        raw_ax = float(fields[5])
        
        raw_gz = float(fields[6])
        raw_gy = float(fields[7])
        raw_gx = float(fields[8])
        
        # 转换为比力和角速度值 (乘以比例因子和采样率得到物理单位)
        # 注意: Y 轴根据硬件手册和原C++代码需取反
        val_az = raw_az * AS_XWGI * SAMPLE_RATE
        val_ay = -raw_ay * AS_XWGI * SAMPLE_RATE
        val_ax = raw_ax * AS_XWGI * SAMPLE_RATE
        
        val_gz = raw_gz * GS_XWGI * SAMPLE_RATE
        val_gy = -raw_gy * GS_XWGI * SAMPLE_RATE
        val_gx = raw_gx * GS_XWGI * SAMPLE_RATE
        
        return {
            'time': time_sec,
            'status': status,
            'raw_acc': np.array([raw_ax, raw_ay, raw_az]),
            'raw_gyr': np.array([raw_gx, raw_gy, raw_gz]),
            'acc': np.array([val_ax, val_ay, val_az]),  # [ax, ay, az] (m/s^2)
            'gyr': np.array([val_gx, val_gy, val_gz])   # [gx, gy, gz] (rad/s)
        }
    except (ValueError, IndexError):
        return None

def read_imu_data(file_path):
    """
    从数据文件中读取所有 IMU 历元的数据
    返回: dict of numpy arrays
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"IMU 数据文件未找到: {file_path}")
        
    times = []
    statuses = []
    accs = []
    gyrs = []
    raw_accs = []
    raw_gyrs = []
    
    with open(file_path, 'r') as f:
        for line in f:
            epoch = parse_imu_line(line)
            if epoch is not None:
                times.append(epoch['time'])
                statuses.append(epoch['status'])
                accs.append(epoch['acc'])
                gyrs.append(epoch['gyr'])
                raw_accs.append(epoch['raw_acc'])
                raw_gyrs.append(epoch['raw_gyr'])
                
    return {
        'time': np.array(times),
        'status': np.array(statuses),
        'acc': np.array(accs),        # Nx3, m/s^2
        'gyr': np.array(gyrs),        # Nx3, rad/s
        'raw_acc': np.array(raw_accs), # Nx3, raw LSB
        'raw_gyr': np.array(raw_gyrs)  # Nx3, raw LSB
    }
