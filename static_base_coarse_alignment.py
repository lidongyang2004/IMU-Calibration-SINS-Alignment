import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import math
from pyins import earth, transform, util, filters, measurements, inertial_sensor, strapdown

# ==========================================
# 1. 加载 PSINS 真实数据并转换坐标系
# ==========================================
file_path = 'data/ts_static.mat'
print(f"--- 正在加载数据: {file_path} ---")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"找不到文件：{file_path}，请检查文件夹名称是否正确。")

data_mat = loadmat(file_path)
dd = data_mat['dd']  

# 试验参数设置
fs = 400.0           # 采样频率 400Hz
dt = 1 / fs
n_samples = len(dd)
times = np.arange(n_samples) * dt

# 【核心转换 1】：增量转速率
gyro_rate_psins = dd[:, 0:3] / dt
accel_rate_psins = dd[:, 3:6] / dt

# 【核心转换 2】：坐标系映射 (PSINS 右-前-上 -> pyins 前-右-下)
gyro_rate_frd = np.column_stack((gyro_rate_psins[:, 1], gyro_rate_psins[:, 0], -gyro_rate_psins[:, 2]))
accel_rate_frd = np.column_stack((accel_rate_psins[:, 1], accel_rate_psins[:, 0], -accel_rate_psins[:, 2]))

imu_data = np.hstack((gyro_rate_frd, accel_rate_frd))

# 构建 pyins 标准的 IMU DataFrame
imu = pd.DataFrame(
    imu_data, 
    index=times, 
    columns=util.GYRO_COLS + util.ACCEL_COLS
)

# 初始地理位置 (PSINS 示例: 29.7061, 115.96, 16)
lat, lon, alt = 29.7061, 115.96, 16.0

# ==========================================
# 2. 粗对准 (Coarse Alignment - 双矢量定姿法)
# ==========================================
coarse_duration = 300.00
n_coarse = int(coarse_duration * fs)
imu_avg = imu.iloc[:n_coarse].mean()

# 提取比力和角速度平均值
f_b = imu_avg[util.ACCEL_COLS].values
w_b = imu_avg[util.GYRO_COLS].values

# --- 严格遵循 对准算法的逻辑求解 ---
gravity = 9.7936174        # 重力加速度常数
we = 7.292115e-5           # 地球自转角速度 (rad/s)
fai = np.deg2rad(lat)      # 纬度转弧度

# 1. 构造 b 系下的观测向量
gb = -f_b                  # 加速度计输出为 -f，故反号
wb = w_b
vb = np.cross(gb, wb)      # 构造第三正交向量

# 2. 构造 n 系下的基准输入向量 (NED 系)
gn = np.array([0.0, 0.0, gravity])
wn = np.array([we * math.cos(fai), 0.0, -we * math.sin(fai)])
vn = np.cross(gn, wn)

# 3. 向量正交化与归一化
wg = gb / np.linalg.norm(gb)
ww = vb / np.linalg.norm(vb)
wgw = np.cross(vb, gb)
wgw = wgw / np.linalg.norm(wgw)
W = np.column_stack((wg, ww, wgw))

vg = gn / np.linalg.norm(gn)
vw = vn / np.linalg.norm(vn)
vgw = np.cross(vn, gn)
vgw = vgw / np.linalg.norm(vgw)
V = np.column_stack((vg, vw, vgw))

# 4. 求解姿态矩阵 Cbn
Cbn = V @ W.T

# 5. 从姿态矩阵中提取欧拉角
C31, C32, C33 = Cbn[2, 0], Cbn[2, 1], Cbn[2, 2]
C21, C11 = Cbn[1, 0], Cbn[0, 0]

est_pitch = math.degrees(math.atan(-C31 / math.sqrt(C32**2 + C33**2)))
est_heading = math.degrees(math.atan2(C21, C11))
est_roll = math.degrees(math.atan2(C32, C33))

# 【修改点 1】：将粗对准航向角映射到 -180度 到 180度 范围内
est_heading = (est_heading + 180) % 360 - 180

print(f"双矢量解析粗对准结果: Roll={est_roll:.4f}°, Pitch={est_pitch:.4f}°, Heading={est_heading:.4f}°")

# 构造初始 PVA 序列
coarse_pva = pd.Series(
    [lat, lon, alt, 0.0, 0.0, 0.0, est_roll, est_pitch, est_heading],
    index=util.TRAJECTORY_COLS,
    name=times[0]  
)

# 保存对准结果到 results/ 目录
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
report_path = os.path.join(output_dir, "static_base_coarse_alignment_result.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("============================================================\n")
    f.write("              pyins 静态基座双矢量粗对准结算报告\n")
    f.write("============================================================\n\n")
    f.write(f"数据源文件: {file_path}\n")
    f.write("实验参数:\n")
    f.write(f"  采样频率: {fs} Hz\n")
    f.write(f"  积分对准时长: {coarse_duration} 秒\n")
    f.write(f"  初始地理位置: 纬度={lat}°, 经度={lon}°, 高度={alt}m\n\n")
    f.write("对准姿态结算结果 (欧拉角 NED-FRD):\n")
    f.write(f"  横滚角 (Roll)   : {est_roll:.6f}°\n")
    f.write(f"  俯仰角 (Pitch)  : {est_pitch:.6f}°\n")
    f.write(f"  航向角 (Heading): {est_heading:.6f}° (已映射至 [-180°, 180°])\n\n")
    f.write("方向余弦姿态矩阵 C_b_n:\n")
    f.write(np.array2string(Cbn, formatter={'float_kind': lambda x: f"{x:12.8f}"}) + "\n")

print(f"静态基座粗对准结果已保存至: {report_path}")

