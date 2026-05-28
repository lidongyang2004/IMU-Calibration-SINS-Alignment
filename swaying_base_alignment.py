import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import os
from pyins import earth, transform, util, filters, measurements, inertial_sensor, strapdown

# ==========================================
# 1. 加载 PSINS 摇摆基座数据并转换坐标系
# ==========================================
file_path = 'data/ts_imo_rot.mat'
print(f"--- 正在加载摇摆基座数据: {file_path} ---")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"找不到文件：{file_path}，请检查文件路径。")

data_mat = loadmat(file_path)
dd = data_mat['dd']  

# 试验参数设置 (400Hz 激光陀螺)
fs = 400.0           
dt = 1 / fs
n_samples = len(dd)
times = np.arange(n_samples) * dt

# 增量转速率，并完成 PSINS 到 pyins 的坐标系映射 (右-前-上 -> 前-右-下)
gyro_rate_psins = dd[:, 0:3] / dt
accel_rate_psins = dd[:, 3:6] / dt

gyro_rate_frd = np.column_stack((gyro_rate_psins[:, 1], gyro_rate_psins[:, 0], -gyro_rate_psins[:, 2]))
accel_rate_frd = np.column_stack((accel_rate_psins[:, 1], accel_rate_psins[:, 0], -accel_rate_psins[:, 2]))

imu_data = np.hstack((gyro_rate_frd, accel_rate_frd))

# 构建 pyins 标准的 IMU DataFrame
imu = pd.DataFrame(
    imu_data, 
    index=times, 
    columns=util.GYRO_COLS + util.ACCEL_COLS
)

lat, lon, alt = 29.7061, 115.96, 16.0

print("正在计算 IMU 严密增量...")
increments = strapdown.compute_increments_from_imu(imu, 'rate')

# ==========================================
# 2. 严密凝固惯性系粗对准 (Inertial Frame Coarse Alignment)
# ==========================================
print("正在执行严密凝固惯性系双矢量定姿 (TRIAD)...")
coarse_duration = 30.0  # 积分时间：30秒
n_coarse = int(coarse_duration * fs)
coarse_increments = increments.iloc[:n_coarse]

# --- 理论参考参数计算 ---
lat_rad = np.deg2rad(lat)
omega_ie = earth.RATE  # 地球自转角速度
# 导航系(NED)下的地球自转矢量
wie_n = np.array([omega_ie * np.cos(lat_rad), 0.0, -omega_ie * np.sin(lat_rad)])
# 导航系(NED)下的理论比力 (重力向下为正，静止比力向上，故为负)
g0 = earth.gravity(lat, alt)
f_n = np.array([0.0, 0.0, -g0])

# --- 初始化方向余弦矩阵 ---
# C_b_ib0: 从当前载体系(b)到起始惯性系(ib0)的转移矩阵
C_b_ib0 = np.eye(3)
# C_n_in0: 从当前导航系(n)到起始导航惯性系(in0)的转移矩阵
C_n_in0 = np.eye(3)

# 观测矢量与理论参考矢量 (分前后两段积分以构造不共线向量)
mid_point = n_coarse // 2
v1_ib0, v2_ib0 = np.zeros(3), np.zeros(3)  # 实际观测比力积分
w1_in0, w2_in0 = np.zeros(3), np.zeros(3)  # 理论参考比力积分

for i in range(n_coarse):
    theta = coarse_increments.iloc[i][util.THETA_COLS].values
    dv = coarse_increments.iloc[i][util.DV_COLS].values
    
    # 步骤一：隔离角运动 (更新 C_b^{ib0})
    # 注意：传感器输出的是 b_{t-1} 到 b_t 的旋转，累乘得到 b_t 到 ib0(b_0) 的关系
    delta_C_b = Rotation.from_rotvec(theta).as_matrix()
    C_b_ib0 = C_b_ib0 @ delta_C_b
    
    # 步骤二：计算地球自转 (更新 C_n^{in0})
    delta_C_n = Rotation.from_rotvec(wie_n * dt).as_matrix()
    C_n_in0 = C_n_in0 @ delta_C_n
    
    # 将实际速度增量投影到 ib0 系，理论比力积分投影到 in0 系
    dv_ib0 = C_b_ib0 @ dv
    dw_in0 = C_n_in0 @ (f_n * dt)
    
    # 步骤三：比力积分平滑线运动噪声
    if i < mid_point:
        v1_ib0 += dv_ib0
        w1_in0 += dw_in0
    else:
        v2_ib0 += dv_ib0
        w2_in0 += dw_in0

# 步骤四：双矢量定姿 (TRIAD 算法求解常值姿态矩阵 C_{ib0}^{in0})
def normalize(v):
    return v / np.linalg.norm(v)

# 构造观测正交基
r1 = normalize(v1_ib0)
r2 = normalize(np.cross(v1_ib0, v2_ib0))
r3 = np.cross(r1, r2)
R_ib0 = np.column_stack((r1, r2, r3))

# 构造理论正交基
s1 = normalize(w1_in0)
s2 = normalize(np.cross(w1_in0, w2_in0))
s3 = np.cross(s1, s2)
S_in0 = np.column_stack((s1, s2, s3))

# 求解常值矩阵
C_ib0_in0 = S_in0 @ R_ib0.T

# t=0 时刻的初始姿态即为常值矩阵 (因为 t=0 时，两个凝固系分别与载体/导航系重合)
Cb0_n0 = C_ib0_in0

# 解析出欧拉角 (pyins 的 NED 系对应 ZYX 旋转顺序：航向-俯仰-横滚)
heading, pitch, roll = Rotation.from_matrix(Cb0_n0).as_euler('ZYX', degrees=True)
heading = heading % 360.0  # 将航向约束到 0~360度

# 构造 t=0 时刻的初始 PVA
coarse_pva = pd.Series(
    [lat, lon, alt, 0.0, 0.0, 0.0, roll, pitch, heading],
    index=util.TRAJECTORY_COLS,
    name=times[0]  
)

print(f"TRIAD 粗对准解析结果(t=0): Roll={roll:.4f}°, Pitch={pitch:.4f}°, Heading={heading:.4f}°")
print("注：仅依靠30秒积分求得的航向误差可能在几度甚至十几度（受限于地球自转的微小角度），需要KF进一步收敛。\n")


# ==========================================
# 3. 精对准 (Fine Alignment - KF)
# ==========================================
# 调整不确定度配置：粗对准的水平角较准，但航向方差需给足
pos_sd = 1.0         
velocity_sd = 0.1      
level_sd = 0.5       
azimuth_sd = 15.0  # TRIAD 在30秒内的航向解析结果较粗，增大初始方差赋予滤波器更大的修正权重   

# 观测值：静止基座零速修正 (ZUPT)
obs_times = times[::int(fs/2)]  
v_obs_data = pd.DataFrame(0.0, index=obs_times, columns=['VN', 'VE', 'VD'])
v_measurement = measurements.NedVelocity(v_obs_data, sd=0.05) 

# --- 配置 0.01°/h 激光陀螺误差模型 ---
gyro_bias_sd = np.deg2rad(0.01) / 3600  
accel_bias_sd = 20e-6 * 9.81            

gyro_model = inertial_sensor.EstimationModel(bias_sd=gyro_bias_sd, noise=1e-7)
accel_model = inertial_sensor.EstimationModel(bias_sd=accel_bias_sd, noise=1e-5)

print("正在运行摇摆基座卡尔曼滤波精对准 (可能需要几秒钟)...")

# 滤波器从 t=0 开始，结合粗对准初值，贯穿整个过程进行状态估计
result = filters.run_feedback_filter(
    coarse_pva, 
    pos_sd, velocity_sd, level_sd, azimuth_sd,
    increments, 
    measurements=[v_measurement],
    gyro_model=gyro_model,
    accel_model=accel_model
)

# ==========================================
# 4. 结果分析与可视化
# ==========================================
res = result.trajectory.iloc[-1]
print(f"\n--- 摇摆对准最终结果 (KF平滑后) ---")
print(f"Roll: {res['roll']:.4f}° | Pitch: {res['pitch']:.4f}° | Heading: {res['heading']:.4f}°")

# 保存对准结果到 results/ 目录
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
report_path = os.path.join(output_dir, "swaying_base_alignment_result.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("============================================================\n")
    f.write("         pyins 摇摆基座粗对准与卡尔曼滤波精对准结算报告\n")
    f.write("============================================================\n\n")
    f.write(f"数据源文件: {file_path}\n")
    f.write("实验参数:\n")
    f.write(f"  采样频率: {fs} Hz\n")
    f.write(f"  初始地理位置: 纬度={lat}°, 经度={lon}°, 高度={alt}m\n\n")
    f.write("1. 凝固惯性系双矢量 TRIAD 粗对准结果 (t=0 时刻初值):\n")
    f.write(f"  横滚角 (Roll)   : {roll:.6f}°\n")
    f.write(f"  俯仰角 (Pitch)  : {pitch:.6f}°\n")
    f.write(f"  航向角 (Heading): {heading:.6f}°\n\n")
    f.write("初始方向余弦矩阵 C_b0_n0:\n")
    f.write(np.array2string(Cb0_n0, formatter={'float_kind': lambda x: f"{x:12.8f}"}) + "\n\n")
    f.write("2. 摇摆基座卡尔曼滤波 (ZUPT KF) 精对准最终收敛结果:\n")
    f.write(f"  横滚角 (Roll)   : {res['roll']:.6f}°\n")
    f.write(f"  俯仰角 (Pitch)  : {res['pitch']:.6f}°\n")
    f.write(f"  航向角 (Heading): {res['heading']:.6f}°\n")

print(f"摇摆基座对准结算结果已保存至: {report_path}")

# 优化图表文字大小与布局以提升可读性
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # 保证中英文字体渲染正常
plt.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(10, 8))

# 绘制航向角收敛 (从 TRIAD 解析的初始粗航向逐渐收敛)
plt.subplot(2, 1, 1)
plt.plot(result.trajectory.index, result.trajectory['heading'], 'r', label='Estimated Heading (KF)')
plt.axhline(res['heading'], color='k', linestyle='--', label='Final Converged Heading')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Heading (Deg)', fontsize=14)
plt.title('Heading Convergence Under Swaying Base', fontsize=15, fontweight='bold')
plt.legend(fontsize=12)
plt.tick_params(labelsize=12)

# 绘制水平角验证
plt.subplot(2, 1, 2)
plt.plot(result.trajectory.index, result.trajectory['roll'], 'b', label='Roll (Dynamic)')
plt.plot(result.trajectory.index, result.trajectory['pitch'], 'g', label='Pitch (Dynamic)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Attitude (Deg)', fontsize=14)
plt.title('Dynamic Roll and Pitch Tracking', fontsize=15, fontweight='bold')
plt.legend(fontsize=12)
plt.tick_params(labelsize=12)

plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/swaying_base_alignment.png", dpi=300)
print("摇摆基座对准姿态曲线已保存至: images/swaying_base_alignment.png")
plt.show()