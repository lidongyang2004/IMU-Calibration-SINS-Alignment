import numpy as np

# 圆周率及弧度/度转换系数
PAI = 3.141592653589793
DEG_TO_RAD = PAI / 180.0
RAD_TO_DEG = 180.0 / PAI

# XW-GI7681 IMU 比例因子与采样率
GS_XWGI = 1.0850694444e-7      # 陀螺仪比例因子
AS_XWGI = 1.5258789063e-6      # 加速度计比例因子
SAMPLE_RATE = 100              # 采样率 100Hz
DT = 1.0 / SAMPLE_RATE         # 采样间隔 (s)

# 地球与转台物理常数
GRAVITY = 9.7936174            # 武汉实验室重力加速度 (m/s^2)
WE = 7.292115e-5 * RAD_TO_DEG  # 地球自转角速度 (deg/s)
FAI = 30.531651244 * DEG_TO_RAD # 武汉转台实验室纬度 (rad)

# 陀螺仪标定积分参数
READ_TIME = 40.00              # 积分读取时长 (s)
INCREMENT_THRESHOLD = 0.00018  # 积分开启的角度积分增量阈值 (deg)
CIRCLE_DEG = 360.0             # 一整圈度数
