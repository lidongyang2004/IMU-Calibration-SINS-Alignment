# IMU-Calibration-Alignment
**IMU 误差标定与捷联惯导静态解析粗对准算法平台**

本开源项目是为测绘、导航、航天航空及自动控制相关专业设计的捷联惯性测量单元（IMU）误差参数标定、数据误差补偿以及捷联惯导（SINS）静态解析粗对准的一站式算法平台。

基于真实高精度三轴转台采集的 **XW-GI7681** 惯导原始数据，本项目使用 Python 实现了学术级/工业级的完整标定编排与对准结算流程。本平台所使用的全部物理模型与公式严格遵循惯性导航学科的经典教科书与主流科研论文规范。

---

## 🌟 核心特色与算法功能

1. **加速度计静态六位置标定 (Static Six-Position Calibration)**
   - 使用经典的最小二乘（Least Squares）算法，估计加速度计的 **3个零偏（Bias）**、**3个比例因子误差（Scale Factor）** 以及 **6个交轴耦合误差（Cross-axis Coupling）**（共12个齐次参数）。
   - 通过对 X/Y/Z 三轴分别向上和向下静置 3 分钟的静态数据进行白噪声抑制与高精密均值提取，实现超低随机噪声干扰的精密参数反演。

2. **陀螺仪角位置动态标定 (Angular Position Calibration)**
   - 使用角位置积分法标定陀螺仪。通过主轴旋转 360° 正转与反转的角度差分，有效消除地球自转以及零偏在积分时段内的线性漂移误差。
   - 估计陀螺仪的 **3个零偏**、**3个比例因子误差** 以及 **6个交轴耦合误差**（包含非对角耦合项的提取）。
   - 实现了基于积分角度阈值触发的同步积分启动算法，确保三轴梯形积分区间在时域上的严格一致性。

3. **IMU 传感器级数据误差补偿 (IMU Error Compensation)**
   - 实现了严密的传感器级逆变换物理补偿模型：
     $$\tilde{\boldsymbol{f}} = (\boldsymbol{I} + \boldsymbol{S}_a + \boldsymbol{N}_a)^{-1} (\boldsymbol{f}^{raw} - \boldsymbol{b}_a)$$
     $$\tilde{\boldsymbol{\omega}} = (\boldsymbol{I} + \boldsymbol{S}_g + \boldsymbol{N}_g)^{-1} (\boldsymbol{w}^{raw} - \boldsymbol{b}_{g})$$
   - 能够对离线大数据集进行超高效的历元级批量补偿，彻底消除零偏、标度误差及安装交轴耦合引起的系统偏差。

4. **SINS 静态解析粗对准算法 (SINS Static Analytical Coarse Alignment)**
   - 支持**北-东-地 (NED) 导航坐标系**与**前-右-下 (FRD) 载体系**的物理矢量精密投影变换。
   - 基于重力比力 $\boldsymbol{f}^n = [0, 0, -g]^T$ 与地球自转角速度 $\boldsymbol{w}_{ie}^n = [\omega_e \cos\varphi, 0, -\omega_e \sin\varphi]^T$ 在两系下的不共线双矢量定姿算法，基于基变换直接计算方向余弦姿态矩阵 $\boldsymbol{C}_b^n$。
   - 提供多尺度的姿态结算与波动分析：
     - **整段时间平均对准**：滤除所有中高频噪声，获得理论极致初始姿态。
     - **滑动秒级平均对准**：评估微小外界扰动和中低频晃动下的姿态稳定性，输出 Roll/Pitch/Yaw 随时间的变化曲线。
     - **历元级单点对准**：直接暴露传感器高频白噪声和振动，分析定姿结果的标准差与不确定度。

5. **基于 pyins 的高级初始对准与卡尔曼滤波精对准 (Advanced Alignment & KF Fine Alignment)**
   - **静态基座双矢量粗对准**：基于高频静置数据 `ts_static.mat`，引入 `pyins` 的地球自转与重力物理模型进行双矢量解析粗对准，完成高频历元姿态结算。
   - **摇摆基座凝固惯性系粗对准**：针对摇摆晃动基座场景，设计并实现了**凝固惯性系双矢量定姿算法**。将运动过程中的速度增量和理论参考比力分别积分并投影到初始惯性系与导航系中，利用 TRIAD 算法在强晃动或行船晃动中精确分离出姿态矩阵。
   - **零速卡尔曼滤波反馈精对准 (ZUPT KF Fine Alignment)**：结合粗对准欧拉角初值，构建了包含位置误差、速度误差、姿态角误差、陀螺三轴零偏和加速度计三轴零偏在内的 **15 维惯导状态空间反馈滤波器**，引入零速修正（ZUPT）观测量进行卡尔曼滤波收敛计算，实现了航向角（Yaw）在高随机晃动下的精密极速收敛。

---

## 📈 算法物理数学公式说明

### 1. 加速度计静态六位置标定模型
加速度计测量的齐次形式为：
$$\boldsymbol{f}^{raw} = \boldsymbol{M} \begin{bmatrix} \boldsymbol{f}^{real} \\ 1 \end{bmatrix}$$
其中，$\boldsymbol{M} = [\boldsymbol{I} + \boldsymbol{S}_a + \boldsymbol{N}_a \mid \boldsymbol{b}_a]$ 是一个 $3 \times 4$ 的误差映射矩阵。
控制转台使得三轴分别向上和向下静置，总共采集 6 个位置的数据。将所有 6 个位置下的测量比力均值 $\boldsymbol{f}^{raw}_i$ 组成比力均值矩阵 $\boldsymbol{F}^{raw} \in \mathbb{R}^{3 \times 6}$，已知输入真实比力矩阵 $\boldsymbol{F}^{real} \in \mathbb{R}^{4 \times 6}$。
利用最小二乘法求解映射矩阵 $\boldsymbol{M}$：
$$\boldsymbol{M} = \boldsymbol{F}^{raw} (\boldsymbol{F}^{real})^T \left[ \boldsymbol{F}^{real} (\boldsymbol{F}^{real})^T \right]^{-1}$$
解出 $\boldsymbol{M}$ 后，提取第 4 列即为零偏向量 $\boldsymbol{b}_a$，左侧 $3 \times 3$ 矩阵减去单位阵后，对角线元素为比例因子误差 $\boldsymbol{S}_a$，非对角线元素为交轴耦合误差 $\boldsymbol{N}_a$。

### 2. 陀螺仪两位置角位置标定模型
控制转台绕主敏感轴正转 $\theta_1 = +360^\circ$ 和反转 $\theta_2 = -360^\circ$，记录积分角度与用时。
- **主轴陀螺零偏 $\boldsymbol{b}_g$（deg/h）**：
  $$b_g = \left( \frac{\alpha_{pos} + \alpha_{neg}}{2 \cdot t_{avg}} - \omega_e \sin\varphi \right) \cdot 3600$$
- **主轴比例因子误差 $S_g$**：
  $$S_g = \frac{|\alpha_{pos} - \alpha_{neg}|}{2 \cdot 360^\circ} - 1.0$$
- **交轴耦合误差 $N_{g,ij}$**：
  $$N_{g,ij} = \frac{\alpha_{pos, i} - \alpha_{neg, i}}{2 \cdot 360^\circ}$$

### 3. 双矢量解析粗对准定姿
在静态条件下，定义 $n$ 系和 $b$ 系下的正交基：
$$\boldsymbol{u}_1^n = \frac{\boldsymbol{f}^n}{\|\boldsymbol{f}^n\|}, \quad \boldsymbol{u}_2^n = \frac{\boldsymbol{f}^n \times \boldsymbol{w}_{ie}^n}{\|\boldsymbol{f}^n \times \boldsymbol{w}_{ie}^n\|}, \quad \boldsymbol{u}_3^n = \boldsymbol{u}_1^n \times \boldsymbol{u}_2^n$$
$$\boldsymbol{u}_1^b = \frac{\boldsymbol{f}^b}{\|\boldsymbol{f}^b\|}, \quad \boldsymbol{u}_2^b = \frac{\boldsymbol{f}^b \times \boldsymbol{w}_{ie}^b}{\|\boldsymbol{f}^b \times \boldsymbol{w}_{ie}^b\|}, \quad \boldsymbol{u}_3^b = \boldsymbol{u}_1^b \times \boldsymbol{u}_2^b$$
方向余弦姿态矩阵 $\boldsymbol{C}_b^n$ 唯一确定为：
$$\boldsymbol{C}_b^n = [\boldsymbol{u}_1^n, \boldsymbol{u}_2^n, \boldsymbol{u}_3^n] \cdot [\boldsymbol{u}_1^b, \boldsymbol{u}_2^b, \boldsymbol{u}_3^b]^T$$
根据方向余弦矩阵，即可严格解析反解出欧拉姿态角：
- 俯仰角：$\theta = \arcsin(-C_{31})$
- 横滚角：$\phi = \text{arctan2}(C_{32}, C_{33})$
- 航向角：$\psi = \text{arctan2}(C_{21}, C_{11})$，并映射到 $[0^\circ, 360^\circ]$ 范围。

---

## 📂 项目目录结构

```
IMU-Calibration-Alignment/
│
├── imu_calib/                  # 核心算法 Python 模块包
│   ├── __init__.py             # 包初始化文件
│   ├── constants.py            # 全局物理、地理及传感器常数配置
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── io.py               # 高精密 IMU 数据文本流解析器
│   │
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── accelerometer.py    # 加速度计静态六位置标定估计器
│   │   └── gyroscope.py        # 陀螺仪角位置动态标定积分估计器
│   │
│   ├── compensation/
│   │   ├── __init__.py
│   │   └── compensate.py       # IMU 历元级/数据集多轴标定误差物理补偿
│   └── alignment/
│       ├── __init__.py
│       └── coarse_align.py     # 静态双矢量定姿粗对准计算核心 (NED-FRD)
│
├── data/
│   ├── Calibration/            # IMU 标定与静态原始 ASC 数据文件夹
│   ├── ts_static.mat           # SINS 静态基座对准高频 .mat 数据集
│   └── ts_imo_rot.mat          # SINS 摇摆基座对准高频 .mat 数据集
│
├── results/                    # 算法输出的所有标定矩阵、均值对比和粗对准报告
│   ├── AccCaliResult_Matrix.txt  # 加速度计标定矩阵
│   ├── GyrCaliResult_Matrix.txt  # 陀螺仪标定矩阵
│   ├── coarse_alignment_report.txt  # 静态粗对准实验详细分析报告
│   └── z_up_3min_compensated.txt  # 补偿后的精密 IMU 数据集
│
├── images/                     # 自动生成的算法可视化结果图
│   ├── compensation_comparison.png  # 补偿前后比力/角速度波动对比图
│   ├── attitude_curves.png          # 粗对准不同尺度姿态角-时间曲线图
│   └── swaying_base_alignment.png   # 摇摆基座粗对准+卡尔曼滤波精对准姿态角收敛曲线图
│
├── main.py                     # 全自动化运行的一键式主测试平台入口
├── Align.py                    # 独立运行的静态解析粗对准实验程序
├── static_base_coarse_alignment.py  # 基于 pyins 的静态基座双矢量粗对准测试程序
├── swaying_base_alignment.py        # 基于 pyins 的摇摆基座凝固惯性系粗对准与卡尔曼滤波精对准程序
├── acc_Calibration.py          # 兼容原版的独立加速度计标定脚本
├── Gyro_Calibration.py         # 兼容原版的独立陀螺仪标定脚本
├── requirements.txt            # 项目 Python 依赖项
├── .gitignore                  # Git 忽略配置
├── README.md                   # 项目中文主说明文档
```

---

## 🛠️ 安装与快速上手

### 1. 准备 Python 运行环境
本项目对三轴高频数据执行高精密计算及绘图，需要安装 `numpy`、`pandas`、`matplotlib`、`scipy` 以及惯性导航专用算法库 `pyins`。

在项目根目录下，运行终端命令一键安装所有依赖项：
```bash
pip install -r requirements.txt
```

### 2. 运行一键式全自动化测试平台 (`main.py`)
主控脚本 `main.py` 会全自动按顺序执行**加速度计静态六位置标定**、**陀螺仪角位置动态标定**、**特定测试数据集批量物理误差补偿**、**补偿前后多尺度双矢量粗对准定姿解算**，并将所有生成的可视化曲线与分析报告保存至本地：
```bash
python main.py
```

### 3. 运行静态解析粗对准实验独立程序 (`Align.py`)
如果您只需运行静态解析双矢量粗对准并绘制滑动姿态曲线，可直接运行：
```bash
python Align.py
```

### 4. 运行基于 pyins 的静态基座粗对准独立程序 (`static_base_coarse_alignment.py`)
运行基于高精密惯导库 `pyins` 编写的静态基座双矢量定姿解算，计算 `ts_static.mat` 数据集前 300 秒的平均姿态欧拉角：
```bash
python static_base_coarse_alignment.py
```

### 5. 运行基于 pyins 的摇摆基座粗对准与卡尔曼滤波精对准程序 (`swaying_base_alignment.py`)
运行摇摆基座对准算法，首先在强晃动基座下采用**凝固惯性系双矢量 TRIAD 定姿算法**计算 $t=0$ 刻初始姿态，接着一键启动 **15 维状态零速修正卡尔曼滤波器（ZUPT Feedback KF）**进行精对准姿态解算。它将自动收敛姿态角，绘制收敛曲线，并自动将图表保存至 `images/swaying_base_alignment.png`：
```bash
python swaying_base_alignment.py
```

---

## 📊 标定与粗对准输出实例展示

### 1. 加速度计标定零偏与矩阵结果 ($b_a, S_a, N_a$)
```
=== ACCELEROMETER CALIBRATION MATRICES ===

1. Bias Vector b_a (m/s^2):
[[-0.00943451],
 [ 0.00142041],
 [-0.01676003]]

2. Scale Factor Error Matrix S_a:
[[0.00026483, 0.        , 0.        ],
 [0.        , 0.00038355, 0.        ],
 [0.        , 0.        , 0.00044563]]

3. Cross-axis Coupling Matrix N_a:
[[ 0.        ,  0.0039051 , -0.00799005],
 [-0.00409682,  0.        ,  0.0004882 ],
 [ 0.0083232 , -0.00042507,  0.        ]]
```

### 2. 双矢量静态解析粗对准姿态角解算对比 (Z轴向上静态数据)
- **使用未标定的原始数据进行双矢量解析定姿：**
  - 俯仰角 (Pitch): $0.032471^\circ$
  - 横滚角 (Roll) : $0.518837^\circ$
  - 航向角 (Yaw)  : $89.928183^\circ$
- **使用经过标定参数物理误差补偿后的数据定姿：**
  - 俯仰角 (Pitch): $-0.003870^\circ$ (完美收敛在零点附近)
  - 横滚角 (Roll) : $0.005188^\circ$ (完美收敛在零点附近)
  - 航向角 (Yaw)  : $89.906935^\circ$
- **数据补偿带来的提升分析：**
  在误差补偿之前，受加速度计三轴零偏和交轴耦合影响，解算出来的 Pitch 和 Roll 偏离了理想状态（该位置下 Roll 和 Pitch 应极为接近 $0^\circ$）。**通过补偿参数，Pitch 优化了 $0.036^\circ$，Roll 优化了 $0.513^\circ$**，展现了标定补偿矩阵对捷联惯导定姿精度的极致改良！

---

## 📈 自动生成的可视化图表目录
- **`images/compensation_comparison.png`**：直观展示加速度计三轴比力与陀螺仪三轴角速度在误差补偿前后的信号对比。补偿后加速度计的常值系统零偏和非正交轴耦合信号被彻底消除，曲线平稳贴合物理真值。
- **`images/attitude_curves.png`**：对比展示了“每秒滑动平均定姿”和“每历元单点定姿”下的 Pitch, Roll, Yaw 随时间的变化曲线。每秒滑动平均曲线极其平滑稳定，印证了时间窗低通滤波在压制高频白噪声中的显著作用。

---

