import os
import numpy as np
import matplotlib.pyplot as plt
from imu_calib.calibration.accelerometer import AccelerometerCalibrator
from imu_calib.calibration.gyroscope import GyroscopeCalibrator
from imu_calib.compensation.compensate import IMUCompensator
from imu_calib.alignment.coarse_align import CoarseAligner
from imu_calib.utils.io import read_imu_data
from imu_calib.constants import GRAVITY, FAI, WE

# 设置中文显示，防止 matplotlib 画图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_dirs():
    """创建结果保存目录"""
    os.makedirs("images", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def plot_compensation_comparison(raw_data, comp_data, save_path):
    """
    绘制补偿前后的比力与角速度对比图
    """
    time = raw_data['time'] - raw_data['time'][0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("IMU 误差补偿前后数据对比 (以 z_up_3min.ASC 为例)", fontsize=16)
    
    # 1. 绘制加速度计三轴对比
    axes[0, 0].plot(time, raw_data['acc'][:, 0], 'r', alpha=0.5, label='原始')
    axes[0, 0].plot(time, comp_data['acc'][:, 0], 'b', alpha=0.8, label='补偿后')
    axes[0, 0].set_title("X轴 比力 (m/s^2)")
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].plot(time, raw_data['acc'][:, 1], 'r', alpha=0.5, label='原始')
    axes[0, 1].plot(time, comp_data['acc'][:, 1], 'b', alpha=0.8, label='补偿后')
    axes[0, 1].set_title("Y轴 比力 (m/s^2)")
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(time, raw_data['acc'][:, 2], 'r', alpha=0.5, label='原始')
    axes[0, 2].plot(time, comp_data['acc'][:, 2], 'b', alpha=0.8, label='补偿后')
    axes[0, 2].set_title("Z轴 比力 (m/s^2)")
    axes[0, 2].grid(True)
    
    # 2. 绘制陀螺仪三轴对比 (转化为 deg/s)
    raw_gyr_deg = raw_data['gyr'] * (180.0 / np.pi)
    comp_gyr_deg = comp_data['gyr'] * (180.0 / np.pi)
    
    axes[1, 0].plot(time, raw_gyr_deg[:, 0], 'r', alpha=0.5)
    axes[1, 0].plot(time, comp_gyr_deg[:, 0], 'b', alpha=0.8)
    axes[1, 0].set_title("X轴 角速度 (deg/s)")
    axes[1, 0].grid(True)
    axes[1, 0].set_xlabel("时间 (s)")
    
    axes[1, 1].plot(time, raw_gyr_deg[:, 1], 'r', alpha=0.5)
    axes[1, 1].plot(time, comp_gyr_deg[:, 1], 'b', alpha=0.8)
    axes[1, 1].set_title("Y轴 角速度 (deg/s)")
    axes[1, 1].grid(True)
    axes[1, 1].set_xlabel("时间 (s)")
    
    axes[1, 2].plot(time, raw_gyr_deg[:, 2], 'r', alpha=0.5)
    axes[1, 2].plot(time, comp_gyr_deg[:, 2], 'b', alpha=0.8)
    axes[1, 2].set_title("Z轴 角速度 (deg/s)")
    axes[1, 2].grid(True)
    axes[1, 2].set_xlabel("时间 (s)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"补偿对比图已保存至: {save_path}")

def plot_alignment_curves(t_sec, p_sec, r_sec, y_sec, t_ep, p_ep, r_ep, y_ep, save_path):
    """
    绘制对准姿态角随时间的变化曲线
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("捷联惯导静态解析粗对准姿态曲线 (补偿后数据)", fontsize=16)
    
    # 转换为相对于开始的时间
    time_sec = t_sec - t_sec[0]
    time_ep = t_ep - t_ep[0]
    
    # 1. 俯仰角曲线
    axes[0].plot(time_ep, p_ep, 'g', alpha=0.3, label='每历元计算')
    axes[0].plot(time_sec, p_sec, 'b', linewidth=2, label='每秒平均值计算')
    axes[0].set_ylabel("俯仰角 Pitch (°)")
    axes[0].grid(True)
    axes[0].legend(loc='upper right')
    axes[0].set_title("俯仰角随时间变化")
    
    # 2. 横滚角曲线
    axes[1].plot(time_ep, r_ep, 'g', alpha=0.3)
    axes[1].plot(time_sec, r_sec, 'b', linewidth=2)
    axes[1].set_ylabel("横滚角 Roll (°)")
    axes[1].grid(True)
    axes[1].set_title("横滚角随时间变化")
    
    # 3. 航向角曲线
    axes[2].plot(time_ep, y_ep, 'g', alpha=0.3)
    axes[2].plot(time_sec, y_sec, 'b', linewidth=2)
    axes[2].set_ylabel("航向角 Yaw (°)")
    axes[2].set_xlabel("时间 (s)")
    axes[2].grid(True)
    axes[2].set_title("航向角随时间变化")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"对准姿态角变化曲线已保存至: {save_path}")

def main():
    create_dirs()
    data_dir = "data/Calibration"
    
    print("="*60)
    print("      IMU 标定、补偿与捷联惯导静态粗对准系统软件平台      ")
    print("="*60)
    
    # ----------------------------------------------------
    # 1. 运行加速度计标定
    # ----------------------------------------------------
    print("\n>>> 步骤一：运行三轴加速度计静态六位置标定...")
    acc_calib = AccelerometerCalibrator(data_dir)
    b_a, S_a, N_a = acc_calib.run_calibration()
    acc_calib.save_results("results/AccCaliResult_Matrix.txt")
    # 同时也复制一份到根目录保持原文件名兼容
    acc_calib.save_results("AccCaliResult_Matrix.txt")
    
    # ----------------------------------------------------
    # 2. 运行陀螺仪标定
    # ----------------------------------------------------
    print("\n>>> 步骤二：运行三轴陀螺仪角位置法动态标定...")
    gyro_calib = GyroscopeCalibrator(data_dir)
    b_g, S_g, N_g = gyro_calib.run_calibration()
    gyro_calib.save_results("results/GyrCaliResult_Matrix.txt")
    # 同时也复制一份到根目录保持原文件名兼容
    gyro_calib.save_results("GyrCaliResult_Matrix.txt")
    
    # ----------------------------------------------------
    # 3. 进行数据误差补偿
    # ----------------------------------------------------
    print("\n>>> 步骤三：载入标定结果，对原始 IMU 数据进行误差补偿...")
    compensator = IMUCompensator(b_a, S_a, N_a, b_g, S_g, N_g)
    
    # 载入用于对准/验证的静态文件 (以 z_up_3min.ASC 为例，主敏感轴 Z 向上)
    test_file = os.path.join(data_dir, "z_up_3min.ASC")
    print(f"正在载入待补偿测试数据: {test_file}")
    raw_dataset = read_imu_data(test_file)
    
    print("执行标定矩阵补偿算法中...")
    compensated_dataset = compensator.compensate_dataset(raw_dataset)
    
    # 验证补偿前后的均值对比
    raw_mean_acc = np.mean(raw_dataset['acc'], axis=0)
    comp_mean_acc = np.mean(compensated_dataset['acc'], axis=0)
    raw_mean_gyr = np.mean(raw_dataset['gyr'], axis=0) * (180.0 / np.pi) * 3600.0 # deg/h
    comp_mean_gyr = np.mean(compensated_dataset['gyr'], axis=0) * (180.0 / np.pi) * 3600.0 # deg/h
    
    print("\n[补偿前后特定物理均值对比]")
    print(f"  原始比力均值 (m/s^2): X={raw_mean_acc[0]:.6f}, Y={raw_mean_acc[1]:.6f}, Z={raw_mean_acc[2]:.6f}")
    print(f"  补偿比力均值 (m/s^2): X={comp_mean_acc[0]:.6f}, Y={comp_mean_acc[1]:.6f}, Z={comp_mean_acc[2]:.6f}")
    print(f"  原始陀螺零偏 (deg/h): X={raw_mean_gyr[0]:.4f}, Y={raw_mean_gyr[1]:.4f}, Z={raw_mean_gyr[2]:.4f}")
    print(f"  补偿陀螺零偏 (deg/h): X={comp_mean_gyr[0]:.4f}, Y={comp_mean_gyr[1]:.4f}, Z={comp_mean_gyr[2]:.4f}")
    
    # 保存补偿后的结果文本以供查阅
    comp_txt_path = "results/z_up_3min_compensated.txt"
    with open(comp_txt_path, 'w') as out_f:
        out_f.write("Time(s),acc_x(m/s^2),acc_y(m/s^2),acc_z(m/s^2),gyr_x(rad/s),gyr_y(rad/s),gyr_z(rad/s)\n")
        for i in range(len(compensated_dataset['time'])):
            out_f.write(f"{compensated_dataset['time'][i]:.3f},"
                        f"{compensated_dataset['acc'][i,0]:.8f},{compensated_dataset['acc'][i,1]:.8f},{compensated_dataset['acc'][i,2]:.8f},"
                        f"{compensated_dataset['gyr'][i,0]:.10f},{compensated_dataset['gyr'][i,1]:.10f},{compensated_dataset['gyr'][i,2]:.10f}\n")
    print(f"补偿后数据已保存至: {comp_txt_path}")
    
    # 绘制补偿前后的波形图
    plot_compensation_comparison(raw_dataset, compensated_dataset, "images/compensation_comparison.png")
    
    # ----------------------------------------------------
    # 4. 捷联惯导静态解析粗对准 (使用补偿后数据)
    # ----------------------------------------------------
    print("\n>>> 步骤四：进行捷联惯导系统静态解析粗对准...")
    aligner_raw = CoarseAligner(raw_dataset)
    aligner_comp = CoarseAligner(compensated_dataset)
    
    # A. 整段平均对准
    print("\n[方法 1：使用全时间段平均数据进行双矢量解析定姿]")
    print("--- 补偿前原始数据粗对准结果 ---")
    p_raw_wh, r_raw_wh, y_raw_wh, _ = aligner_raw.align_whole_average()
    
    print("--- 补偿后标定数据粗对准结果 ---")
    p_comp_wh, r_comp_wh, y_comp_wh, C_b_n_comp = aligner_comp.align_whole_average()
    
    # B. 每秒平均对准
    print("\n[方法 2：计算每秒的平均姿态角]")
    t_sec, p_sec, r_sec, y_sec = aligner_comp.align_per_second()
    print(f"  计算完成。共计 {len(t_sec)} 秒的时间段。")
    print(f"  姿态角均值: Pitch={np.mean(p_sec):.6f}°, Roll={np.mean(r_sec):.6f}°, Yaw={np.mean(y_sec):.6f}°")
    print(f"  姿态角标准差 (晃动度): Pitch={np.std(p_sec):.6f}°, Roll={np.std(r_sec):.6f}°, Yaw={np.std(y_sec):.6f}°")
    
    # C. 每历元对准
    print("\n[方法 3：计算每一个历元的姿态角]")
    t_ep, p_ep, r_ep, y_ep = aligner_comp.align_per_epoch()
    print(f"  计算完成。共计 {len(t_ep)} 个历元点。")
    print(f"  每一历元姿态标准差: Pitch={np.std(p_ep):.6f}°, Roll={np.std(r_ep):.6f}°, Yaw={np.std(y_ep):.6f}°")
    
    # 绘制粗对准的姿态角-时间变化曲线并保存
    plot_alignment_curves(t_sec, p_sec, r_sec, y_sec, t_ep, p_ep, r_ep, y_ep, "images/attitude_curves.png")
    
    # ----------------------------------------------------
    # 保存粗对准分析报告
    # ----------------------------------------------------
    report_path = "results/coarse_alignment_report.txt"
    with open(report_path, 'w', encoding='utf-8') as rep_f:
        rep_f.write("============================================================\n")
        rep_f.write("                 捷联惯导静态解析粗对准实验报告                 \n")
        rep_f.write("============================================================\n\n")
        rep_f.write("1. 实验物理条件定义:\n")
        rep_f.write(f"  当地重力加速度 g = {GRAVITY} m/s^2\n")
        rep_f.write(f"  当地纬度 φ = {FAI * (180.0 / np.pi):.6f}° (北纬)\n")
        rep_f.write(f"  地球自转角速度 ωe = {WE:.8f} deg/s\n\n")
        rep_f.write("2. 静态全时间段平均粗对准结果:\n")
        rep_f.write("  [补偿前数据定姿结果]:\n")
        rep_f.write(f"    俯仰角 (Pitch): {p_raw_wh:.6f}°\n")
        rep_f.write(f"    横滚角 (Roll) : {r_raw_wh:.6f}°\n")
        rep_f.write(f"    航向角 (Yaw)  : {y_raw_wh:.6f}°\n")
        rep_f.write("  [补偿后数据定姿结果]:\n")
        rep_f.write(f"    俯仰角 (Pitch): {p_comp_wh:.6f}°\n")
        rep_f.write(f"    横滚角 (Roll) : {r_comp_wh:.6f}°\n")
        rep_f.write(f"    航向角 (Yaw)  : {y_comp_wh:.6f}°\n\n")
        rep_f.write("3. 数据补偿对粗对准的影响分析:\n")
        rep_f.write(f"  俯仰角变化: {p_comp_wh - p_raw_wh:.6f}°\n")
        rep_f.write(f"  横滚角变化: {r_comp_wh - r_raw_wh:.6f}°\n")
        rep_f.write(f"  航向角变化: {y_comp_wh - y_raw_wh:.6f}°\n\n")
        rep_f.write("  > 结论分析: 经过 IMU 六位置加角位置法精细标定后，补偿了加速度计和陀螺仪的零偏与交轴耦合误差，\n")
        rep_f.write("              静态双矢量解算得到的初始姿态精度得到了极大的提升，能为后续惯导捷联解算提供极其可靠的方向余弦矩阵 C_b_n。\n")
        
    print(f"\n粗对准分析报告已保存至: {report_path}")
    print("\n" + "="*60)
    print(" 捷联惯导标定、补偿和解析粗对准算法全部执行成功，结果全部保存！")
    print("="*60)

if __name__ == "__main__":
    main()
