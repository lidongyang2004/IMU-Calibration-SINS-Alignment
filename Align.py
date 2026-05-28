import os
import numpy as np
import matplotlib.pyplot as plt
from imu_calib.utils.io import read_imu_data
from imu_calib.alignment.coarse_align import CoarseAligner
from imu_calib.compensation.compensate import IMUCompensator

def main():
    data_dir = "data/Calibration"
    test_file = os.path.join(data_dir, "z_up_3min.ASC")
    
    print("==================================================")
    # 静态解析粗对准实验独立运行入口
    print("      捷联惯导静态解析粗对准 (独立运行程序)       ")
    print("==================================================")
    
    if not os.path.exists(test_file):
        print(f"错误: 静态对准测试文件未找到 {test_file}")
        return
        
    print(f"正在读取静态惯导数据: {test_file} ...")
    raw_dataset = read_imu_data(test_file)
    
    # 尝试加载标定补偿矩阵
    acc_cali_path = "AccCaliResult_Matrix.txt"
    gyr_cali_path = "GyrCaliResult_Matrix.txt"
    
    comp = IMUCompensator()
    if os.path.exists(acc_cali_path) and os.path.exists(gyr_cali_path):
        print("检测到标定矩阵文件，执行标定后数据对准补偿...")
        comp.load_parameters(acc_cali_path, gyr_cali_path)
        dataset = comp.compensate_dataset(raw_dataset)
        is_compensated = True
    else:
        print("警告: 未检测到标定矩阵，将使用未标定的原始数据进行对准...")
        dataset = raw_dataset
        is_compensated = False
        
    # 创建粗对准器
    aligner = CoarseAligner(dataset)
    
    # 运行整段平均对准
    print("\n>>> 正在运行方法 1: 全时段平均双矢量对准...")
    pitch, roll, yaw, C_b_n = aligner.align_whole_average()
    
    # 运行每秒平均对准
    print("\n>>> 正在运行方法 2: 每秒平均值计算姿态角随时间变化...")
    t_sec, p_sec, r_sec, y_sec = aligner.align_per_second()
    print(f"  计算完成。每秒对准结果标准差 (波动量): Pitch={np.std(p_sec):.6f}°, Roll={np.std(r_sec):.6f}°, Yaw={np.std(y_sec):.6f}°")
    
    # 运行每历元对准
    print("\n>>> 正在运行方法 3: 每历元计算姿态角随时间变化...")
    t_ep, p_ep, r_ep, y_ep = aligner.align_per_epoch()
    print(f"  计算完成。每历元对准结果标准差 (高频波动): Pitch={np.std(p_ep):.6f}°, Roll={np.std(r_ep):.6f}°, Yaw={np.std(y_ep):.6f}°")
    
    # 绘制结果曲线
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    status_str = " (已进行标定补偿)" if is_compensated else " (未进行标定补偿)"
    fig.suptitle("静态解析粗对准姿态角变化曲线" + status_str, fontsize=14)
    
    time_sec = t_sec - t_sec[0]
    time_ep = t_ep - t_ep[0]
    
    axes[0].plot(time_ep, p_ep, 'g', alpha=0.3, label='每历元计算')
    axes[0].plot(time_sec, p_sec, 'b', linewidth=2, label='每秒平均计算')
    axes[0].set_ylabel("俯仰角 Pitch (°)")
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].plot(time_ep, r_ep, 'g', alpha=0.3)
    axes[1].plot(time_sec, r_sec, 'b', linewidth=2)
    axes[1].set_ylabel("横滚角 Roll (°)")
    axes[1].grid(True)
    
    axes[2].plot(time_ep, y_ep, 'g', alpha=0.3)
    axes[2].plot(time_sec, y_sec, 'b', linewidth=2)
    axes[2].set_ylabel("航向角 Yaw (°)")
    axes[2].set_xlabel("时间 (s)")
    axes[2].grid(True)
    
    plt.tight_layout()
    plot_name = "coarse_align_attitude_curves.png"
    plt.savefig(plot_name, dpi=300)
    plt.close()
    print(f"\n姿态曲线图已生成并保存至: {plot_name}")
    print("\n静态粗对准实验执行完成！")

if __name__ == "__main__":
    main()
