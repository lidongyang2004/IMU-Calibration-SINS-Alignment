import os
import numpy as np
from imu_calib.utils.io import read_imu_data
from imu_calib.constants import GRAVITY

class AccelerometerCalibrator:
    """
    三轴加速度计的静态六位置法标定器
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = [
            "x_up_3min.ASC",    # 0: X轴向上
            "x_down_3min.ASC",  # 1: X轴向下
            "y_up_3min.ASC",    # 2: Y轴向上
            "y_down_3min.ASC",  # 3: Y轴向下
            "z_up_3min.ASC",    # 4: Z轴向上
            "z_down_3min.ASC"   # 5: Z轴向下
        ]
        
        # 标定结果
        self.b_a = None  # 零偏 (m/s^2)
        self.S_a = None  # 比例因子误差矩阵 (3x3 对角阵)
        self.N_a = None  # 交轴耦合误差矩阵 (3x3 非对角阵)
        self.M = None    # 12参数齐次映射矩阵

    def run_calibration(self):
        """
        运行静态六位置标定
        """
        # 1. 填充实验测得比力平均值矩阵 fmean (对应 PDF 中的 L 矩阵, 维度 3x6)
        fmean = np.zeros((3, 6))
        
        print("开始读取静态数据文件并计算比力平均值...")
        for place, file_name in enumerate(self.file_names):
            file_path = os.path.join(self.data_dir, file_name)
            data = read_imu_data(file_path)
            
            # 计算三轴加速度比力平均值
            means = np.mean(data['acc'], axis=0)
            fmean[:, place] = means
            print(f"位置 {place} ({file_name[:-9]}): 均值 X={means[0]:.6f}, Y={means[1]:.6f}, Z={means[2]:.6f}")
            
        # 2. 填充已知真实输入比力矩阵 freal (对应 PDF 中的 A 矩阵, 齐次形式维度 4x6)
        freal = np.zeros((4, 6))
        for i in range(3):
            freal[i, i * 2] = GRAVITY
            freal[i, i * 2 + 1] = -GRAVITY
            freal[3, i * 2] = 1.0      # 用于计算零偏的齐次项
            freal[3, i * 2 + 1] = 1.0

        # 3. 最小二乘法求解包含12个误差参数的齐次转换矩阵 M (维度 3x4)
        # M = [ I + Sa + Na | ba ]
        self.M = fmean @ freal.T @ np.linalg.inv(freal @ freal.T)
        
        # 4. 组织并分解成标准的 b_a, S_a, N_a 矩阵形式
        # (1) 零偏向量 b_a (3x1)
        self.b_a = self.M[:, 3].reshape(3, 1)
        
        # (2) 提取左侧的 3x3 矩阵 (等于 I + Sa + Na)
        M_3x3 = self.M[:, :3]
        
        # (3) 比例因子误差矩阵 S_a (3x3 对角阵)
        scale_elements = np.diag(M_3x3) - 1.0
        self.S_a = np.diag(scale_elements)
        
        # (4) 交轴耦合误差矩阵 N_a (3x3 对角线全为0)
        self.N_a = M_3x3 - np.diag(np.diag(M_3x3))
        
        return self.b_a, self.S_a, self.N_a

    def save_results(self, output_path):
        """
        保存标定矩阵结果到文件
        """
        if self.b_a is None:
            raise ValueError("请先运行 run_calibration() 标定程序")
            
        with open(output_path, 'w') as out_f:
            out_f.write("=== ACCELEROMETER CALIBRATION MATRICES ===\n\n")
            out_f.write("1. Bias Vector b_a (m/s^2):\n")
            out_f.write(np.array2string(self.b_a, separator=', ') + "\n\n")
            out_f.write("2. Scale Factor Error Matrix S_a:\n")
            out_f.write(np.array2string(self.S_a, separator=', ') + "\n\n")
            out_f.write("3. Cross-axis Coupling Matrix N_a:\n")
            out_f.write(np.array2string(self.N_a, separator=', ') + "\n")
            
        print(f"加速度计结果矩阵已保存至: {output_path}")
