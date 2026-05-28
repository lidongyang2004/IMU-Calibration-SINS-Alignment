import numpy as np
import os
import re
from imu_calib.constants import DEG_TO_RAD

class IMUCompensator:
    """
    IMU 标定误差补偿器
    """
    def __init__(self, b_a=None, S_a=None, N_a=None, b_g=None, S_g=None, N_g=None):
        self.b_a = b_a  # 3x1, m/s^2
        self.S_a = S_a  # 3x3, 对角
        self.N_a = N_a  # 3x3, 非对角
        
        self.b_g = b_g  # 3x1, deg/h
        self.S_g = S_g  # 3x3, 对角
        self.N_g = N_g  # 3x3, 非对角

    def load_parameters(self, acc_cali_path, gyr_cali_path):
        """
        从标定输出文本文件中解析标定矩阵参数
        """
        if os.path.exists(acc_cali_path):
            self.b_a, self.S_a, self.N_a = self._parse_matrix_file(acc_cali_path)
            print(f"成功加载加速度计补偿矩阵: {acc_cali_path}")
        else:
            print(f"警告: 无法加载 {acc_cali_path}")

        if os.path.exists(gyr_cali_path):
            self.b_g, self.S_g, self.N_g = self._parse_matrix_file(gyr_cali_path)
            print(f"成功加载陀螺仪补偿矩阵: {gyr_cali_path}")
        else:
            print(f"警告: 无法加载 {gyr_cali_path}")

    def _parse_matrix_file(self, file_path):
        """
        解析输出矩阵文件
        """
        with open(file_path, 'r') as f:
            content = f.read()

        # 使用正则提取三块矩阵数据
        matrices = re.findall(r'\[\[[\s\S]*?\]\]', content)
        if len(matrices) < 3:
            raise ValueError(f"文件格式不正确，未能解析出3个矩阵: {file_path}")
            
        def str_to_array(mat_str):
            # 将 numpy output string 转化为 np.array
            cleaned = mat_str.replace('[', '').replace(']', '').replace('\n', ' ')
            nums = [float(x) for x in cleaned.split(',') if x.strip()]
            if not nums:
                # 兼容空格分隔的形式
                nums = [float(x) for x in cleaned.split() if x.strip()]
            
            # 判断维度
            size = len(nums)
            if size == 3:
                return np.array(nums).reshape(3, 1)
            elif size == 9:
                return np.array(nums).reshape(3, 3)
            else:
                raise ValueError(f"不受支持的矩阵大小: {size}")

        bias = str_to_array(matrices[0])
        S = str_to_array(matrices[1])
        N = str_to_array(matrices[2])
        
        return bias, S, N

    def compensate_epoch(self, acc_raw, gyr_raw):
        """
        对单个历元数据进行误差补偿
        :param acc_raw: 3维比力输入向量 [ax, ay, az] (m/s^2)
        :param gyr_raw: 3维角速度输入向量 [gx, gy, gz] (rad/s)
        :return: acc_compensated, gyr_compensated (物理单位)
        """
        acc_compensated = acc_raw.copy()
        gyr_compensated = gyr_raw.copy()
        
        # 1. 加速度计补偿
        # f_comp = (I + S_a + N_a)^-1 * (f_raw - b_a)
        if self.b_a is not None and self.S_a is not None and self.N_a is not None:
            I = np.eye(3)
            inv_acc_mat = np.linalg.inv(I + self.S_a + self.N_a)
            # bias is 3x1, acc_raw is 1D or 3x1
            diff_acc = acc_raw.reshape(3, 1) - self.b_a.reshape(3, 1)
            acc_compensated = (inv_acc_mat @ diff_acc).flatten()

        # 2. 陀螺仪补偿
        # w_comp = (I + S_g + N_g)^-1 * (w_raw - b_g)
        # 注意: 陀螺零偏 b_g 存储单位是 deg/h，需转换为 rad/s 才能与原始的 rad/s 做减法
        if self.b_g is not None and self.S_g is not None and self.N_g is not None:
            # deg/h -> deg/s -> rad/s
            b_g_rad_s = (self.b_g.reshape(3, 1) / 3600.0) * DEG_TO_RAD
            I = np.eye(3)
            inv_gyr_mat = np.linalg.inv(I + self.S_g + self.N_g)
            diff_gyr = gyr_raw.reshape(3, 1) - b_g_rad_s
            gyr_compensated = (inv_gyr_mat @ diff_gyr).flatten()
            
        return acc_compensated, gyr_compensated

    def compensate_dataset(self, dataset):
        """
        批量补偿整个数据集
        :param dataset: dict containing 'acc' and 'gyr' keys with Nx3 arrays
        :return: dict containing compensated 'acc' and 'gyr'
        """
        acc_raw_all = dataset['acc']
        gyr_raw_all = dataset['gyr']
        N = len(acc_raw_all)
        
        acc_comp_all = np.zeros_like(acc_raw_all)
        gyr_comp_all = np.zeros_like(gyr_raw_all)
        
        for i in range(N):
            ac, gc = self.compensate_epoch(acc_raw_all[i], gyr_raw_all[i])
            acc_comp_all[i] = ac
            gyr_comp_all[i] = gc
            
        return {
            'time': dataset['time'].copy(),
            'acc': acc_comp_all,
            'gyr': gyr_comp_all
        }
