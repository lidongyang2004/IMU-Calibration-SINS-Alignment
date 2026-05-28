import os
import numpy as np

# -----------------------------------------------
#     常数定义 (保持与 C++ IMU_Structs.h 符号一致)
# -----------------------------------------------
AS_XWGI = 1.5258789063e-6  # XW-GI7681 加速度计数据转换比例因子
r_XWGI = 100               # XW-GI7681 采样率100Hz
gravity = 9.7936174        # 重力加速度

def read_raw_data_cali(file_path):
    """
    解析单轴静态文件的加速度计数据并计算平均值
    数据排列格式参考 ReadFile.cpp:
    分号后为: 周,周内秒,状态位,acc_z,acc_y,acc_x,gyr_z,gyr_y,gyr_x*校验码
    """
    acc_sum = np.zeros(3)  # [X, Y, Z]
    epoch_num = 0
    
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到 {file_path}")
        return None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ';' not in line:
                continue
            try:
                # 按照分号切分，跳过前面的 %RAWIMUSA,...
                _, data_part = line.split(';', 1)
                
                # 去除末尾的星号及校验码
                if '*' in data_part:
                    data_part = data_part.split('*', 1)[0]
                
                fields = data_part.split(',')
                
                # 严格对应 C++ ReadFile.cpp 中的解析顺序:
                # fields[3]: Z, fields[4]: Y, fields[5]: X
                raw_z = float(fields[3])
                raw_y = float(fields[4])
                raw_x = float(fields[5])
                
                # 转换公式与 C++ 一致 (乘以比例因子和采样率)
                # 注意: C++中 Y 轴读入时取了负号
                val_z = raw_z * AS_XWGI * r_XWGI
                val_y = -raw_y * AS_XWGI * r_XWGI
                val_x = raw_x * AS_XWGI * r_XWGI
                
                acc_sum[0] += val_x
                acc_sum[1] += val_y
                acc_sum[2] += val_z
                epoch_num += 1
            except (ValueError, IndexError):
                continue
                
    if epoch_num == 0:
        raise ValueError(f"文件 {file_path} 中未读取到有效数据")
        
    return acc_sum / epoch_num  # 返回该位置下的 [X, Y, Z] 比力平均值

def main():
    # 数据文件夹路径
    data_dir = "data/Calibration"
    
    # 六个位置对应的文件名
    file_names = [
        "x_up_3min.ASC",    # 0: X轴向上
        "x_down_3min.ASC",  # 1: X轴向下
        "y_up_3min.ASC",    # 2: Y轴向上
        "y_down_3min.ASC",  # 3: Y轴向下
        "z_up_3min.ASC",    # 4: Z轴向上
        "z_down_3min.ASC"   # 5: Z轴向下
    ]
    
    # 1. 填充实验测得比力平均值矩阵 fmean (对应 PDF 中的 L 矩阵, 维度 3x6)
    fmean = np.zeros((3, 6))
    
    print("开始读取静态数据文件并计算比力平均值...")
    for place, file_name in enumerate(file_names):
        file_path = os.path.join(data_dir, file_name)
        means = read_raw_data_cali(file_path)
        if means is not None:
            fmean[:, place] = means
            print(f"位置 {place} ({file_name[:-9]}): 均值 X={means[0]:.6f}, Y={means[1]:.6f}, Z={means[2]:.6f}")
            
    # 2. 填充已知输入比力矩阵 freal (对应 PDF 中的 A 矩阵, 齐次形式维度 4x6)
    freal = np.zeros((4, 6))
    for i in range(3):
        freal[i, i * 2] = gravity
        freal[i, i * 2 + 1] = -gravity
        freal[3, i * 2] = 1.0      # 用于计算零偏的齐次项
        freal[3, i * 2 + 1] = 1.0

    # 3. 最小二乘法求解包含12个误差参数的齐次转换矩阵 M (维度 3x4)
    # M = [ I + Sa + Na | ba ]
    M = fmean @ freal.T @ np.linalg.inv(freal @ freal.T)
    
    # 4. 组织成标准的 b_a, S_a, N_a 矩阵形式
    # (1) 零偏向量 b_a (3x1)
    b_a = M[:, 3].reshape(3, 1)
    
    # (2) 提取左侧的 3x3 矩阵 (等于 I + Sa + Na)
    M_3x3 = M[:, :3]
    
    # (3) 比例因子误差矩阵 S_a (3x3 对角阵)
    scale_elements = np.diag(M_3x3) - 1.0
    S_a = np.diag(scale_elements)
    
    # (4) 交轴耦合误差矩阵 N_a (3x3 对角线全为0)
    N_a = M_3x3 - np.diag(np.diag(M_3x3))

    # 5. 屏幕格式化打印矩阵结果
    np.set_printoptions(suppress=True, precision=12) # 防止科学计数法
    print("\n" + "="*50)
    print(" 标定成功！标准误差矩阵形式输出 (保存在标准国际单位制)：")
    print("="*50)
    
    print("1. 加速度计零偏向量 b_a (m/s^2):")
    print(b_a)
    print("\n2. 比例因子误差矩阵 S_a:")
    print(S_a)
    print("\n3. 交轴耦合误差矩阵 N_a:")
    print(N_a)
    
    print("\n" + "-"*50)
    print("验证: (I + S_a + N_a) 应该等于解算矩阵的前三列:")
    print(np.eye(3) + S_a + N_a)

    # 6. 保存标准矩阵结果到文件
    output_path = "AccCaliResult_Matrix.txt"
    with open(output_path, 'w') as out_f:
        out_f.write("=== ACCELEROMETER CALIBRATION MATRICES ===\n\n")
        out_f.write("1. Bias Vector b_a (m/s^2):\n")
        out_f.write(np.array2string(b_a, separator=', ') + "\n\n")
        out_f.write("2. Scale Factor Error Matrix S_a:\n")
        out_f.write(np.array2string(S_a, separator=', ') + "\n\n")
        out_f.write("3. Cross-axis Coupling Matrix N_a:\n")
        out_f.write(np.array2string(N_a, separator=', ') + "\n")
        
    print(f"\n结果矩阵已成功保存在: {output_path}")

if __name__ == "__main__":
    main()