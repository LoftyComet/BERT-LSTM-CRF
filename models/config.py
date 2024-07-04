import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu = '0'





# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 32
    # 学习速率
    lr = 0.001
    epochs = 100
    print_step = 50


class LSTMConfig(object):
    hidden_size = 64  # lstm隐向量的维数
    num_layers = 2  # lstm层数
    time_step = 20  # 时间步长 !!! 滑动窗口大小
    # input_size = 36 * 20  # if MLP
    input_size = 36
    out_size = 3
    completion_percentage = 1  # 选择过程完成的百分比
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu')
    print("计算设备为", device)

