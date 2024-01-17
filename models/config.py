import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu = '0'


device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu')
print("计算设备为", device)


# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 8
    # 学习速率
    lr = 0.001
    epochs = 30
    print_step = 1


class LSTMConfig(object):
    hidden_size = 64  # lstm隐向量的维数
    num_layers = 1  # lstm层数
    time_step = 20  # 时间步长 !!!暂定20
    input_size = 44
    out_size = 3
    completion_percentage = 1  # 选择过程完成的百分比

