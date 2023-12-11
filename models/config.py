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
    epoches = 5  # bert 数据集3 5     bert 数据集1 5
    print_step = 5


class LSTMConfig(object):
    emb_size = 768  # 词向量的维数
    hidden_size = 400  # lstm隐向量的维数

