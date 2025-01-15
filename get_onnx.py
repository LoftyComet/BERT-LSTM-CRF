import math
import time

import numpy as np
import torch

from data import load_data, divide_data
from models.bilstm_crf import LstmModel
from models.config import LSTMConfig
from utils import load_model, draw_point, draw_finger, draw_error
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score



data, tag = load_data(1, 1, for_train=False)
(train_lists, train_tag_lists), (dev_lists, dev_tag_lists), (test_lists, test_tag_lists) = divide_data(data, tag)


def get_pre(to_pre):
    """加载模型，输入时序数据，返回预测结果
    :param to_pre: 待测的数据 [1, time_step * completion_percentage, input_size]

    :return: pred_tag_lists, [[x, y, z]]
    """
    # lstm_model = load_model('./model_saved/' + "lstm" + '1' + '-' + '14' + '.pkl')
    lstm_model = load_model('./train_model_saved0/' + "lstm" + '100' + '.pkl')
    # data = torch.zeros(1, LSTMConfig.time_step, LSTMConfig.input_size).to(LSTMConfig.device)
    data = torch.zeros(1, LSTMConfig.time_step, LSTMConfig.input_size).to(LSTMConfig.device)  # if mlp
    torch.onnx.export(lstm_model, data, './model_saved/exp' + '.onnx', export_params=True, verbose=False, input_names=None,
                      output_names=None, do_constant_folding=True, dynamic_axes=None, opset_version=14)

    # lstm_model.lstm.flatten_parameters()  # remove warning
    to_pre = to_pre.to(LSTMConfig.device)
    pred_tag_lists = lstm_model.forward(to_pre)

    pred_tag_lists = pred_tag_lists.cpu().detach().numpy()
    print("输出onnx成功!")
    return pred_tag_lists

trigger_dis = []
# 记录开始时间
start_time = time.time()
pred_tag_lists = get_pre(test_lists)

# 记录结束时间
end_time = time.time()
# 计算运行时间（以秒为单位）
execution_time = end_time - start_time
print("代码段运行时间：", execution_time, "秒")
