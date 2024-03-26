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



data, tag = load_data(1, 15)
(train_lists, train_tag_lists), (dev_lists, dev_tag_lists), (test_lists, test_tag_lists) = divide_data(data, tag)


def get_pre(to_pre):
    """加载模型，输入时序数据，返回预测结果
    :param to_pre: 待测的数据 [1, time_step * completion_percentage, input_size]

    :return: pred_tag_lists, [[x, y, z]]
    """
    lstm_model = load_model('./model_saved/lstm_' + str(int(LSTMConfig.completion_percentage * 100)) + '.pkl')
    torch.onnx.export(lstm_model, to_pre[0], './model_saved/lstm_' + str(int(LSTMConfig.completion_percentage * 100)) + '.onnx', export_params=True, verbose=False, input_names=None,
                      output_names=None, do_constant_folding=True, dynamic_axes=None, opset_version=9)

    lstm_model.model.lstm.flatten_parameters()  # remove warning
    pred_tag_lists, _ = lstm_model.test(
        to_pre, 1)

    pred_tag_lists = pred_tag_lists.numpy()

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
