import math
import time

import numpy as np

from data import load_data, divide_data
from models.bilstm_crf import LstmModel
from models.config import LSTMConfig
from utils import load_model, draw_point, draw_finger

data, tag = load_data(1, 3)
(train_lists, train_tag_lists), (dev_lists, dev_tag_lists), (test_lists, test_tag_lists) = divide_data(data, tag)


def get_pre(to_pre):
    """加载模型，输入时序数据，返回预测结果
    :param to_pre: 待测的数据 [1, time_step * completion_percentage, input_size]

    :return: pred_tag_lists, [[x, y, z]]
    """

    lstm_model = load_model('./model_saved/lstm_.pkl')
    lstm_model.model.lstm.flatten_parameters()  # remove warning
    pred_tag_lists, _ = lstm_model.test(
        to_pre, 1)

    pred_tag_lists = pred_tag_lists.numpy()

    return pred_tag_lists


# 记录开始时间
start_time = time.time()
pred_tag_lists = get_pre(test_lists)
# 记录结束时间
end_time = time.time()
# 计算运行时间（以秒为单位）
execution_time = end_time - start_time
print("代码段运行时间：", execution_time, "秒")
ans = []
for i in range(len(pred_tag_lists)):
    # for j in range(len(pred_tag_lists[0])):
    print("i", i, test_tag_lists[i][0], test_tag_lists[i][1], test_tag_lists[i][2])
    print("i", i, pred_tag_lists[i][0], pred_tag_lists[i][1], pred_tag_lists[i][2])
    temp = math.sqrt(
        (test_tag_lists[i][0] - pred_tag_lists[i][0]) ** 2 + (test_tag_lists[i][1] - pred_tag_lists[i][1]) ** 2 + (
                    test_tag_lists[i][2] - pred_tag_lists[i][2]) ** 2)
    print(temp)
    ans.append(temp)
print("测试集中预测位置与真实位置的平均距离为", np.mean(ans))
draw_finger(train_lists[1][31], train_lists[1][32], train_lists[1][33], train_tag_lists[1])
draw_point(pred_tag_lists[:40], test_tag_lists[:40])
for i in range(len(pred_tag_lists)):
    draw_point([pred_tag_lists[i]], [test_tag_lists[i]])
# draw_point(pred_tag_lists, test_tag_lists)
