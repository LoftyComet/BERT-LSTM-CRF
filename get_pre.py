import math
import time

import numpy as np

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
    lstm_model.model.lstm.flatten_parameters()  # remove warning
    pred_tag_lists, _ = lstm_model.test(
        to_pre, 1)

    pred_tag_lists = pred_tag_lists.numpy()

    return pred_tag_lists

trigger_dis = []
# 记录开始时间
start_time = time.time()
pred_tag_lists = get_pre(test_lists)
for i, test_list in enumerate(test_lists):
    # 预测时手的位置
    fingerX = test_list[30][-1]
    fingerY = test_list[31][-1]
    fingerZ = test_list[32][-1]
    temp = math.sqrt(
        (test_tag_lists[i][0] - fingerX) ** 2 + (test_tag_lists[i][1] - fingerY) ** 2 + (
                test_tag_lists[i][2] - fingerZ) ** 2)
    trigger_dis.append(temp)
print(np.mean(trigger_dis))
# 记录结束时间
end_time = time.time()
# 计算运行时间（以秒为单位）
execution_time = end_time - start_time
print("代码段运行时间：", execution_time, "秒")
test_tag_lists_x = []
test_tag_lists_y = []
test_tag_lists_z = []
pred_tag_lists_x = []
pred_tag_lists_y = []
pred_tag_lists_z = []
ans = []
zero = []
for i in range(len(pred_tag_lists)):
    temp = math.sqrt(
        (test_tag_lists[i][0] - pred_tag_lists[i][0]) ** 2 + (test_tag_lists[i][1] - pred_tag_lists[i][1]) ** 2 + (
                    test_tag_lists[i][2] - pred_tag_lists[i][2]) ** 2)
    ans.append(temp)
    zero.append(0)
    test_tag_lists_x.append(test_tag_lists[i][0])
    test_tag_lists_y.append(test_tag_lists[i][1])
    test_tag_lists_z.append(test_tag_lists[i][2])
    pred_tag_lists_x.append(pred_tag_lists[i][0])
    pred_tag_lists_y.append(pred_tag_lists[i][1])
    pred_tag_lists_z.append(pred_tag_lists[i][2])


print("测试集中预测位置与真实位置的平均距离为", np.mean(ans))
print("X坐标R^2", r2_score(test_tag_lists_x, pred_tag_lists_x))
print("Y坐标R^2", r2_score(test_tag_lists_y, pred_tag_lists_y))
print("Z坐标R^2", r2_score(test_tag_lists_z, pred_tag_lists_z))

# 画预测点和目标球
# for i in range(len(train_lists)):
#     draw_finger(train_lists[i][30], train_lists[i][31], train_lists[i][32], train_tag_lists[i])
# draw_point(pred_tag_lists[:10], test_tag_lists[:10])
# for i in range(len(pred_tag_lists)):
#     draw_point([pred_tag_lists[i]], [test_tag_lists[i]])
# draw_point(pred_tag_lists, test_tag_lists)

draw_error(pred_tag_lists, test_tag_lists, ans)

# 结果可视化
fig, ax = plt.subplots(4, 1)
fig.set_size_inches(10, 4)

ax[0].plot(range(len(test_tag_lists_x))[:40], test_tag_lists_x[:40], linewidth=1.5, linestyle='-', label='True')
ax[0].plot(range(len(pred_tag_lists_x))[:40], pred_tag_lists_x[:40], linewidth=1, linestyle='-.', label='Predicted')
ax[0].set_title("x")

ax[1].plot(range(len(test_tag_lists_y))[:40], test_tag_lists_y[:40], linewidth=1.5, linestyle='-', label='True')
ax[1].plot(range(len(pred_tag_lists_y))[:40], pred_tag_lists_y[:40], linewidth=1, linestyle='-.', label='Predicted')
ax[1].set_title("y")

ax[2].plot(range(len(test_tag_lists_z))[:40], test_tag_lists_z[:40], linewidth=1.5, linestyle='-', label='True')
ax[2].plot(range(len(pred_tag_lists_z))[:40], pred_tag_lists_z[:40], linewidth=1, linestyle='-.', label='Predicted')
ax[2].set_title("z")

ax[3].plot(range(len(ans))[:40], zero[:40], linewidth=1.5, linestyle='-', label='True')
ax[3].plot(range(len(ans))[:40], ans[:40], linewidth=1, linestyle='-', label='Predicted')
ax[3].set_title("distance")
plt.legend()
plt.show()

for ans1 in ans:
    print(ans1)
