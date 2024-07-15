import math
import time

import numpy as np

from data import load_data, divide_data
from models.bilstm_crf import LstmModel
from models.config import LSTMConfig
from utils import load_model, draw_point, draw_finger, draw_error, draw_points
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

def get_pre(to_pre):
    """加载模型，输入时序数据，返回预测结果
    :param to_pre: 待测的数据 [1, time_step * completion_percentage, input_size]

    :return: pred_tag_lists, [[x, y, z]]
    """
    lstm_model = load_model('./model_saved/lstm_2' + str(int(LSTMConfig.completion_percentage * 100)) + '.pkl')
    # lstm_model.lstm.flatten_parameters()  # remove warning
    to_pre = to_pre.to(LSTMConfig.device)
    pred_tag_lists = lstm_model.forward(to_pre)

    pred_tag_lists = pred_tag_lists.cpu().detach().numpy()

    return pred_tag_lists


data, tag = load_data(33, 36, for_train=False, percentage=0.7)
(train_lists, train_tag_lists), (dev_lists, dev_tag_lists), (test_lists, test_tag_lists) = divide_data(data, tag)

# draw_points(train_tag_lists)

trigger_dis = []
tra_dis = []
# 记录开始时间
start_time = time.time()
pred_tag_lists = get_pre(test_lists)
for i, test_list in enumerate(test_lists):
    # 预测时手的位置
    fingerX = test_list[-1][24]
    fingerY = test_list[-1][25]
    fingerZ = test_list[-1][26]
    temp = math.sqrt(
        (test_tag_lists[i][0] - fingerX) ** 2 + (test_tag_lists[i][1] - fingerY) ** 2 + (
                test_tag_lists[i][2] - fingerZ) ** 2)
    trigger_dis.append(temp)
    temp2 = math.sqrt(
        (test_list[0][24] - fingerX) ** 2 + (test_list[0][25] - fingerY) ** 2 + (
                test_list[0][26] - fingerZ) ** 2)
    tra_dis.append(temp2)
print(np.mean(trigger_dis))
print(np.mean(tra_dis))
# 记录结束时间
end_time = time.time()
# 计算运行时间（以秒为单位）
execution_time = end_time - start_time
# print("代码段运行时间：", execution_time, "秒")
test_tag_lists_x = []
test_tag_lists_y = []
test_tag_lists_z = []
pred_tag_lists_x = []
pred_tag_lists_y = []
pred_tag_lists_z = []
x_error = []
y_error = []
z_error = []
ans = []
zero = []
for i in range(len(pred_tag_lists)):
    temp = math.sqrt(
        (test_tag_lists[i][0] - pred_tag_lists[i][0]) ** 2 + (test_tag_lists[i][1] - pred_tag_lists[i][1]) ** 2 + (
                    test_tag_lists[i][2] - pred_tag_lists[i][2]) ** 2)
    x_error.append(abs(test_tag_lists[i][0] - pred_tag_lists[i][0]))
    y_error.append(abs(test_tag_lists[i][1] - pred_tag_lists[i][1]))
    z_error.append(abs(test_tag_lists[i][2] - pred_tag_lists[i][2]))
    ans.append(temp)
    zero.append(0)
    test_tag_lists_x.append(test_tag_lists[i][0])
    test_tag_lists_y.append(test_tag_lists[i][1])
    test_tag_lists_z.append(test_tag_lists[i][2])
    pred_tag_lists_x.append(pred_tag_lists[i][0])
    pred_tag_lists_y.append(pred_tag_lists[i][1])
    pred_tag_lists_z.append(pred_tag_lists[i][2])

    temp2 = math.sqrt(
        (test_lists[i][-1][24] - pred_tag_lists[i][0]) ** 2 + (test_lists[i][-1][25] - pred_tag_lists[i][1]) ** 2 + (
                test_lists[i][-1][26] - pred_tag_lists[i][2]) ** 2)
    # print("手距离预测点距离", temp2)
    # print(pred_tag_lists[i][3])


print("测试集中预测位置与真实位置的平均距离为", np.mean(ans))
print("测试集中预测位置与真实位置的x坐标平均偏差为", np.mean(x_error))
print("测试集中预测位置与真实位置的y坐标平均偏差为", np.mean(y_error))
print("测试集中预测位置与真实位置的z坐标平均偏差为", np.mean(z_error))
print("X坐标R^2", r2_score(test_tag_lists_x, pred_tag_lists_x))
print("Y坐标R^2", r2_score(test_tag_lists_y, pred_tag_lists_y))
print("Z坐标R^2", r2_score(test_tag_lists_z, pred_tag_lists_z))

# 画手的位置和要预测点
# for i in range(len(train_lists)):
#     draw_finger(train_lists[i][:, 24], train_lists[i][:, 25], train_lists[i][:, 26], train_tag_lists[i])

# for i in range(len(train_lists)):
#     draw_finger(test_lists[i][:, 24], test_lists[i][:, 25], test_lists[i][:, 26], pred_tag_lists[i])

print("开始画图")
# draw_points(train_tag_lists)
# draw_point(pred_tag_lists[:10], test_tag_lists[:10])


# 预测点和目标球单独放一起
# for i in range(len(pred_tag_lists)):
#     draw_point([pred_tag_lists[i]], [test_tag_lists[i]])
# draw_point(pred_tag_lists, test_tag_lists)

draw_error(pred_tag_lists, test_tag_lists, ans)

# 结果可视化
# fig, ax = plt.subplots(4, 1)
# fig.set_size_inches(10, 4)
#
# ax[0].bar(range(len(test_tag_lists_x))[:500], test_tag_lists_x[:500], linewidth=1.5, linestyle='-', label='True')
# ax[0].bar(range(len(pred_tag_lists_x))[:500], pred_tag_lists_x[:500], linewidth=1, linestyle='-.', label='Predicted')
# ax[0].set_title("x")
#
# ax[1].bar(range(len(test_tag_lists_y))[:500], test_tag_lists_y[:500], linewidth=1.5, linestyle='-', label='True')
# ax[1].bar(range(len(pred_tag_lists_y))[:500], pred_tag_lists_y[:500], linewidth=1, linestyle='-.', label='Predicted')
# ax[1].set_title("y")
#
# ax[2].bar(range(len(test_tag_lists_z))[:500], test_tag_lists_z[:500], linewidth=1.5, linestyle='-', label='True')
# ax[2].bar(range(len(pred_tag_lists_z))[:500], pred_tag_lists_z[:500], linewidth=1, linestyle='-.', label='Predicted')
# ax[2].set_title("z")
#
# ax[3].bar(range(len(ans))[:500], zero[:500], linewidth=1.5, linestyle='-', label='True')
# ax[3].bar(range(len(ans))[:500], ans[:500], linewidth=1, linestyle='-', label='Predicted')
# ax[3].set_title("distance")
# plt.legend()
# plt.show()

# ax[0].bar(range(len(test_tag_lists_x)), test_tag_lists_x, linewidth=1.5, linestyle='-', label='True')
# ax[0].bar(range(len(pred_tag_lists_x)), pred_tag_lists_x, linewidth=1, linestyle='-.', label='Predicted')
# ax[0].set_title("x")
#
# ax[1].bar(range(len(test_tag_lists_y)), test_tag_lists_y, linewidth=1.5, linestyle='-', label='True')
# ax[1].bar(range(len(pred_tag_lists_y)), pred_tag_lists_y, linewidth=1, linestyle='-.', label='Predicted')
# ax[1].set_title("y")
#
# ax[2].bar(range(len(test_tag_lists_z)), test_tag_lists_z, linewidth=1.5, linestyle='-', label='True')
# ax[2].bar(range(len(pred_tag_lists_z)), pred_tag_lists_z, linewidth=1, linestyle='-.', label='Predicted')
# ax[2].set_title("z")
#
# ax[3].bar(range(len(ans)), zero, linewidth=1.5, linestyle='-', label='True')
# ax[3].bar(range(len(ans)), ans, linewidth=1, linestyle='-', label='Predicted')
# ax[3].set_title("distance")
# plt.legend()
# plt.show()

# for ans1 in ans:
#     print(ans1)
