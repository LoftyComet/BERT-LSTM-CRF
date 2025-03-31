import math
import time
import csv
import numpy as np
import pandas as pd
from scipy import stats

from data import load_data, divide_data, load_data2
from models.bilstm_crf import LstmModel
from models.config import LSTMConfig
from utils import load_model, draw_point, draw_finger, draw_error, draw_points
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

def get_pre(to_pre, model_name, model_index):
    """加载模型，输入时序数据，返回预测结果
    :param to_pre: 待测的数据 [1, time_step * completion_percentage, input_size]

    :return: pred_tag_lists, [[x, y, z]]
    """
    lstm_model = load_model('./train_model_saved' + str(model_index) + '/' + model_name + '.pkl')
    # lstm_model.lstm.flatten_parameters()  # remove warning
    to_pre = to_pre.to(LSTMConfig.device)
    pred_tag_lists = lstm_model.forward(to_pre)

    pred_tag_lists = pred_tag_lists.cpu().detach().numpy()

    return pred_tag_lists

def get_dis(now_point, ori_point):
    return (now_point[0] - ori_point[0]) * (now_point[0] - ori_point[0]) + (now_point[1] - ori_point[1]) * (now_point[1] - ori_point[1]) + (now_point[2] - ori_point[2]) * (now_point[2] - ori_point[2])


# (train_lists, train_tag_lists), (dev_lists, dev_tag_lists), (test_lists, test_tag_lists) = divide_data(data, tag)
for kk in range(5):
    print("第", kk, "次")
    for process in range(3, 11):
        percentage = process / 10
        if process == 10:
            percentage = 0.99
        print("percentage:", percentage)

        test_lists, test_tag_lists = load_data2([17, 18, 19, 20] , percentage=percentage)
        total_error = []
        # for k in range(150, 200):
        for k in range(1, 200):
            trigger_dis = []
            tra_dis = []
            pred_tag_lists = get_pre(test_lists, "lstm" + str(k), kk)
            # for i, test_list in enumerate(test_lists):
                # 预测时手的位置
                # fingerX = test_list[-1][24]
                # fingerY = test_list[-1][25]
                # fingerZ = test_list[-1][26]
                # temp = math.sqrt(
                #     (test_tag_lists[i][0] - fingerX) ** 2 + (test_tag_lists[i][1] - fingerY) ** 2 + (
                #             test_tag_lists[i][2] - fingerZ) ** 2)
                # trigger_dis.append(temp)
                # temp2 = math.sqrt(
                #     (test_list[0][24] - fingerX) ** 2 + (test_list[0][25] - fingerY) ** 2 + (
                #             test_list[0][26] - fingerZ) ** 2)
                # tra_dis.append(temp2)
            # print(np.mean(trigger_dis))
            # print(np.mean(tra_dis))

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

                # temp2 = math.sqrt(
                #     (test_lists[i][-1][24] - pred_tag_lists[i][0]) ** 2 + (test_lists[i][-1][25] - pred_tag_lists[i][1]) ** 2 + (
                #             test_lists[i][-1][26] - pred_tag_lists[i][2]) ** 2)
                # print("手距离预测点距离", temp2)
                # print(pred_tag_lists[i][3])

            # print("测试集中预测位置与真实位置的平均距离为", np.mean(ans))
            # print("x坐标平均偏差为", np.mean(x_error))
            # print("y坐标平均偏差为", np.mean(y_error))
            # print("z坐标平均偏差为", np.mean(z_error))
            # # region 创建图形和坐标轴
            # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))  # 创建 3 行 1 列的子图
            #
            # axes[0].plot(test_tag_lists_x, label='True')
            # axes[0].plot(pred_tag_lists_x, label='Predicted')
            # axes[0].set_title('X coordinate prediction result')
            # # axes[0].set_xlabel('the number of samples')
            # axes[0].set_ylabel('X')
            # axes[0].legend()
            # # axes[0].grid(True)
            #
            # axes[1].plot(test_tag_lists_y, label='True')
            # axes[1].plot(pred_tag_lists_y, label='Predicted')
            # axes[1].set_title('Y coordinate prediction result')
            # # axes[1].set_xlabel('the number of samples')
            # axes[1].set_ylabel('Y')
            # axes[1].legend()
            # # axes[1].grid(True)
            #
            # axes[2].plot(test_tag_lists_z, label='True')
            # axes[2].plot(pred_tag_lists_z, label='Predicted')
            # axes[2].set_title('Z coordinate prediction result')
            # axes[2].set_xlabel('the number of samples')
            # axes[2].set_ylabel('Z')
            # axes[2].legend()
            # # axes[2].grid(True)
            # # 调整子图布局，避免重叠
            # plt.tight_layout()
            # # 显示图形
            # plt.show()

            # endregion

            total_error.append(np.mean(ans))
        stop_epoch = total_error.index(min(total_error)) + 1
        #
        # trigger_dis = []
        # tra_dis = []
        # # 记录开始时间
        # start_time = time.time()
        pred_tag_lists = get_pre(test_lists, "lstm" + str(stop_epoch), kk)
        # for i, test_list in enumerate(test_lists):
            # 预测时手的位置
            # fingerX = test_list[-1][24]
            # fingerY = test_list[-1][25]
            # fingerZ = test_list[-1][26]
            # temp = math.sqrt(
            #     (test_tag_lists[i][0] - fingerX) ** 2 + (test_tag_lists[i][1] - fingerY) ** 2 + (
            #             test_tag_lists[i][2] - fingerZ) ** 2)
            # trigger_dis.append(temp)
            # temp2 = math.sqrt(
            #     (test_list[0][24] - fingerX) ** 2 + (test_list[0][25] - fingerY) ** 2 + (
            #             test_list[0][26] - fingerZ) ** 2)
            # tra_dis.append(temp2)
        # print(np.mean(trigger_dis))
        # print(np.mean(tra_dis))
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

        print("loss最小为第", stop_epoch, "次")
        print("测试集中预测位置与真实位置的平均距离为", np.mean(ans))
        print("----------------------------------------")
        # print("测试集中预测位置与真实位置的中位数为", np.median(ans))
        # print("测试集中预测位置与真实位置的x坐标平均偏差为", np.mean(x_error))
        # print("测试集中预测位置与真实位置的y坐标平均偏差为", np.mean(y_error))
        # print("测试集中预测位置与真实位置的z坐标平均偏差为", np.mean(z_error))
        # print("X坐标R^2", r2_score(test_tag_lists_x, pred_tag_lists_x))
        # print("Y坐标R^2", r2_score(test_tag_lists_y, pred_tag_lists_y))
        # print("Z坐标R^2", r2_score(test_tag_lists_z, pred_tag_lists_z))
        # # 创建图表
        # plt.figure(figsize=(8, 6))  # 设置图表大小
        # plt.plot(range(30, 110, 10), error_percentage, marker='o', linestyle='-', color='b')  # 绘制折线图
        #
        # # 设置图表标题和轴标签
        # plt.title('loss')
        # plt.xlabel('percentage')
        # plt.ylabel('error in test data')
        #
        # # 显示网格线
        # plt.grid(True)
        #
        # # 显示图表
        # plt.show()

        # with open(csv_file, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(error_percentage)
