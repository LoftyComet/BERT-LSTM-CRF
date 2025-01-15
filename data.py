"""
处理数据
"""
import copy
import math
from os.path import join
from codecs import open

import numpy
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from models.config import LSTMConfig
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d


def load_data(start, end, data_dir="AI_magic_data", for_train=True, step=1, percentage=1):
    """加载数据

    :param start: 开始人员编号
    :param end: 结束人员编号
    :param data_dir: 数据集文件路径
    :param step: 滑动窗口移动距离
    :return: 数值列表 标签列表
    """
    data_all = []
    tag_all = []

    # fig, ax = plt.subplots(3, 1)
    # fig.set_size_inches(10, 4)
    for dataIndex in range(start, end + 1):
    # for dataIndex in range(start, end + 1):
        if dataIndex in range(21, 31) and for_train:
            continue
        print("读取", dataIndex, "号被试数据")
        prefix = data_dir + "\\" + str(dataIndex) + "\\"
        df = pd.read_csv(prefix + "Eye.csv", encoding="utf-8").interpolate()  # 线性插值处理缺失值
        df2 = pd.read_csv(prefix + "Hand.csv", encoding="utf-8").interpolate()
        df_tag = pd.read_csv(prefix + "Round.csv", encoding="utf-8").interpolate()
        data_per_person = df.groupby('Case')
        data_per_person2 = df2.groupby('Case')
        tag_per_person = df_tag.groupby('Case')
        for i in range(1, 150 + 1):
            try:
                # 每一个人每case数据
                data_per_case = np.array(data_per_person.get_group(i).reset_index())
                data_per_case2 = np.array(data_per_person2.get_group(i).reset_index())
                tag_per_case = np.array(tag_per_person.get_group(i).reset_index())
                # eye_move_characteristic = get_characteristic(data_per_case, 6, 9)
                head_characteristic = get_characteristic(data_per_case, 12, 15)
                head_move_characteristic = get_characteristic(data_per_case, 15, 18)
                finger_characteristic = get_characteristic(data_per_case2, 63, 66)

                # 手指尖指向和头手射线连线
                # dir_char = get_dir_char(data_per_case2)
                # 全加在hand后面
                # data_per_case2 = np.hstack((data_per_case2, eye_move_characteristic))
                # data_per_case2 = np.hstack((data_per_case2, head_characteristic))
                # data_per_case2 = np.hstack((data_per_case2, head_move_characteristic))
                data_per_case2 = np.hstack((data_per_case2, finger_characteristic))


                start_idx = 0
                # 选定开始的起点是速度到底最大速度的5%
                hand_vs = data_per_case2[:, -1]
                hand_v_max = max(hand_vs)
                for idxV, hand_V in enumerate(hand_vs):
                    if hand_V > hand_v_max * 0.05:
                        start_idx = idxV
                        break
                if len(data_per_case) - start_idx >= LSTMConfig.time_step + LSTMConfig.pre_frame + 15 and percentage == 1 and len(data_per_case) < 180:  # 后半部分应大于20
                    # 只需要确定时间跨度的数据

                    # 使用滑动窗口创建多个样本
                    # for start_idx in range(round(len(data_per_case) * 0.3), len(data_per_case) - LSTMConfig.time_step, step):
                    temp1 = data_per_case2[:, 63:66][-1]
                    # if (temp1[0] * temp1[0] + (temp1[1] - 1) * (temp1[1] - 1) + (
                    #         temp1[2] - 0.2) * (temp1[2] - 0.2) < 0.4 * 0.4):
                    #     print("短小")

                    for start_idx in range(start_idx, len(data_per_case) - LSTMConfig.time_step - LSTMConfig.pre_frame - 15, step):
                        end_idx = start_idx + LSTMConfig.time_step

                        sample_data = data_per_case[start_idx:end_idx]
                        sample_data2 = data_per_case2[start_idx:end_idx]
                        tag_sample = data_per_case2[:, 63:66][end_idx - 1 + LSTMConfig.pre_frame]
                        # temp = sample_data2[:, 63:66][0]
                        # 起点在初始球3cm外的数据
                        # if (temp[0] * temp[0] + (temp[1] - 1) * (temp[1] - 1) + (
                        #         temp[2] - 0.2) * (temp[2] - 0.2) > 0.0009):
                        data_all.append(extend_data(sample_data, sample_data2))
                        # tag_sample = tag_per_case[0][9:12]
                        tag_all.append(tag_sample)  # 假设标签在这个位置
                    # data_all.append(extend_data(data_per_case, data_per_case2))
                    #  tag_all.append(tag_per_case[0][9:12])
            except KeyError:
                print("case", i, "被完全去除了")
    # 添加图例
    # plt.legend()

    # 显示图形
    # plt.grid(True)
    # plt.show()
    print("数据读取完成")
    # 转换为tensor
    # data_all_ans = torch.zeros((len(data_all), len(data_all[0]), len(data_all[0][0])))
    # for i in range(len(data_all)):
    #     for j in range(len(data_all[0])):
    #         for k in range(len(data_all[0][0])):
    #             try:
    #                 data_all_ans[i][j][k] = data_all[i][j][k]
    #             except IndexError:
    #                 data_all_ans[i][j][k] = 0
    jump_time = []
    print("开始转换input为tensor")
    data_all_ans = torch.tensor(data_all, dtype=torch.float32)
    # data_all_ans = torch.zeros((len(data_all), len(data_all[0]), len(data_all[0][0])))
    # for i in range(len(data_all)):
    #     for j in range(len(data_all[0])):
    #         for k in range(len(data_all[0][0])):
    #             data_all_ans[i][j][k] = data_all[i][j][k]

    tag_all_ans = np.array(tag_all)[:, 8:11]
    # tag 带时序
    # tag_all_ans = torch.ones((len(tag_all_ans), LSTMConfig.time_step, len(tag_all[0])))
    # for i in range(len(tag_all)):
    #     for j in range(LSTMConfig.time_step):
    #         for k in range(len(tag_all[0])):
    #             tag_all_ans[i][j][k] = tag_all[i][k]
    print("开始转换tag为tensor")
    # tag不带时序
    tag_all_ans = torch.zeros((len(tag_all_ans), len(tag_all[0])))
    for i in range(len(tag_all)):
        for k in range(len(tag_all[0])):
            tag_all_ans[i][k] = tag_all[i][k]


    return data_all_ans, tag_all_ans  # [num, time_step, input_size]


def load_data2(loadList,data_dir="AI_magic_data", for_train=True, step=1, percentage=1):
    """加载数据

    :param start: 开始人员编号
    :param end: 结束人员编号
    :param data_dir: 数据集文件路径
    :param step: 滑动窗口移动距离
    :return: 数值列表 标签列表
    """
    data_all = []
    tag_all = []

    to_draw = []
    in_point_index = []
    # fig, ax = plt.subplots(3, 1)
    # fig.set_size_inches(10, 4)
    for dataIndex in loadList:
    # for dataIndex in range(start, end + 1):

        print("读取", dataIndex, "号被试数据")
        prefix = data_dir + "\\" + str(dataIndex) + "\\"
        df = pd.read_csv(prefix + "Eye.csv", encoding="utf-8").interpolate()  # 线性插值处理缺失值
        df2 = pd.read_csv(prefix + "Hand.csv", encoding="utf-8").interpolate()
        df_tag = pd.read_csv(prefix + "Round.csv", encoding="utf-8").interpolate()
        data_per_person = df.groupby('Case')
        data_per_person2 = df2.groupby('Case')
        tag_per_person = df_tag.groupby('Case')
        for i in range(1, 150 + 1):
            try:
                # 每一个人每case数据
                data_per_case = np.array(data_per_person.get_group(i).reset_index())
                data_per_case2 = np.array(data_per_person2.get_group(i).reset_index())
                tag_per_case = np.array(tag_per_person.get_group(i).reset_index())
                # eye_move_characteristic = get_characteristic(data_per_case, 6, 9)
                head_characteristic = get_characteristic(data_per_case, 12, 15)
                head_move_characteristic = get_characteristic(data_per_case, 15, 18)
                finger_characteristic = get_characteristic(data_per_case2, 63, 66)

                # 手指尖指向和头手射线连线
                # dir_char = get_dir_char(data_per_case2)
                # 全加在hand后面
                # data_per_case2 = np.hstack((data_per_case2, eye_move_characteristic))
                # data_per_case2 = np.hstack((data_per_case2, head_characteristic))
                # data_per_case2 = np.hstack((data_per_case2, head_move_characteristic))
                data_per_case2 = np.hstack((data_per_case2, finger_characteristic))
                # data_per_case2 = np.hstack((data_per_case2, dir_char))

                # selected_columns = [-1]  # 选择列
                # # selected_data = data_per_case2[:, selected_columns]
                #
                # selected_data = data_per_case2[:, selected_columns].T  # 选择所有行和指定列
                # # 使用高斯滤波进行平滑处理
                # sigma = 2  # 标准差，决定平滑的程度
                # selected_data = gaussian_filter1d(selected_data, sigma).T
                # to_draw.append(selected_data)
                # temp = data_per_case2[:, 63:66]
                # tag_sample = tag_per_case[0][9:12]
                # for fingerI in range(len(temp)):
                #     if ((temp[fingerI][0] - tag_sample[0]) * (temp[fingerI][0] - tag_sample[0]) + (temp[fingerI][1] - tag_sample[1]) * (temp[fingerI][1] - tag_sample[1]) + (temp[fingerI][2] - tag_sample[2]) * (temp[fingerI][2] - tag_sample[2]) < 0.0004):
                #         in_point_index.append(fingerI)
                #         break


                start_idx = 0
                # 选定开始的起点是速度到底最大速度的5%
                hand_vs = data_per_case2[:, -2]
                hand_v_max = max(hand_vs)
                for idxV, hand_V in enumerate(hand_vs):
                    if hand_V > hand_v_max * 0.05:
                        start_idx = idxV
                        break
                if len(data_per_case) - start_idx >= LSTMConfig.time_step + LSTMConfig.pre_frame + 15 and percentage == 1 and len(data_per_case) < 180:  # 后半部分应大于20
                    # fig = plt.figure(figsize=(10, 8))
                    # ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(-data_per_case2[:, 63][0], data_per_case2[:, 64][0], data_per_case2[:, 65][0], c='black', s=100)
                    # # 绘制散点图
                    # scatter = ax.scatter(-data_per_case2[:, 63], data_per_case2[:, 64], data_per_case2[:, 65], c=data_per_case2[:, -1],
                    #                      cmap='viridis', alpha=0.6)
                    #
                    # # 设置标题和轴标签
                    # ax.set_title('hand movement')
                    # ax.set_xlabel('X')
                    # ax.set_ylabel('Y')
                    # ax.set_zlabel('Z')
                    #
                    # # 添加颜色条
                    # plt.colorbar(scatter)
                    #
                    # # 显示图形
                    # plt.show()


                    for start_idx in range(start_idx, len(data_per_case) - LSTMConfig.time_step - LSTMConfig.pre_frame - 15, step):
                        end_idx = start_idx + LSTMConfig.time_step

                        # region 获取数据
                        sample_data = data_per_case[start_idx:end_idx]
                        sample_data2 = data_per_case2[start_idx:end_idx]
                        tag_sample = data_per_case2[:, 63:66][end_idx - 1 + LSTMConfig.pre_frame]
                        temp = sample_data2[:, 63:66][0]
                        merged_data = extend_data(sample_data, sample_data2)
                        # endregion
                        # if start_idx + 5 == len(data_per_case) - LSTMConfig.time_step - LSTMConfig.pre_frame - 15:
                        #     # region 可视化滑动窗口平滑
                        #     selected_columns = [-2]  # 选择列
                        #     selected_data = sample_data2[:, selected_columns]
                        #     # selected_data = sample_data2[:, selected_columns].T  # 选择所有行和指定列
                        #     # # 使用高斯滤波进行平滑处理
                        #     # sigma = 2  # 标准差，决定平滑的程度
                        #     # selected_data = gaussian_filter1d(selected_data, sigma).T
                        #     to_draw.append(selected_data)
                        #     temp = sample_data2[:, 63:66]
                        #     tag_sample = tag_per_case[0][9:12]
                        #     for fingerI in range(len(temp)):
                        #         if ((temp[fingerI][0] - tag_sample[0]) * (temp[fingerI][0] - tag_sample[0]) + (
                        #                 temp[fingerI][1] - tag_sample[1]) * (temp[fingerI][1] - tag_sample[1]) + (
                        #                 temp[fingerI][2] - tag_sample[2]) * (temp[fingerI][2] - tag_sample[2]) < 0.0004):
                        #             in_point_index.append(fingerI)
                        #             break
                            # endregion

                        # fig = plt.figure(figsize=(10, 8))
                        # ax = fig.add_subplot(111, projection='3d')
                        # ax.scatter(-sample_data2[:, 63][0], sample_data2[:, 64][0], sample_data2[:, 65][0], c='black', s=100)
                        # ax.scatter(-tag_sample[0], tag_sample[1], tag_sample[2], c='red',
                        #            s=100)
                        # # 绘制散点图
                        # scatter = ax.scatter(-sample_data2[:, 63], sample_data2[:, 64], sample_data2[:, 65], c=sample_data2[:, -1],
                        #                      cmap='viridis', alpha=0.6)
                        #
                        # # 设置标题和轴标签
                        # ax.set_title('hand movement')
                        # ax.set_xlabel('X')
                        # ax.set_ylabel('Y')
                        # ax.set_zlabel('Z')
                        #
                        # # 添加颜色条
                        # plt.colorbar(scatter)
                        #
                        # # 显示图形
                        # plt.show()

                        data_all.append(merged_data)
                        # tag_sample = tag_per_case[0][9:12]
                        tag_all.append(tag_sample)  # 假设标签在这个位置
                    # data_all.append(extend_data(data_per_case, data_per_case2))
                    #  tag_all.append(tag_per_case[0][9:12])

                else:
                    # pass
                    print("太短辣！")
            except KeyError:
                print("case", i, "被完全去除了")
    # 添加图例
    # plt.legend()

    # 显示图形
    # plt.grid(True)
    # plt.show()
    print("数据读取完成")
    # 转换为tensor
    # data_all_ans = torch.zeros((len(data_all), len(data_all[0]), len(data_all[0][0])))
    # for i in range(len(data_all)):
    #     for j in range(len(data_all[0])):
    #         for k in range(len(data_all[0][0])):
    #             try:
    #                 data_all_ans[i][j][k] = data_all[i][j][k]
    #             except IndexError:
    #                 data_all_ans[i][j][k] = 0
    jump_time = []
    print("开始转换input为tensor")
    data_all_ans = torch.tensor(data_all, dtype=torch.float32)
    # data_all_ans = torch.zeros((len(data_all), len(data_all[0]), len(data_all[0][0])))
    # for i in range(len(data_all)):
    #     for j in range(len(data_all[0])):
    #         for k in range(len(data_all[0][0])):
    #             data_all_ans[i][j][k] = data_all[i][j][k]

    tag_all_ans = np.array(tag_all)[:, 8:11]
    # tag 带时序
    # tag_all_ans = torch.ones((len(tag_all_ans), LSTMConfig.time_step, len(tag_all[0])))
    # for i in range(len(tag_all)):
    #     for j in range(LSTMConfig.time_step):
    #         for k in range(len(tag_all[0])):
    #             tag_all_ans[i][j][k] = tag_all[i][k]
    print("开始转换tag为tensor")
    # tag不带时序
    tag_all_ans = torch.zeros((len(tag_all_ans), len(tag_all[0])))
    for i in range(len(tag_all)):
        for k in range(len(tag_all[0])):
            tag_all_ans[i][k] = tag_all[i][k]

    # # region 可视化所有人速度加速度
    # # 设置图表样式
    # plt.figure(figsize=(10, 6))
    # colors = plt.cm.tab10(np.linspace(0, 1, len(to_draw)))  # 自动生成不同颜色
    #
    # # 绘制每条折线
    # for i, line_data in enumerate(to_draw):
    #     if i > 5:
    #         break
    #     plt.plot(line_data,
    #              # marker='o',  # 数据点标记
    #              linestyle='-',  # 线型
    #              label=f'线{i + 1}',  # 图例标签
    #              # color=colors[i]  # 线条颜色
    #              )
    #     # plt.plot(in_point_index[i], line_data[in_point_index[i]],
    #     #          marker='o',  # 数据点标记
    #     #          color='black'  # 线条颜色
    #     #          )
    #
    # # 添加图表元素
    # plt.title('V of finger during each selection process', fontsize=14)
    # plt.xlabel('X', fontsize=12)
    # plt.ylabel('Y', fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # # plt.legend()
    # plt.show()
    # endregion
    return data_all_ans, tag_all_ans  # [num, time_step, input_size]

def divide_data(data, tag):
    """划分训练测试和验证集

    :param data: 要划分的数据
    :param tag: 要划分的标签
    :return:
    """
    train_data, test_data = train_test_split(data, train_size=0.8, random_state=111)
    train_data, dev_data = train_test_split(train_data, train_size=0.9, random_state=111)

    train_tag, test_tag = train_test_split(tag, train_size=0.8, random_state=111)
    train_tag, dev_tag = train_test_split(train_tag, train_size=0.9, random_state=111)
    # train_data, dev_data = train_test_split(data, train_size=0.9, random_state=111)
    # train_tag, dev_tag = train_test_split(tag, train_size=0.9, random_state=111)
    # return (train_data, train_tag), (dev_data, dev_tag)
    return (train_data, train_tag), (dev_data, dev_tag), (test_data, test_tag)


def extend_data(eye, hand):
    """将两个模态的数据合并

    :param eye:
    :param hand:
    :return:
    """
    # 按列取 左闭右开 要+1，因为取数时前面会多一行序号，还要+2，因为有两行轮次序号
    # t1 = eye[:, 6:9]
    #
    #
    # t12 = eye[:, 12:18]
    # t1 = np.hstack((t1, t12))
    # 在手部数据再筛选一些，只需要手掌手腕和最重要的食指数据
    # t2 = hand[:, 3:9]
    # t22 = hand[:, 21:30]
    # t2 = np.hstack((t2, t22))
    t100 = hand[:, 63:66]
    # print(t3)
    # p = pv.Plotter()
    # for qq in range(len(t3)):
    #     point = [t3[qq][0], t3[qq][1], t3[qq][2]]
    #     mesh = pv.PolyData(point)  # PolyData对象的实例化
    #     p.add_mesh(mesh, point_size=5)
    #
    # p.camera_position = 'xy'
    # p.show_grid()
    # p.show(cpos="xy")
    # t5 = hand[:, -13:-10]
    # t3 = np.hstack((t3, t5))
    t4 = hand[:, -5:]
    # t4 = hand[:, -1:]
    # print(t4)
    # t2 = np.hstack((t2, t3))
    # t2 = np.hstack((t2, t4))
    # return np.hstack((t1, t2))

    #
    # t6 = eye[:, 3:6]
    # t1 = eye[:, 9:12]
    # t2 = eye[:, 15:18]
    # t1 = t1 + t2
    # t1 = np.hstack((t1, t6))
    t4 = np.hstack((t100, t4))
    # t4 = np.hstack((t1, t4))
    # t5 = eye[:, 12:18]
    # t4 = np.hstack((t5, t4))
    # t4 = np.hstack((t2, t4))
    return t4
    # return t100


def get_characteristic(ori_data, start, end):
    """添加手部特征

    :param ori_data:要添加的数据
    :param start:开始的序号
    :param end:结束的序号

    :return:添加后的特征附在最后
    """
    t1 = ori_data[:, start:end]
    t2 = np.zeros((len(t1), 5), dtype=float)
    # 前两组数据不足，速度加速度都是0
    for i in range(1, len(t1)):
        # t2[i][0] = math.sqrt(pow((t1[i-1][0] - t1[i][0]), 2) + pow((t1[i-1][1] - t1[i][1]), 2) + pow((t1[i-1][2] - t1[i][2]), 2)) / 0.02
        # print(t2[i][0])
        # t2[i][1] = math.sqrt(pow(((t1[i][0] - t1[i-1][0]) / 0.02 - (t1[i-1][0] - t1[i-2][0]) / 0.02), 2) + pow(((t1[i][1] - t1[i-1][1]) / 0.02 - (t1[i-1][1] - t1[i-2][1]) / 0.02), 2) + pow(((t1[i][2] - t1[i-1][2]) / 0.02 - (t1[i-1][2] - t1[i-2][2]) / 0.02), 2)) / 0.02
        # print(t2[i][1])
        # 这3列是方向
        t2[i][0] = t1[i][0] - t1[i - 1][0]
        t2[i][1] = t1[i][1] - t1[i - 1][1]
        t2[i][2] = t1[i][2] - t1[i - 1][2]
        t2[i][3] = math.sqrt(pow((t1[i-1][0] - t1[i][0]), 2) + pow((t1[i-1][1] - t1[i][1]), 2) + pow((t1[i-1][2] - t1[i][2]), 2)) / 0.02
        # t2[i][4] = math.sqrt(pow(((t1[i][0] - t1[i-1][0]) / 0.02 - (t1[i-1][0] - t1[i-2][0]) / 0.02), 2) + pow(((t1[i][1] - t1[i-1][1]) / 0.02 - (t1[i-1][1] - t1[i-2][1]) / 0.02), 2) + pow(((t1[i][2] - t1[i-1][2]) / 0.02 - (t1[i-1][2] - t1[i-2][2]) / 0.02), 2)) / 0.02
    for i in range(2, len(t1)):
        t2[i][4] = (t2[i][3] - t2[i - 1][3]) / 0.02
    # print(t2[i][0])

    # # 绘制函数曲线
    # ax[0].plot(range(len(t2)), t2[:, 0], label='V')
    # ax[0].set_title("V")
    # ax[1].plot(range(len(t2)), t2[:, 1], label='A')
    # ax[1].set_title("A")
    # ax[2].plot(range(len(t1)), t1[:, 0], label='X')
    # ax[2].plot(range(len(t1)), t1[:, 1], label='Y')
    # ax[2].plot(range(len(t1)), t1[:, 2], label='Z')
    # if start == 6:
    #     ax[2].set_title("eye_move_Loc")
    # elif start == 12:
    #     ax[2].set_title("head_Loc")
    # elif start == 15:
    #     ax[2].set_title("head_move_Loc")
    # elif start == 63:
    #     ax[0].set_title("X")
    #     ax[1].set_title("Y")
    #     ax[2].set_title("Z")
    #     ax[0].plot(range(len(t1)), t1[:, 0])
    #     ax[1].plot(range(len(t1)), t1[:, 1])
    #     ax[2].plot(range(len(t1)), t1[:, 2])
    # # 添加图例
    # plt.legend()
    #
    # # 显示图形
    # plt.grid(True)
    # plt.show()

    # 使用高斯滤波进行平滑处理
    sigma = 1  # 标准差，决定平滑的程度
    t2[:, 0] = gaussian_filter1d(t2[:, 0], sigma)
    t2[:, 1] = gaussian_filter1d(t2[:, 1], sigma)
    return t2

def get_dir_char(ori_data):
    head_loc = ori_data[:, 12:15]
    index_loc = ori_data[:, 63:66]
    t2 = np.zeros((len(head_loc), 3), dtype=float)
    for i in range(len(head_loc)):
        t2[i][0] = index_loc[i][0] - head_loc[i][0]
        t2[i][1] = index_loc[i][1] - head_loc[i][1]
        t2[i][2] = index_loc[i][2] - head_loc[i][2]
    # 使用高斯滤波进行平滑处理
    sigma = 1  # 标准差，决定平滑的程度
    t2[:, 0] = gaussian_filter1d(t2[:, 0], sigma)
    t2[:, 1] = gaussian_filter1d(t2[:, 1], sigma)
    t2[:, 2] = gaussian_filter1d(t2[:, 2], sigma)
    return t2

