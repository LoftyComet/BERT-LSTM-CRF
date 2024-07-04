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
import pyvista as pv

def del_tensor_ele_n(arr, index, n):
    """
    arr: 输入tensor
    index: 需要删除位置的索引
    n: 从index开始，需要删除的行数
    """
    arr1 = arr[0:index]
    arr2 = arr[index + n:]
    return torch.cat((arr1, arr2), dim=0)


def load_data(start, end, data_dir="AI_magic_data", for_train=True, step=1):
    """加载数据

    :param start: 开始人员编号
    :param end: 结束人员编号
    :param data_dir: 数据集文件路径
    :param step: 滑动窗口移动距离
    :return: 数值列表 标签列表
    """
    data_all = []
    tag_all = []

    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(10, 4)

    for dataIndex in range(start, end + 1) or [21, 24, 27, 30, 31]:
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
                head_characteristic = get_characteristic(data_per_case, 12, 15, ax)
                head_move_characteristic = get_characteristic(data_per_case, 15, 18, ax)
                finger_characteristic = get_characteristic(data_per_case2, 63, 66, ax)

                # 手指尖指向和头手射线连线
                # dir_char = get_dir_char(data_per_case2)
                # 全加在hand后面
                # data_per_case2 = np.hstack((data_per_case2, eye_move_characteristic))
                data_per_case2 = np.hstack((data_per_case2, head_characteristic))
                data_per_case2 = np.hstack((data_per_case2, head_move_characteristic))
                data_per_case2 = np.hstack((data_per_case2, finger_characteristic))
                # data_per_case2 = np.hstack((data_per_case2, dir_char))
                # 前六组数据停止时间有问题，矫正一下
                # if dataIndex in range(1, 8) or dataIndex in range(21, 50):
                # if dataIndex in range(1, 8):
                if len(data_per_case) >= LSTMConfig.time_step:
                    # 只需要确定时间跨度的数据
                    # data_per_case = data_per_case[-(1 + LSTMConfig.time_step + 12):-1 - 12 - round(
                    #     (1 - LSTMConfig.completion_percentage) * LSTMConfig.time_step)]
                    # data_per_case2 = data_per_case2[-(1 + LSTMConfig.time_step + 12):-1 - 12 - round(
                    #     (1 - LSTMConfig.completion_percentage) * LSTMConfig.time_step)]

                    # 使用滑动窗口创建多个样本
                    # for start_idx in range(0, len(data_per_case) - round(LSTMConfig.time_step) - 12, step):
                    for start_idx in range(0, len(data_per_case) - LSTMConfig.time_step, step):
                        end_idx = start_idx + LSTMConfig.time_step

                        sample_data = data_per_case[start_idx:end_idx]
                        sample_data2 = data_per_case2[start_idx:end_idx]

                        temp = sample_data2[:, 63:66][0]
                        # 起点在初始球3cm外的数据
                        if (temp[0] * temp[0] + (temp[1] - 1) * (temp[1] - 1) + (
                                temp[2] - 0.2) * (temp[2] - 0.2) > 0.0009):
                            # 假设extend_data函数能够处理两个数据帧
                            data_all.append(extend_data(sample_data, sample_data2))
                            tag_sample = tag_per_case[0][9:12]
                            # tag_sample = data_per_case2[start_idx + 12 - 1][63:66]

                            tag_all.append(tag_sample)  # 假设标签在这个位置
                        # else:
                        #     print("犹豫就会被淘汰")

                    # data_all.append(extend_data(data_per_case, data_per_case2))
                    #  tag_all.append(tag_per_case[0][9:12])
                else:
                    print("太短辣！")
                # else:
                #     if len(data_per_case) >= LSTMConfig.time_step:
                #         # 只需要确定时间跨度的数据
                #         # data_per_case = data_per_case[-(1 + LSTMConfig.time_step):-1-round((1 - LSTMConfig.completion_percentage) * LSTMConfig.time_step)]
                #         # data_per_case2 = data_per_case2[-(1 + LSTMConfig.time_step):-1-round((1 - LSTMConfig.completion_percentage) * LSTMConfig.time_step)]
                #         #
                #         # data_all.append(extend_data(data_per_case, data_per_case2))
                #         # tag_all.append(tag_per_case[0][9:12])
                #
                #         # 使用滑动窗口创建多个样本
                #         for start_idx in range(0, len(data_per_case) - LSTMConfig.time_step, step):
                #             end_idx = start_idx + LSTMConfig.time_step
                #
                #             sample_data = data_per_case[start_idx:end_idx]
                #             sample_data2 = data_per_case2[start_idx:end_idx]
                #
                #             temp = sample_data2[:, 63:66][0]
                #             if (temp[0] * temp[0] + (temp[1] - 1) * (temp[1] - 1) + (
                #                     temp[2] - 0.2) * (temp[2] - 0.2) > 0.0025):
                #                 # 假设extend_data函数能够处理两个数据帧
                #                 data_all.append(extend_data(sample_data, sample_data2))
                #                 tag_sample = tag_per_case[0][9:12]
                #
                #                 # if ((temp[0] - tag_sample[0]) * (temp[0] - tag_sample[0]) + (temp[1] - tag_sample[1]) * (temp[1] - tag_sample[1]) + (
                #                 #     temp[2] - tag_sample[2]) * (temp[2] - tag_sample[2]) < 0.0064):
                #                 #     tag_sample = np.hstack((tag_sample, [1]))
                #                 # else:
                #                 #     tag_sample = np.hstack((tag_sample, [0]))
                #
                #                 tag_all.append(tag_sample)  # 假设标签在这个位置
                #             # else:
                #             #     print("犹豫就会被淘汰")
                #     else:
                #         print("太短辣！")
            except KeyError:
                print("case", i, "被完全去除了")
    # 添加图例
    plt.legend()

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

    # !!!分开按照各个特征归一化
    # for i in range(data_all_ans.shape[-1]):
    #     temp = data_all_ans[..., i].shape
    #     temp1 = data_all_ans[..., i].flatten()
    #     data_all_ans[..., i] = normalize_data(data_all_ans[..., i].reshape(len(temp1), 1)).reshape(temp)
    #
    # for i in range(3):
    #     temp = tag_all_ans[:, i].shape
    #     temp1 = tag_all_ans[:, i].flatten()
    #     tag_all_ans[:, i] = normalize_data(tag_all_ans[:, i].reshape(len(temp1), 1)).reshape(temp)
    # for i in range(len(data_all_ans)):
    #     for j in range(len(data_all_ans[0])):
    #         print(data_all_ans[:, j].shape)
    #         data_all_ans[:, j] = normalize_data(data_all_ans[:, j])
    # for i in range(len(tag_all_ans)):
    #     for j in range(len(tag_all_ans[0])):
    #         tag_all_ans[:, j] = normalize_data(tag_all_ans[:, j])
    # return normalize_data(data_all_ans), normalize_data(tag_all_ans)
    # for i in jump_time:
    #     del_tensor_ele_n(data_all_ans, i, 1)
    #     del_tensor_ele_n(tag_all_ans, i, 1)
    # print(data_all_ans)

    # 计算每个特征的均值和标准差
    # if for_train:
    #     mean = data_all_ans.mean(dim=(0, 1), keepdim=True)
    #     std = data_all_ans.std(dim=(0, 1), keepdim=True)
    #     print(mean)
    #     print(std)
    # else:
    #     mean = torch.Tensor([[[ 1.0402e-01],
    #      [ 8.7239e-03],
    #      [-1.1163e-03],
    #      [-2.3749e-02],
    #      [ 1.1742e+00],
    #      [ 3.7552e-02],
    #      [-4.0705e-02],
    #      [-2.4295e-02],
    #      [ 8.7753e-01],
    #      [ 4.3618e-02],
    #      [ 9.1677e-01],
    #      [ 2.8998e-01],
    #      [ 4.3618e-02],
    #      [ 9.1677e-01],
    #      [ 2.8998e-01],
    #      [ 9.4858e-03],
    #      [ 9.7814e-01],
    #      [ 3.3738e-01],
    #      [-3.6412e-03],
    #      [ 9.9097e-01],
    #      [ 3.6269e-01],
    #      [-1.4259e-02],
    #      [ 9.9711e-01],
    #      [ 3.7847e-01],
    #      [-2.2310e-02],
    #      [ 1.0026e+00],
    #      [ 3.9385e-01],
    #      [ 4.5608e-01],
    #      [ 2.7237e-01],
    #      [ 2.3369e+01],
    #      [ 1.1020e-01],
    #      [ 3.5618e+00],
    #      [ 3.1868e-01],
    #      [ 1.0817e+01],
    #      [ 5.3727e-01],
    #      [ 2.0563e+01],
    #      [-3.1348e-02],
    #      [ 6.2575e-02]]])
    #     std = torch.Tensor([[[1.1563e-01],
    #      [7.6606e-02],
    #      [1.0528e-02],
    #      [7.2178e-02],
    #      [4.1436e-02],
    #      [4.3244e-02],
    #      [3.7594e-01],
    #      [2.8140e-01],
    #      [8.4710e-02],
    #      [1.9737e-01],
    #      [1.9295e-01],
    #      [9.3123e-02],
    #      [1.9737e-01],
    #      [1.9295e-01],
    #      [9.3123e-02],
    #      [2.2512e-01],
    #      [2.1791e-01],
    #      [9.3710e-02],
    #      [2.3768e-01],
    #      [2.3081e-01],
    #      [9.5652e-02],
    #      [2.4545e-01],
    #      [2.3928e-01],
    #      [9.7491e-02],
    #      [2.5299e-01],
    #      [2.4740e-01],
    #      [9.9106e-02],
    #      [9.4068e-02],
    #      [3.7188e-01],
    #      [3.0609e+01],
    #      [7.8751e-02],
    #      [3.1457e+00],
    #      [4.8095e-01],
    #      [1.9712e+01],
    #      [3.7278e-01],
    #      [1.4614e+01],
    #      [4.9886e-02],
    #      [5.1240e-02]]])
    # std += 1e-6
    # data_all_ans_normalized = (data_all_ans - mean) / std
    # # 为防止除以零，可以添加一个小的常数
    #
    # return data_all_ans_normalized, tag_all_ans  # [num, time_step, input_size]
    # # print("数据处理完成")
    return data_all_ans, tag_all_ans  # [num, time_step, input_size]


def divide_data(data, tag):
    """划分训练测试和验证集

    :param data: 要划分的数据
    :param tag: 要划分的标签
    :return:
    """
    train_data, test_data = train_test_split(data, train_size=0.8, random_state=111)
    train_data, dev_data = train_test_split(train_data, train_size=0.75, random_state=111)
    train_tag, test_tag = train_test_split(tag, train_size=0.8, random_state=111)
    train_tag, dev_tag = train_test_split(train_tag, train_size=0.75, random_state=111)
    return (train_data, train_tag), (test_data, test_tag), (dev_data, dev_tag)


def extend_data(eye, hand):
    """将两个模态的数据合并

    :param eye:
    :param hand:
    :return:
    """
    # 按列取 左闭右开 要+1，因为取数时前面会多一行序号，还要+2，因为有两行轮次序号
    t1 = eye[:, 6:9]
    p = pv.Plotter()

    t12 = eye[:, 12:18]
    t1 = np.hstack((t1, t12))
    # 在手部数据再筛选一些，只需要手掌手腕和最重要的食指数据
    t2 = hand[:, 3:9]
    t22 = hand[:, 21:30]
    t2 = np.hstack((t2, t22))
    t3 = hand[:, 63:66]
    #
    # for qq in range(len(t3)):
    #     point = [t3[qq][0], t3[qq][1], t3[qq][2]]
    #     mesh = pv.PolyData(point)  # PolyData对象的实例化
    #     p.add_mesh(mesh, point_size=5)
    #
    # p.camera_position = 'xy'
    # p.show_grid()
    # p.show(cpos="xy")
    t5 = hand[:, -13:-10]
    t3 = np.hstack((t3, t5))
    t4 = hand[:, -7:-1]
    t2 = np.hstack((t2, t3))
    t2 = np.hstack((t2, t4))
    return np.hstack((t1, t2))


def normalize_data(ori_data):
    """数据归一化

    :param ori_data:
    :return:
    """
    normalized_data = torch.nn.functional.normalize(ori_data.float(), p=1, dim=1)
    return normalized_data


def get_characteristic(ori_data, start, end, ax):
    """添加手部特征

    :param ori_data:要添加的数据
    :param start:开始的序号
    :param end:结束的序号

    :return:添加后的特征附在最后
    """
    t1 = ori_data[:, start:end]
    t2 = np.zeros((len(t1), 8), dtype=float)
    # 前两组数据不足，速度加速度都是0
    for i in range(2, len(t1)):
        t2[i][0] = math.sqrt(pow((t1[i-1][0] - t1[i][0]), 2) + pow((t1[i-1][1] - t1[i][1]), 2) + pow((t1[i-1][2] - t1[i][2]), 2)) / 0.02
        t2[i][1] = math.sqrt(pow(((t1[i][0] - t1[i-1][0]) / 0.02 - (t1[i-1][0] - t1[i-2][0]) / 0.02), 2) + pow(((t1[i][1] - t1[i-1][1]) / 0.02 - (t1[i-1][1] - t1[i-2][1]) / 0.02), 2) + pow(((t1[i][2] - t1[i-1][2]) / 0.02 - (t1[i-1][2] - t1[i-2][2]) / 0.02), 2)) / 0.02
        # 这3列是方向
        # t2[i][2] = t1[i][0] - t1[i - 1][0]
        # t2[i][3] = t1[i][1] - t1[i - 1][1]
        # t2[i][4] = t1[i][2] - t1[i - 1][2]


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

# data, tag = load_data(1, 4)

