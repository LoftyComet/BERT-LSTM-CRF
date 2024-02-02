"""
处理数据
"""
import math
from os.path import join
from codecs import open

import numpy
import numpy as np
import pandas as pd
import torch

from models.config import LSTMConfig
from sklearn.model_selection import train_test_split


def load_data(start, end, data_dir="AI_magic_data"):
    """加载数据

    :param start: 开始人员编号
    :param end: 结束人员编号
    :param data_dir: 数据集文件路径
    :return: 数值列表 标签列表
    """
    data_all = []
    tag_all = []
    for dataIndex in range(start, end + 1):
        prefix = data_dir + "\\" + str(dataIndex) + "\\"
        df = pd.read_csv(prefix + "Eye.csv", encoding="utf-8")
        df2 = pd.read_csv(prefix + "Hand.csv", encoding="utf-8")
        df_tag = pd.read_csv(prefix + "Round.csv", encoding="utf-8")
        data_per_person = df.groupby('Case')
        data_per_person2 = df2.groupby('Case')
        tag_per_person = df_tag.groupby('Case')
        for i in range(1, 150 + 1):
            try:
                # 每一个人每case数据
                data_per_case = np.array(data_per_person.get_group(i).reset_index())
                data_per_case2 = np.array(data_per_person2.get_group(i).reset_index())
                tag_per_case = np.array(tag_per_person.get_group(i).reset_index())
                eye_characteristic = get_characteristic(data_per_case, 2, 5)
                eye_move_characteristic = get_characteristic(data_per_case, 5, 8)
                head_characteristic = get_characteristic(data_per_case, 11, 14)
                finger_characteristic = get_characteristic(data_per_case2, 32, 35)
                # 全加在hand后面
                data_per_case2 = np.hstack((data_per_case2, eye_characteristic))
                data_per_case2 = np.hstack((data_per_case2, eye_move_characteristic))
                data_per_case2 = np.hstack((data_per_case2, head_characteristic))
                data_per_case2 = np.hstack((data_per_case2, finger_characteristic))
                if len(data_per_case) >= LSTMConfig.time_step:
                    # 只需要确定时间跨度的数据
                    data_per_case = data_per_case[-(1 + LSTMConfig.time_step):-1-int((1 - LSTMConfig.completion_percentage) * LSTMConfig.time_step)]
                    data_per_case2 = data_per_case2[-(1 + LSTMConfig.time_step):-1-int((1 - LSTMConfig.completion_percentage) * LSTMConfig.time_step)]

                    data_all.append(extend_data(data_per_case, data_per_case2))
                    tag_all.append(tag_per_case[0][9:12])
                else:
                    print("太短辣！")
            except KeyError:
                print("case", i, "被完全去除了")

    # 转换为tensor
    data_all_ans = torch.zeros((len(data_all), len(data_all[0]), len(data_all[0][0])))
    for i in range(len(data_all)):
        for j in range(len(data_all[0])):
            for k in range(len(data_all[0][0])):
                try:
                    data_all_ans[i][j][k] = data_all[i][j][k]
                except IndexError:
                    data_all_ans[i][j][k] = 0
    tag_all_ans = np.array(tag_all)[:, 8:11]
    # tag 带时序
    # tag_all_ans = torch.ones((len(tag_all_ans), LSTMConfig.time_step, len(tag_all[0])))
    # for i in range(len(tag_all)):
    #     for j in range(LSTMConfig.time_step):
    #         for k in range(len(tag_all[0])):
    #             tag_all_ans[i][j][k] = tag_all[i][k]

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
    return data_all_ans, tag_all_ans


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
    # 按列取 左开右闭
    t1 = eye[:, 2:17]
    # 在手部数据再筛选一些，只需要手掌手腕和最重要的食指数据
    t2 = hand[:, 2:8]
    t3 = hand[:, 20:35]  # 2:68
    t4 = hand[:, -9:-1]
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


def get_characteristic(ori_data, start, end):
    """添加手部特征

    :param ori_data:要添加的数据
    :param start:开始的序号
    :param end:结束的序号

    :return:添加后的特征附在最后
    """
    t1 = ori_data[:, start:end]
    t2 = np.zeros((len(t1), 2), dtype=float)
    # 前两组数据不足，速度加速度都是0
    for i in range(2, len(t1)):
        t2[i][0] = math.sqrt((t1[i-1][0] - t1[i][0])**2 + (t1[i-1][1] - t1[i][1])**2 + (t1[i-1][2] - t1[i][2])**2) / 0.02
        t2[i][1] = math.sqrt(((t1[i][0] - t1[i-1][0]) / 0.02 - (t1[i-1][0] - t1[i-2][0]) / 0.02)**2 + ((t1[i][1] - t1[i-1][1]) / 0.02 - (t1[i-1][1] - t1[i-2][1]) / 0.02)**2 + ((t1[i][2] - t1[i-1][2]) / 0.02 - (t1[i-1][2] - t1[i-2][2]) / 0.02)**2) / 0.02
    return t2


# data, tag = load_data(1, 4)

