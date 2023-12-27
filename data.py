"""
处理数据
"""
from os.path import join
from codecs import open

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
                if len(data_per_case) >= LSTMConfig.time_step:
                    # 只需要确定时间跨度的数据
                    data_per_case = data_per_case[-(1+LSTMConfig.time_step):-1]
                    data_per_case2 = data_per_case2[-(1 + LSTMConfig.time_step):-1]

                    data_all.append(extend_data(data_per_case, data_per_case2))
                    tag_all.append(tag_per_case[0][9:12])
            except KeyError:
                print("case", i, "被完全去除了")

    # 转换为tensor
    data_all_ans = torch.ones((len(data_all), len(data_all[0]), len(data_all[0][0])))
    for i in range(len(data_all)):
        for j in range(len(data_all[0])):
            for k in range(len(data_all[0][0])):
                data_all_ans[i][j][k] = data_all[i][j][k]
    tag_all_ans = np.array(tag_all)[:, 8:11]
    tag_all_ans = torch.ones((len(tag_all_ans), LSTMConfig.time_step, len(tag_all[0])))
    for i in range(len(tag_all)):
        for j in range(LSTMConfig.time_step):
            for k in range(len(tag_all[0])):
                tag_all_ans[i][j][k] = tag_all[i][k]
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
    # 按列取
    t1 = eye[:, 2:17]
    t2 = hand[:, 2:68]
    return np.hstack((t1, t2))


data, tag = load_data(1, 4)
divide_data(data, tag)