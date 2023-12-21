"""
处理数据
"""
from os.path import join
from codecs import open

import numpy as np
import pandas as pd
from models.config import LSTMConfig
from sklearn.model_selection import train_test_split


def load_data(start, end, data_dir="AI_magic_data"):
    """加载数据

    :param start: 开始人员编号
    :param end: 结束人员编号
    :param data_dir: 数据集文件路径
    :return: 数值对列表
    """
    data_all = []
    for dataIndex in range(start, end + 1):
        prefix = data_dir + "\\" + str(dataIndex) + "\\"
        df = pd.read_csv(prefix + "Eye.csv", encoding="utf-8")
        df2 = pd.read_csv(prefix + "Hand.csv", encoding="utf-8")
        data_per_person = df.groupby('Case')
        for i in range(1, 50 + 1):
            try:
                # 每一个人每case数据
                data_per_case = np.array(data_per_person.get_group(i).reset_index())

                if len(data_per_case) >= LSTMConfig.time_step:
                    # 只需要确定时间跨度的数据
                    data_per_case = data_per_case[-(1+LSTMConfig.time_step):-1]
                    data_all.append(data_per_case)
            except KeyError:
                print("case", i, "被完全去除了")
    return data_all


def divide_data(data):
    train_data, test_data = train_test_split(data, train_size=0.8)
    train_data, dev_data = train_test_split(data)
    return


load_data(1, 4)
