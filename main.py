"""
训练模型
"""
import random

import numpy as np

from data import load_data, divide_data, load_data2

from evaluate import bilstm_train_and_eval

from sklearn.model_selection import RepeatedKFold, train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils import draw_points


def main():
    """训练模型，评估结果"""

    model_name = "lstm"
    # dev_lists, dev_tag_lists = load_data(21, 24)
    # for i in range(5, 6):
    #     # 读取数据
    #     print("读取数据...")
    #     # 左闭右闭
    #     data, tag = load_data(1, i)
    #     (train_lists, train_tag_lists), (dev_lists, dev_tag_lists), (test_lists, test_tag_lists) = divide_data(data, tag)
    #     dev_lists, dev_tag_lists = load_data(21, 24)
    #     # train_data, test_data = train_test_split(data, train_size=0.8, random_state=111)
    #     # train_tag, test_tag = train_test_split(tag, train_size=0.8, random_state=111)
    #     bilstm_train_and_eval(
    #                 (train_lists, train_tag_lists),
    #                 (dev_lists, dev_tag_lists),
    #                 (test_lists, test_tag_lists),
    #                 model_name + str(i)
    #             )

    k = 5  # 设置折叠数，通常为 5 或 10
    kf = KFold(n_splits=k, shuffle=True, random_state=66)  # 可以根据需要调整 shuffle 和 random_state
    X = range(1, 21)
    # dev_lists, dev_tag_lists = load_data(21, 24, for_train=False)
    # 读取数据
    print("读取数据...")
    q1 = 0
    for train_index, test_index in kf.split(X):
        print(train_index, test_index)
        print("读取数据...")
        for qq in range(len(train_index)):
            train_index[qq] += 1
        for qq in range(len(test_index)):
            test_index[qq] += 1
        train_lists, train_tag_lists = load_data2(train_index, for_train=True)
        dev_lists, dev_tag_lists = load_data2(test_index, for_train=False)
        # 左闭右闭
        # draw_points(train_tag_lists)
        # (train_lists, train_tag_lists), (dev_lists, dev_tag_lists), (test_lists, test_tag_lists) = divide_data(data, tag)
        train_lists, test_lists = train_test_split(train_lists, train_size=0.99, random_state=111)
        train_tag_lists, test_tag_lists = train_test_split(train_tag_lists, train_size=0.99, random_state=111)
        bilstm_train_and_eval(
                    (train_lists, train_tag_lists),
                    (dev_lists, dev_tag_lists),
                    (test_lists, test_tag_lists),
                    q1
                )
        q1 += 1






if __name__ == "__main__":
    main()

