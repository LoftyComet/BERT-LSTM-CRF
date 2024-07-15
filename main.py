"""
训练模型
"""
from data import load_data, divide_data
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import bilstm_train_and_eval
from sklearn.model_selection import KFold, train_test_split


def main():
    """训练模型，评估结果"""

    # 读取数据
    print("读取数据...")
    # 左闭右闭
    data, tag = load_data(33, 36)
    (train_lists, train_tag_lists), (dev_lists, dev_tag_lists), (test_lists, test_tag_lists) = divide_data(data, tag)
    train_data, test_data = train_test_split(data, train_size=0.8, random_state=111)
    train_tag, test_tag = train_test_split(tag, train_size=0.8, random_state=111)
    lstm_pred = bilstm_train_and_eval(
                (train_lists, train_tag_lists),
                (dev_lists, dev_tag_lists),
                (test_data, test_tag)
            )

    # # 创建KFold对象
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # # (train_lists, train_tag_lists), (dev_lists, dev_tag_lists), (test_lists, test_tag_lists) = divide_data(data, tag)
    # # 训练评估LSTM模型
    # print("正在训练评估LSTM模型...")
    #
    # for train_index, val_index in kf.split(train_data):
    #     # 获取训练集和验证集
    #     train_lists, dev_lists = train_data[train_index], train_data[val_index]
    #     train_tag_lists, dev_tag_lists = train_tag[train_index], train_tag[val_index]
    #
    #     lstm_pred = bilstm_train_and_eval(
    #         (train_lists, train_tag_lists),
    #         (dev_lists, dev_tag_lists),
    #         (test_data, test_tag)
    #     )







if __name__ == "__main__":
    main()

