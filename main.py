"""
训练模型
"""
from data import load_data, divide_data
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import bilstm_train_and_eval


def main():
    """训练模型，评估结果"""
    # 读取数据
    print("读取数据...")
    data, tag = load_data(1, 4)
    (train_lists, train_tag_lists), (dev_lists, dev_tag_lists), (test_lists, test_tag_lists) = divide_data(data, tag)
    # 训练评估LSTM模型
    print("正在训练评估LSTM模型...")

    lstm_pred = bilstm_train_and_eval(
        (train_lists, train_tag_lists),
        (dev_lists, dev_tag_lists),
        (test_lists, test_tag_lists)
    )


if __name__ == "__main__":
    main()

