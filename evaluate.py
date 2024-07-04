"""
调用训练模型函数，评估模型函数，保存模型
"""
import math
import time
from collections import Counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from models.config import LSTMConfig
from models.bilstm_crf import LstmModel
from utils import save_model, draw_point, draw_error
from evaluating import Metrics


def bilstm_train_and_eval2(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False, bert=False, data_index=1):
    """训练评估保存CRF模型

    :param crf: 是否添加cfr
    :param remove_O: 评估时是否去除O标签
    :param bert: 是否添加bert
    :param data_index: 处理的数据集编号
    """
    assert data_index in [1, 2, 3]
    indexes = ['1', '2', '3']
    index = indexes[data_index - 1]

    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data
    test_word_lists = test_word_lists[::]
    test_tag_lists = test_tag_lists[::]
    if bert:
        test_word_lists = test_word_lists[::20]
        test_tag_lists = test_tag_lists[::20]
    elif index == '3':
        test_word_lists = test_word_lists[::120]
        test_tag_lists = test_tag_lists[::120]
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = LstmModel(vocab_size, out_size, crf=crf, bert=bert)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id, bert=bert)

    model_name = "bilstm_crf" if crf else "bilstm"
    model_name = "bert_" + model_name if bert else model_name

    model_name = model_name + index
    save_model(bilstm_model, "./ckpts/" + model_name + ".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    torch.cuda.empty_cache()
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id, bert=bert)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def bilstm_train_and_eval(train_data, dev_data, test_data):
    """训练评估保存CRF模型

        :param train_data: 是否添加cfr
        :param dev_data: 评估时是否去除O标签
        :param test_data: 是否添加bert
        """

    start = time.time()

    lstm_model = LstmModel()
    train_lists = train_data[0]
    train_tag_lists = train_data[1]
    dev_lists = dev_data[0]
    dev_tag_lists = dev_data[1]
    test_lists = test_data[0]
    test_tag_lists = test_data[1]
    lstm_model.train(train_lists, train_tag_lists,
                     dev_lists, dev_tag_lists)
    model_name = "lstm_2" + str(int(LSTMConfig.completion_percentage * 100))
    save_model(lstm_model.model, "./model_saved/" + model_name + ".pkl")
    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    torch.cuda.empty_cache()

    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = lstm_model.test(
        test_lists, test_tag_lists)
    test_tag_lists = test_tag_lists.cpu().numpy()
    pred_tag_lists = pred_tag_lists.cpu().numpy()

    trigger_dis = []
    tra_dis = []
    # 记录开始时间
    start_time = time.time()
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
    print("代码段运行时间：", execution_time, "秒")
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
            (test_lists[i][-1][24] - pred_tag_lists[i][0]) ** 2 + (
                        test_lists[i][-1][25] - pred_tag_lists[i][1]) ** 2 + (
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

    # 画预测点和目标球
    # for i in range(len(train_lists)):
    #     draw_finger(train_lists[i][:, 24], train_lists[i][:, 25], train_lists[i][:, 26], train_tag_lists[i])

    # draw_point(pred_tag_lists[:10], test_tag_lists[:10])

    # 预测点和目标球单独放一起
    # for i in range(len(pred_tag_lists)):
    #     draw_point([pred_tag_lists[i]], [test_tag_lists[i]])
    # draw_point(pred_tag_lists, test_tag_lists)

    # draw_error(pred_tag_lists, test_tag_lists, ans)

    # 结果可视化
    # fig, ax = plt.subplots(4, 1)
    # fig.set_size_inches(10, 4)
    #
    # ax[0].plot(range(len(test_tag_lists_x))[:40], test_tag_lists_x[:40], linewidth=1.5, linestyle='-', label='True')
    # ax[0].plot(range(len(pred_tag_lists_x))[:40], pred_tag_lists_x[:40], linewidth=1, linestyle='-.', label='Predicted')
    # ax[0].set_title("x")
    #
    # ax[1].plot(range(len(test_tag_lists_y))[:40], test_tag_lists_y[:40], linewidth=1.5, linestyle='-', label='True')
    # ax[1].plot(range(len(pred_tag_lists_y))[:40], pred_tag_lists_y[:40], linewidth=1, linestyle='-.', label='Predicted')
    # ax[1].set_title("y")
    #
    # ax[2].plot(range(len(test_tag_lists_z))[:40], test_tag_lists_z[:40], linewidth=1.5, linestyle='-', label='True')
    # ax[2].plot(range(len(pred_tag_lists_z))[:40], pred_tag_lists_z[:40], linewidth=1, linestyle='-.', label='Predicted')
    # ax[2].set_title("z")
    #
    # ax[3].plot(range(len(ans))[:40], zero[:40], linewidth=1.5, linestyle='-', label='True')
    # ax[3].plot(range(len(ans))[:40], ans[:40], linewidth=1, linestyle='-', label='Predicted')
    # ax[3].set_title("distance")
    # plt.legend()
    # plt.show()

    # ax[0].plot(range(len(test_tag_lists_x)), test_tag_lists_x, linewidth=1.5, linestyle='-', label='True')
    # ax[0].plot(range(len(pred_tag_lists_x)), pred_tag_lists_x, linewidth=1, linestyle='-.', label='Predicted')
    # ax[0].set_title("x")
    #
    # ax[1].plot(range(len(test_tag_lists_y)), test_tag_lists_y, linewidth=1.5, linestyle='-', label='True')
    # ax[1].plot(range(len(pred_tag_lists_y)), pred_tag_lists_y, linewidth=1, linestyle='-.', label='Predicted')
    # ax[1].set_title("y")
    #
    # ax[2].plot(range(len(test_tag_lists_z)), test_tag_lists_z, linewidth=1.5, linestyle='-', label='True')
    # ax[2].plot(range(len(pred_tag_lists_z)), pred_tag_lists_z, linewidth=1, linestyle='-.', label='Predicted')
    # ax[2].set_title("z")
    #
    # ax[3].plot(range(len(ans)), zero, linewidth=1.5, linestyle='-', label='True')
    # ax[3].plot(range(len(ans)), ans, linewidth=1, linestyle='-', label='Predicted')
    # ax[3].set_title("distance")
    # plt.legend()
    # plt.show()

    # for ans1 in ans:
    #     print(ans1)

    # metrics = Metrics(test_tag_lists, pred_tag_lists)
    # metrics.report_scores()
    # metrics.report_confusion_matrix()

    return pred_tag_lists
