"""
调用训练模型函数，评估模型函数，保存模型
"""
import time
from collections import Counter

import torch

from models.bilstm_crf import LstmModel
from utils import save_model, flatten_lists
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
    model_name = "lstm_"
    save_model(lstm_model, "./model_saved/" + model_name + ".pkl")
    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    torch.cuda.empty_cache()

    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = lstm_model.test(
        test_lists, test_tag_lists)

    metrics = Metrics(test_tag_lists, pred_tag_lists)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists
