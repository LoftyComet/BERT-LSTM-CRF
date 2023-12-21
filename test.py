"""
加载并评估模型
"""
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from evaluating import Metrics


CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = './ckpts/bilstm'

BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf'
BERT_BiLSTMCRF_MODEL_PATH = './ckpts/bert_bilstm_crf'
suffix = '.pkl'


REMOVE_O = True  # 在评估的时候是否去除O标记


def main(bilstm_crf=False, bert=False, data_index=1):
    assert data_index in [1, 2, 3]
    indexes = ['1', '2', '3']
    index = indexes[data_index - 1]
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train" + index)
    test_word_lists, test_tag_lists = build_corpus("test" + index, make_vocab=False)

    if index == '2':
        # 加载并评估CRF模型
        print("加载并评估crf模型...")
        crf_model = load_model(CRF_MODEL_PATH)
        crf_pred = crf_model.test(test_word_lists)
        metrics = Metrics(test_tag_lists, crf_pred, remove_O=REMOVE_O)
        metrics.report_scores()
        metrics.report_confusion_matrix()

        # bilstm模型
        print("加载并评估bilstm模型...")
        bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
        bilstm_model = load_model(BiLSTM_MODEL_PATH + index + suffix)
        bilstm_model.model.bilstm.flatten_parameters()  # remove warning
        lstm_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                       bilstm_word2id, bilstm_tag2id)
        metrics = Metrics(target_tag_list, lstm_pred, remove_O=REMOVE_O)
        metrics.report_scores()
        metrics.report_confusion_matrix()

    if bilstm_crf:
        if index == '3':
            test_word_lists = test_word_lists[::120]
            test_tag_lists = test_tag_lists[::120]
        print("加载并评估bilstm+crf模型...")
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        bilstm_model = load_model(BiLSTMCRF_MODEL_PATH + index + suffix)
        bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
        test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
            test_word_lists, test_tag_lists, test=True
        )
        lstmcrf_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                          crf_word2id, crf_tag2id)
        metrics = Metrics(target_tag_list, lstmcrf_pred, remove_O=REMOVE_O)
        metrics.report_scores()
        metrics.report_confusion_matrix()

    if bert:
        assert data_index in [1, 2, 3]
        indexes = ['1', '2', '3']
        index = indexes[data_index - 1]
        print("读取数据...")
        train_word_lists, train_tag_lists, word2id, tag2id = \
            build_corpus("train" + index)
        test_word_lists, test_tag_lists = build_corpus("test" + index, make_vocab=False)
        print("加载并评估bert+bilstm+crf模型...")
        if index == '3':
            test_word_lists = test_word_lists[::120]
            test_tag_lists = test_tag_lists[::120]
        else:
            test_word_lists = test_word_lists[::15]
            test_tag_lists = test_tag_lists[::15]
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        bilstm_model = load_model(BERT_BiLSTMCRF_MODEL_PATH + index + suffix)
        bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
        test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
            test_word_lists, test_tag_lists, test=True
        )
        lstmcrf_pred2, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                          crf_word2id, crf_tag2id, bert=True)

        metrics = Metrics(target_tag_list, lstmcrf_pred2, remove_O=REMOVE_O)
        metrics.report_scores()
        metrics.report_confusion_matrix()


def get_pre(s, bert=False, data_index=1):
    """加载模型，输入待测字符串，返回识别结果"""
    assert data_index in [1, 2, 3]
    indexes = ['1', '2', '3']
    index = indexes[data_index - 1]
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train" + index)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    if bert:
        bilstm_model = load_model(BERT_BiLSTMCRF_MODEL_PATH + index + suffix)
    else:
        bilstm_model = load_model(BiLSTMCRF_MODEL_PATH + index + suffix)
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning

    # while s != "q":
    #     s = input("请输入要测试的句子(输入q结束)")
    #     s = list(s)
    #     s.append('<end>')
    #     s2 = [s]
    #     pre_id = bilstm_model.get_pre(s2, crf_word2id, crf_tag2id, bert=bert)[0]
    #     result = []
    #     for i in range(len(pre_id)):
    #         result.append((s[i], pre_id[i]))
    s = list(s)
    s.append('<end>')
    s2 = [s]
    pre_id = bilstm_model.get_pre(s2, crf_word2id, crf_tag2id, bert=bert)[0]
    result = []
    for i in range(len(pre_id)):
        result.append((s[i], pre_id[i]))

    return result


if __name__ == "__main__":
    # main(bilstm_crf=False, bert=True, data_index=1)
    main(bilstm_crf=True, bert=True, data_index=2)

