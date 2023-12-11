"""
训练模型
"""
from data import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import crf_train_eval, \
    bilstm_train_and_eval


def main(bilstm_crf=False, bert=False, data_index=1):
    """训练模型，评估结果"""
    assert data_index in [1, 2, 3]
    indexes = ['1', '2', '3']
    index = indexes[data_index - 1]
    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train"+index)
    dev_word_lists, dev_tag_lists = build_corpus("dev"+index, make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test"+index, make_vocab=False)

    # # 训练评估CRF模型
    # print("正在训练评估CRF模型...")
    # crf_pred = crf_train_eval(
    #     (train_word_lists, train_tag_lists),
    #     (test_word_lists, test_tag_lists)
    # )
    #
    # # 训练评估BI-LSTM模型
    # print("正在训练评估双向LSTM模型...")
    # # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    # bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    # lstm_pred = bilstm_train_and_eval(
    #     (train_word_lists, train_tag_lists),
    #     (dev_word_lists, dev_tag_lists),
    #     (test_word_lists, test_tag_lists),
    #     bilstm_word2id, bilstm_tag2id,
    #     crf=False,
    #     data_index=data_index
    # )

    if bilstm_crf:
        print("正在训练评估Bi-LSTM+CRF模型...")
        # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        # 还需要额外的一些数据处理
        train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
            train_word_lists, train_tag_lists
        )
        dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
            dev_word_lists, dev_tag_lists
        )
        test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
            test_word_lists, test_tag_lists, test=True
        )
        lstmcrf_pred = bilstm_train_and_eval(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            crf_word2id, crf_tag2id,
            data_index=data_index
        )

    elif bert:
        print("正在训练评估Bert+Bi-LSTM+CRF模型...")
        # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        # 还需要额外的一些数据处理 (每句开头加上"[CLS]")
        train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
            train_word_lists, train_tag_lists
        )
        dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
            dev_word_lists, dev_tag_lists
        )
        test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
            test_word_lists, test_tag_lists, test=True
        )
        bert_lstmcrf_pred = bilstm_train_and_eval(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            crf_word2id, crf_tag2id,
            bert=True,
            data_index=data_index
        )



if __name__ == "__main__":
    main(bilstm_crf=False, bert=True, data_index=1)
    # main(bilstm_crf=True, bert=False, data_index=2)
