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

    pred_tag_lists, test_tag_lists = lstm_model.test(
        test_lists, test_tag_lists)
    test_tag_lists = test_tag_lists.numpy()
    pred_tag_lists = pred_tag_lists.numpy()
    ans = []
    for i in range(len(pred_tag_lists)):
        # for j in range(len(pred_tag_lists[0])):
        print("i", i, test_tag_lists[i][0], test_tag_lists[i][1], test_tag_lists[i][2])
        print("i", i, pred_tag_lists[i][0], pred_tag_lists[i][1], pred_tag_lists[i][2])
        temp = math.sqrt(
            (test_tag_lists[i][0] - pred_tag_lists[i][0]) ** 2 + (test_tag_lists[i][1] - pred_tag_lists[i][1]) ** 2 + (
                        test_tag_lists[i][2] - pred_tag_lists[i][2]) ** 2)
        print(temp)
        ans.append(temp)

    draw_point(pred_tag_lists, test_tag_lists)
    print("测试集中预测位置与真实位置的平均距离为", np.mean(ans))


def get_pre(s, bert=False, data_index=1):
    """加载模型，输入时序数据，返回预测结果"""
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

