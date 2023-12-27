"""
基础lstm
"""
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformers.models.bert.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        """初始化参数

        :param input_size:输入向量的维数
        :param hidden_size：隐向量的维数
        :param out_size:输出向量的维数
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 数据投影层，将BiLSTM输出的hidden_size维度的向量映射为输出标签的个数的维度
        self.lin = nn.Linear(hidden_size, out_size)
        # !!!备用
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_dim, output_dim)
        # )

    def forward(self, scents_tensor, lengths):
        emb = scents_tensor
        # packed = pack_padded_sequence(emb, lengths, batch_first=True)  # [Batch_size, Length, out_size]
        # rnn_out, _ = self.lstm(packed)
        # # rnn_out:[B, L, hidden_size*2]
        # rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        rnn_out, _ = self.lstm(emb)
        # 转换为标注种类的维度
        scores = self.lin(rnn_out)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
