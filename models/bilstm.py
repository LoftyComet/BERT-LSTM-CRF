"""
基础lstm
"""
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformers.models.bert.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
from models.config import LSTMConfig
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self,  input_size, hidden_size, num_layers, time_step, completion_percentage, out_size, dropout_prob=0.3):
        """初始化参数

        :param input_size:输入向量的维数
        :param hidden_size：隐向量的维数
        :param out_size:输出向量的维数
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * time_step, out_size)

        # self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        # self.lstm1 = nn.LSTM(input_size=32, hidden_size=hidden_size,
        #                      num_layers=2, bidirectional=True, batch_first=True)
        #
        # # 修改attention的embed_dim以匹配LSTM的输出
        # self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4)
        #
        # # 调整全连接层的维度
        # self.fc1 = nn.Linear(hidden_size * 2 * 20, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, 3)  # 假设我们只预测一个值
        # self.dropout = nn.Dropout(0.2)


        # transformer
        # self.transformer = nn.Transformer(
        #     d_model=64,
        #     nhead=4,
        #     num_encoder_layers=3,
        #     dim_feedforward=128,
        #     dropout=0.1
        # )
        # self.linear = nn.Linear(input_size, 64)
        # self.regressor = nn.Linear(64, out_size)

        # self.embedding = nn.Linear(input_size, 64)
        #
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=64,
        #     nhead=4,
        #     dim_feedforward=64 * 4,
        #     batch_first=True
        # )
        # self.transformer_encoder = nn.TransformerEncoder(
        #     encoder_layer,
        #     num_layers=num_layers
        # )
        #
        # self.fc = nn.Linear(64, 3)


        # # MLP
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        # # x shape: [batch_size, sequence_length, features]
        # batch_size, seq_len, features = x.size()
        #
        # # 转换维度用于Conv1d
        # x = x.permute(0, 2, 1)  # [batch_size, features, sequence_length]
        # x = self.conv1d(x)  # [batch_size, 32, sequence_length]
        # x = F.relu(x)
        #
        # # 准备LSTM输入
        # x = x.permute(0, 2, 1)  # [batch_size, sequence_length, 32]
        #
        # # BiLSTM处理
        # lstm_out, _ = self.lstm1(x)  # [batch_size, sequence_length, hidden_size*2]
        #
        # # 注意力机制
        # # 将形状转换为注意力机制需要的格式：[sequence_length, batch_size, embed_dim]
        # att_input = lstm_out.permute(1, 0, 2)
        # att_output, _ = self.attention(att_input, att_input, att_input)
        #
        # # 转换回原来的形状
        # x = att_output.permute(1, 0, 2)  # [batch_size, sequence_length, hidden_size*2]
        #
        # # 全连接层处理
        # x = self.dropout(x)
        # x = x.contiguous().view([x.size()[0], -1])
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)  # [batch_size, sequence_length, 1]
        # return x

        x = x.to(LSTMConfig.device)
        # x: [batch, time_step, input_size]
        out, (h_n, c_n) = self.lstm(x)
        # Apply dropout to the output of the LSTM layer
        # out = self.dropout(out)

        # Take the output from the last time step
        # out = out[:, -1, :]

        # Take the output from the all the time step
        out = out.contiguous().view([out.size()[0], -1])
        # Pass the output through the fully connected layer
        out = self.fc(out)
        return out

        # transformer
        # x = x.to(LSTMConfig.device)
        # src = self.linear(x)
        # src = src.permute(1, 0, 2)  # Transformer expects [sequence length, batch size, features]
        # transformer_output = self.transformer(src, src)
        # transformer_output = transformer_output.mean(dim=0)  # Average over the sequence length
        # output = self.regressor(transformer_output)
        # return output

        # transformer_encoder
        # x shape: (batch_size, seq_len, input_dim)
        # x = self.embedding(x)
        # x = self.transformer_encoder(x)
        # # 使用序列的最后一个时间步进行预测
        # x = x[:, -1, :]
        # return self.fc(x)

        # MLP
        # x = x.contiguous().view([x.size()[0], -1])
        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.fc3(out)
        # return out

    def test(self, scents_tensor):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        scents_tensor = scents_tensor.to(LSTMConfig.device)
        logit = self.forward(scents_tensor)  # [B, L, out_size]
        # _, batch_tagids = torch.max(logits, dim=2)

        return logit
