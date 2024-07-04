"""
基础lstm
"""
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformers.models.bert.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
from models.config import LSTMConfig


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

        # MLP
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
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

        # transformer
        # x = x.to(LSTMConfig.device)
        # src = self.linear(x)
        # src = src.permute(1, 0, 2)  # Transformer expects [sequence length, batch size, features]
        # transformer_output = self.transformer(src, src)
        # transformer_output = transformer_output.mean(dim=0)  # Average over the sequence length
        # output = self.regressor(transformer_output)
        # return output

        # MLP
        # x = x.contiguous().view([x.size()[0], -1])
        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.fc3(out)
        return out

    def test(self, scents_tensor):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        scents_tensor = scents_tensor.to(LSTMConfig.device)
        logit = self.forward(scents_tensor)  # [B, L, out_size]
        # _, batch_tagids = torch.max(logits, dim=2)

        return logit
