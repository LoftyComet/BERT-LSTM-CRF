"""
bilstm-crf
"""
from itertools import zip_longest
from copy import deepcopy

import numpy as np
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import *
import torch
import torch.nn as nn
import torch.optim as optim

from .util import tensorized, sort_by_lengths, cal_loss, cal_lstm_crf_loss
from .config import TrainingConfig, LSTMConfig
from .bilstm import LSTM
from utils import save_model


class LstmModel(object):
    def __init__(self):
        """对LSTM的模型进行训练与测试
        """

        self.device = LSTMConfig.device

        # 加载模型参数
        self.input_size = LSTMConfig.input_size
        self.hidden_size = LSTMConfig.hidden_size
        self.num_layers = LSTMConfig.num_layers
        self.out_size = LSTMConfig.out_size
        self.model = LSTM(self.input_size, self.hidden_size, self.num_layers, LSTMConfig.time_step, LSTMConfig.completion_percentage, self.out_size).to(self.device)
        self.cal_loss_func = nn.MSELoss(reduction='mean')
        # self.cal_loss_func = nn.SmoothL1Loss(reduction='mean')

        # 加载训练参数：
        self.epochs = TrainingConfig.epochs
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists, model_name):
        B = self.batch_size
        for e in range(1, self.epochs + 1):
            self.step = 0
            losses = 0.
            # 分批次处理数据
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind + B]
                batch_tags = tag_lists[ind:ind + B]

                losses += self.train_step(batch_sents, batch_tags)
                # 每print_step打印一次
                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // B + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.8f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(
                dev_word_lists, dev_tag_lists)
            print("Epoch {}, Val Loss:{:.8f}".format(e, val_loss))
            save_model(self.model, "./train_model_saved" + str(model_name) + "/lstm" + str(e) + ".pkl")
    def train_step(self, batch_sents, batch_tags):
        self.model.train()
        self.step += 1
        # 准备数据
        # 将向量张量化

        # batch_sents = np.array(batch_sents)
        # batch_sents = torch.from_numpy(batch_sents)

        batch_sents = batch_sents.to(self.device)

        targets = batch_tags.to(self.device)
        # 从LSTM层获得发射得分
        # 前向传播
        scores = self.model(batch_sents)
        # 计算损失 更新参数
        # step1 清空梯度
        self.optimizer.zero_grad()

        # step2 计算loss
        # loss = self.cal_loss_func(scores, targets).to(self.device)
        loss = self.cal_loss_func(scores.squeeze(-1), targets).to(self.device)
        # step3 #反向传播 计算梯度
        loss.backward()
        # 根据优化器的策略去更新参数
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind + self.batch_size]
                batch_tags = dev_tag_lists[ind:ind + self.batch_size]
                batch_sents = batch_sents.to(self.device)
                targets = batch_tags.to(self.device)
                # forward
                # print("传入的lengths参数", lengths)
                scores = self.model(batch_sents)

                # 计算损失
                loss = self.cal_loss_func(scores, targets).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss

            return val_loss

    def test(self, word_lists, tag_lists):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        tensorized_sents = word_lists.to(self.device)
        self.best_model.eval()
        with torch.no_grad():
            batch_tag = self.best_model.test(tensorized_sents)

        pred_tag_lists = batch_tag

        return pred_tag_lists, tag_lists

    def get_pre(self, word_lists):
        """预测的接口"""
        # 准备数据
        tensorized_sents = word_lists.to(self.device)
        self.best_model.eval()
        with torch.no_grad():
            batch_tag = self.best_model.test(tensorized_sents)

        pred_tag_lists = batch_tag

        return pred_tag_lists