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
from .config import TrainingConfig, LSTMConfig, device
from .bilstm import LSTM


class LstmModel(object):
    def __init__(self):
        """对LSTM的模型进行训练与测试
        """

        self.device = device

        # 加载模型参数
        self.input_size = LSTMConfig.input_size
        self.hidden_size = LSTMConfig.hidden_size
        self.num_layers = LSTMConfig.num_layers
        self.out_size = LSTMConfig.out_size
        self.model = LSTM(self.input_size, self.hidden_size, self.num_layers, self.out_size).to(self.device)
        self.cal_loss_func = nn.MSELoss(reduction='mean')

        # 加载训练参数：
        self.epochs = TrainingConfig.epochs
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists):
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
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(
                dev_word_lists, dev_tag_lists)
            print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

    def train_step(self, batch_sents, batch_tags):
        self.model.train()
        self.step += 1
        # 准备数据
        # 将向量张量化

        # batch_sents = np.array(batch_sents)
        # batch_sents = torch.from_numpy(batch_sents)

        batch_sents = batch_sents.to(self.device)

        targets = batch_tags.to(self.device)
        # 从BiLSTM层获得发射得分
        # 前向传播
        scores = self.model(batch_sents, LSTMConfig.time_step)
        # 计算损失 更新参数
        # step1 清空梯度
        self.optimizer.zero_grad()
        # step2 计算loss
        loss = self.cal_loss_func(scores, targets).to(self.device)
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
                scores = self.model(batch_sents, LSTMConfig.time_step)

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
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.test(
                tensorized_sents, lengths, tag2id)

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.crf:
                for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists

    def get_pre(self, word_lists, word2id, tag2id, bert=False):
        """web程序调用模型的接口"""
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        if bert:
            tokenizer = BertTokenizer.from_pretrained('pretrained_bert_models')
            list_batch_sents = list(word_lists)
            batch_sentence = []
            for sentence in list_batch_sents:
                temp = ''
                for word in sentence:
                    temp = temp + word
                batch_sentence.append(temp)

            batch = tokenizer(batch_sentence, padding=True, return_tensors="pt").to(self.device)
            tensorized_sents = self.bert_model(input_ids=batch['input_ids'])[0].to(self.device)
        else:
            tensorized_sents, lengths = tensorized(word_lists, word2id)
            tensorized_sents = tensorized_sents.to(self.device)
        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.test(
                tensorized_sents, lengths, tag2id, bert=bert)
            # 将id转化为标注
            pred_tag_lists = []
            id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
            for i, ids in enumerate(batch_tagids):
                tag_list = []
                if self.crf:
                    for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                        tag_list.append(id2tag[ids[j].item()])
                else:
                    for j in range(lengths[i]):
                        tag_list.append(id2tag[ids[j].item()])
                pred_tag_lists.append(tag_list)

        return pred_tag_lists