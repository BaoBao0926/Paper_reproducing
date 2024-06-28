import numpy as np  # 导入NumPy库，用于进行矩阵运算和数据处理
import torch  # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn  # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time  # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘制图表和可视化
from module_muyi import *

class Encoder(nn.Module):
    def __init__(self, dim_word, h, d_ff, dropout, N):
        super(Encoder, self).__init__()
        self.dim_word = dim_word
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout
        self.N = N

        self.Encoders = clones(
            Encoderlayer(self.dim_word, self.h, self.d_ff, self.dropout),
            self.N
        )

    def forward(self, x):
        for encoder in self.Encoders:
            x = encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, h, dim_word, d_ff, dropout, N, tgt_length):
        super(Decoder, self).__init__()
        self.h = h
        self.dim_word = dim_word
        self.d_ff = d_ff
        self.dropout = dropout
        self.N = N
        self.tgt_length = tgt_length

        self.Decoders = clones(
            DecoderLayer(self.h, self.dim_word, self.d_ff, self.dropout, self.tgt_length),
            self.N
        )

    def forward(self, x, decoder_info):
        for decoder in self.Decoders:
            x = decoder(x, decoder_info)
        return x


class Transform(nn.Module):
    def __init__(self, dim_word, scr_length, tgt_length, dropout, h, N, d_ff, src_vocab_size, tgt_vocab_size):
        super(Transform, self).__init__()
        self.dim_word = dim_word
        self.scr_length = scr_length
        self.tgt_length = tgt_length
        self.dropout = dropout
        self.h = h
        self.N = N
        self.d_ff = d_ff
        self.scr_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # 层
        # enc和dec的word embedding和position embedding是不一样的，所以需要两个
        self.enc_word_embedding = Embedding(self.dim_word, self.scr_vocab_size) # 注意这里是词本的大小，而不是预测句子的大小
        # 这里的长度应该是对一句话的每个单词进行编码，所以应该是scource的长度
        self.enc_position_embedding = PositionalEncoding(self.dim_word, self.dropout)
        self.Encoder = Encoder(self.dim_word, self.h, self.d_ff, self.dropout, self.N)

        self.dec_word_embedding = Embedding(self.dim_word, self.tgt_vocab_size) # 这里是target的词本的大小
        self.dec_position_embedding = PositionalEncoding(self.dim_word, self.dropout)
        self.Decoder = Decoder(self.h, self.dim_word, self.d_ff, self.dropout, self.N, self.tgt_length)
        self.generator = Generator(self.dim_word, self.tgt_length)

    def forward(self, enc_input, dec_input):
        # enc的word_embedding
        word_embedding = self.enc_word_embedding(enc_input)
        enc_input = self.enc_position_embedding(word_embedding)
        # enc的计算
        encoder_info = self.Encoder(enc_input)

        # dec的word embedding
        dec_word_embeeding = self.dec_word_embedding(dec_input)
        dec_input = self.dec_position_embedding(dec_word_embeeding)
        decoder_output = self.Decoder(dec_input, encoder_info)  # dec的input有自己的

        gene_output = self.generator(decoder_output)
        return gene_output

