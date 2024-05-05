import numpy as np  # 导入NumPy库，用于进行矩阵运算和数据处理
import torch  # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn  # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time  # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘制图表和可视化
import seaborn  # 导入Seaborn库，用于绘制统计图形和美化图表
from module import *

class EncoderDecoder(nn.Module):
    """
    transform architecture encoder-decoder architecture
    """
    def __init__(self, encoder, decoder, srt_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = srt_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 在forward函数中，有四个参数，source代表源数据，
        # target代表目标数据,source_mask和target_mask代表对应的掩码张量,
        # 在函数中，将source source_mask传入编码函数，
        # 得到结果后与source_mask target 和target_mask一同传给解码函数
        memory = self.encode(src, src_mask)
        res = self.decode(memory, src_mask, tgt, tgt_mask)
        return res

    def encode(self, src, src_mask):
        # 编码函数，以source和source_mask为参数,使用src_embed对source做处理，
        # 然后和source_mask一起传给self.encoder
        src_embedds = self.src_embed(src)
        return self.encoder(src_embedds, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # 解码函数，以memory即编码器的输出，source_mask target target_mask为参数,
        # 使用tgt_embed对target做处理，然后和source_mask,target_mask,memory一起传给self.decoder
        target_embedds = self.tgt_embed(tgt)
        return self.decoder(target_embedds, memory, src_mask, tgt_mask)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    用于构建模型
    :param src_vocab: length of source
    :param tgt_vocab: length of target
    :param N:   编码器和解码器堆叠的数量
    :param d_model: Word embedding的大小，默认为512
    :param d_ff: feedforward layer中embedding的大小，默认为2048
    :param h: 多头的个数，必须可以被d_model整除
    :param dropout:dropout的个数
    :return: 返回模型
    """

    c = copy.deepcopy

    atten = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(atten), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(atten), c(atten), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model







