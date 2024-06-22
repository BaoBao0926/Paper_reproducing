"""
according to this repositroy to see the course code
"""

from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import *


class Mamba(nn.Module):

    def __init__(self,
                 d_model: int=2048,  # 定义模型的隐藏层维度
                 n_layer: int=10,  # 定义模型的层数
                 vocab_size: int=10000,  # 定义词汇表的大小
                 d_state: int = 16,  # 定义状态空间的维度，默认为16，这个是有多少个AB，也就是[B L D N]里面的D，
                 expand: int = 2,  # 定义扩展因子，默认为2,在论文3.4提到了这个,隐状态的维度是词嵌入维度的多少倍
                 dt_rank: Union[int, str] = 'auto',  # 定义输入依赖步长 delta 的秩 ‘auto’代表自动设置
                 d_conv: int = 4,  # 定义卷积核的维度，默认为4
                 pad_vocab_size_multiple: int = 8,  # 定义词汇表大小的最小公倍数，默认为8
                 conv_bias: bool = True,  # 定义卷积层是否使用偏执，默认为true
                 bias: bool = False,  # 定义其他层是否使用偏置，默认为false
                 ):
        super(Mamba, self).__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.expand = expand
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.conv_bias = conv_bias
        self.bias = bias

        self.d_inner = int(self.d_model * self.expand)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)  # 把文字转成embedding，大小为d_model
        self.norm = RMSNorm(self.d_model)
        self.layers = nn.ModuleList([
            ResidualBlock(d_model=self.d_model,  d_inner=self.d_inner, d_state=self.d_state,  dt_rank=self.dt_rank,
                          d_conv=self.d_conv, conv_bias=self.conv_bias, bias=self.bias)
            for _ in range(self.n_layer)
        ])
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=self.bias)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        """
        input_ids： 代表着传入的文字应该是的文件, shape为 [batch size, length],
        return logits: shape [batch size, legth, vocab_size]，对着vocab_size做一个softmax就可以得到那一词要被弄出来 [B, L , 1]
        """
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


class ResidualBlock(nn.Module):

    def __init__(self,
                 d_model: int,  # 定义模型的隐藏层维度
                 d_inner: int,     #
                 d_state: int = 16,  # 定义状态空间的维度，默认为16，这个是有多少个AB，也就是[B L D N]里面的D，
                 dt_rank: Union[int, str] = 'auto',  # 定义输入依赖步长 delta 的秩 ‘auto’代表自动设置
                 d_conv: int = 4,  # 定义卷积核的维度，默认为4
                 conv_bias: bool = True,  # 定义卷积层是否使用偏执，默认为true
                 bias: bool = False,  # 定义其他层是否使用偏置，默认为false):
                 ):
        super(ResidualBlock, self).__init__()
        self.mixer = MambaBlock(d_model, d_inner=d_inner, d_state=d_state, dt_rank=dt_rank, d_conv=d_conv,
                                 conv_bias=conv_bias, bias=bias)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class MambaBlock(nn.Module):

    def __init__(self,
                 d_model: int,  # 定义模型的隐藏层维度
                 d_inner: int,  #
                 d_state: int = 16,  # 定义状态空间的维度，默认为16，这个是有多少个AB，也就是[B L D N]里面的D，
                 dt_rank: Union[int, str] = 'auto',  # 定义输入依赖步长 delta 的秩 ‘auto’代表自动设置
                 d_conv: int = 4,  # 定义卷积核的维度，默认为4
                 conv_bias: bool = True,  # 定义卷积层是否使用偏执，默认为true
                 bias: bool = False,
                 ):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        # 进来的映射层，我拆开成两个，看起来容易懂一点
        self.in_projection_left = nn.Linear(self.d_model, self.d_inner, bias=self.bias)
        self.in_projection_right = nn.Linear(self.d_model, self.d_inner, bias=self.bias)

        # 左边的conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv-1
        )

        # 映射 delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank+self.d_state*2, bias=self.bias) # 这里就按照self.bias去设定
        # 从dt_rank 映射到 d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=self.bias)

        # A
        A = repeat(torch.arange(1, self.d_state+1), 'n -> d n', d=self.d_inner)     # [d_inner, d_state] 这里是反的
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 把inner里面的映射回来
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

    def forward(self, x):
        (b, l, d) = x.shape

        res = self.in_projection_right(x)
        x = self.in_projection_left(x)
        # 卷积
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        # 卷积后的非线性激活
        x = F.silu(x)
        y = self.ssm(x)

        y = y * F.silu(res)
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape        # n=d_inner

        A = - torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)  # [B, L, dt_rank+2*n]
        # delta [B, L, dt_rank]
        # B, C  [B, L, d_inner]
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)

        delta = F.softplus(self.dt_proj(delta))     # [B, L, d_inner]
        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        # 这个eisum就是相当于torch.mutul，但是顺便改变维度大小
        # 根据section2里面的公式4可以得到Abar= exp(delta*A),
        # 而现在A只是对于一个batch里面的一个patch的[d_inner, d_state]，现在扩充到 [B L d_inner d_state]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        # 有点不太清楚这里是为什么，根据原文 Bbar = (delta*A)^(-1) * (exp(delta A)-I) * delta * B
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)

        # 下面的应该就是进行这块的计算 selective scan的计算，可以快速的并行计算
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


if __name__ == "__main__":
    # 如过vocab_size是100，输出结果回事[,, 104]，这是因为pad_vocab_size_multiple导致的，是正常的
    a = Mamba(vocab_size=200, n_layer=1)
    print(a)
    sentence = torch.ones([10, 200], dtype=torch.int)   # 10个batch，每一个batch有100个字
    output = a(sentence)
    print(output.size())
    print("finish")