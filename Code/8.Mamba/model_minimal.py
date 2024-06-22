"""
https://github.com/johnma2006/mamba-minimal/tree/master source code comes from here
"""


"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
# from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import *



@dataclass
class ModelArgs:
    d_model: int = 1000                       # 定义模型的隐藏层维度
    n_layer: int = 1                      # 定义模型的层数
    vocab_size: int = 100                   # 定义词汇表的大小
    d_state: int = 16                   # 定义状态空间的维度，默认为16，这个是有多少个AB，也就是[B L D N]里面的D，
    expand: int = 2                     # 定义扩展因子，默认为2,在论文3.4提到了这个,隐状态的维度是词嵌入维度的多少倍
    # dt_rank: Union[int, str] = 'auto'   # 定义输入依赖步长 delta 的秩 ‘auto’代表自动设置
    dt_rank = 1000
    d_conv: int = 4                     # 定义卷积核的维度，默认为4
    pad_vocab_size_multiple: int = 8    # 定义词汇表大小的最小公倍数，默认为8
    conv_bias: bool = True              # 定义卷积层是否使用偏执，默认为true
    bias: bool = False                  # 定义其他层是否使用偏置，默认为false
    d_inner = 2000
    
    def __post_init__(self):
        # 在__init__之后会自动调用这个post_init，然后根据init里面的设置的数，把这些东西都给计算设置

        # 使用expand（2）来计算内部维度，也就是扩展之后的维度。在mamba文章3.4中描述了一个expansion factor
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            # 如果使用auto来计算rank的大小，这使用下面的方式，通过模型隐藏层维度的大小除以16然后ceil一下
            self.dt_rank = math.ceil(self.d_model / 16)

        # 这里是要确保vocab_size是pad_vocab_size_multiple的倍数，如果不是，计算调整为最近的倍数
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        # 词嵌入层，其实可以直接看作MLP层，也是可以训练的- 第一个参数是词表大小，第二个参数是词嵌入维度大小
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        # layers现在是一个module list，里面装了n_layer个 residualBlock
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)     # 进行的norm是RMSNorm，这个RMSnorm是自己实现的-在底下实现的，很好理解 norm里面也有科学系参数

        # 这个线性层是把隐藏状态的维度映射会词表的维度大小
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        # 将线性层的输出权重与嵌入层的权重绑定，这是权重共享的一种形式，有助于减少参数数量并可能提高模型的泛化能力
        # 感觉可能是 weight tying paper里面研究的
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)       # 所有的x都进行word embedding
        
        for layer in self.layers:           # 然后把所有的layer全部都过一遍
            x = layer(x)
            
        x = self.norm_f(x)                  # 过一个RMSNorm，进行归一化
        logits = self.lm_head(x)            # 这个是最后经过一个MLP层，把隐状态映射到词表的大小，然后就可以做分类 就知道下一个词是多少

        return logits

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model


class ResidualBlock(nn.Module):
    """
    这里主要是为了参加而单独进行的
    """

    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        # 这个是minimal版本的 [norm -> mamba -> add]
        output = self.mixer(self.norm(x)) + x
        # 而官方版本的是 [add -> norm -> mamba]--有点不太理解怎么先加，我觉得是因为构架不一样，其实本质是一样的

        return output
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args
        # 内部使用的都是d_model(应该为词嵌入的两倍)，但是这里映射成了 d_model的四倍（d_inner已经是两倍了），底下有一步映射之后的拆成两部分分别使用
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        # 进行卷积，对着序列数据进行卷积，可以在这一步把前后的内容相关联起来 [B, 数据通道数，数据的length]->[B, 数据输出的通道数， 处理后的length]
        # 这里处理之后是 [B,C,L] == [B, d_inner, L] -> [B, d_inner, L]， 只是进行了关联，chanel和length都没有发生改变
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,       # 输入图像的通道数
            out_channels=args.d_inner,      # 聚氨基产生的通道数，多少个卷积核
            bias=args.conv_bias,            # 是否要bias
            kernel_size=args.d_conv,        # 卷积核的大小，一维应该是左右的大小，左右横跨多少个
            groups=args.d_inner,            # 从输入通道到输出通道的阻塞连接数，要是什么倍数关系，反正好像用不到
            padding=args.d_conv - 1,        # 做多少个padding，因为stride是1，所以只需要padding步长-1个就可以了
        )

        # 在原文3.4中architecture中看，经过cov1d，经过一个activation，之后就是经过一个SSM模块进行计算，
        # 下面的都是ssm模块要用到的，ssm算法在3.3 algorithm里面描述

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        # 我觉得这里是 Δ, B, C都是根据输入x进行映射的，直接用一个linear层一起映射，然后到时候划分开来就是了
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        # torch.arange是 生成了一个维度大小Tensor[d_state]那么大的tensor，然后里面的东西是从1一直到d_state
        # 然后使用repeat，重复的d_inner次，也就是  [d_inner, d_state]，
        # 这里可以理清楚，d_inner为维度大小，有多少个A矩阵叠在一起，d_state为隐状态维度大小， 但这里好像反了，可能是为了后面的操作
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        # 我没有找到为什么要log，我有一个猜测，因为这里反正都是要训练得到的，所以初始化一个这个也无所谓，相当于只是进行初始化了
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner)) # D没有被忽略，按照可训练参数，维度大小为[d_inner]设立的
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape

        # 这一步是section3.4架构中的，前面最下面两个映射
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        # 现在再用split，分成了两块，每一个的shape都是 [B,L, d_in]
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        # 左边通路的做卷积
        # [B L d_in]->[B, d_in L],这步是因为，我们希望进行卷积是对着不同patch进行的，所以就是把d_in当作channel，有这么L个，然后进行下一步的卷积
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]                    # 我猜可能是由于padding，尾巴会多出一点东西来
        x = rearrange(x, 'b d_in l -> b l d_in')        # 然后给他转回来 [B L d_inner]

        # 这一步是卷积之后的非线性激活
        x = F.silu(x)
        # 然后左边通路的ssm模块， 这里的x是 [B, L, d_inner]
        y = self.ssm(x)
        # 右边通路的非线性激活 * ssm模块出来的
        y = y * F.silu(res)
        # 最后一个映射出来就是结果， 内部计算都是d_inner,输出的时候是d_model(=词嵌入大小)
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        # 在__init__里面创建的A_log，是由A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        (d_in, n) = self.A_log.shape    # [d_inner, d_state]

        # Compute ∆ A B C D, the state space parameters.
        # A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        # ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        # and is why Mamba is called **selective** state spaces)

        # 这里的A的负号-是因为在状态空间模型中，矩阵A通常表示的是一个离散时间系统的转换矩阵，它描述了系统状态随时间的演变。
        # 在许多情况下，A矩阵的元素应该是负的，以确保系统的稳定性。这是因为在离散时间系统中，
        # 我们希望系统的状态随着时间的推移而衰减或稳定下来，而不是增长，从而避免系统变得不稳定或发散
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # [B, L, d_inner]->[b, l, dt_rank + 2*n]
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        # 这里才是在计算ssm的东西，使用了selective scan去加速，上面都是在准备ABCD和delta
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u:      shape (b, l, d_inner)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta:  shape (b, l, d_inner)
            A:      shape (d_inner, d_state)
            B:      shape (b, l, d_state)
            C:      shape (b, l, d_state)
            D:      shape (d_inner,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
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
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))     # 这个作为gi，是一个可学习参数


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        
if __name__ == "__main__":
    mam = Mamba(ModelArgs)
    print(mam)
    sentence = torch.ones([10, 100], dtype=torch.int)  # 10个batch，每一个batch有100个字
    output = mam(sentence)
    print(output.size())
    print("finish")