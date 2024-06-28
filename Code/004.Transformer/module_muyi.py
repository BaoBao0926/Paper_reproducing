import numpy as np  # 导入NumPy库，用于进行矩阵运算和数据处理
import torch  # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn  # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time  # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘制图表和可视化

def clones(module, N):
    """
    要复制一个module出来，方便集体操作，产生N个module出来，装到一个list里面
    :param module: 要被复制的module
    :param N: 要复制几个
    :return: 返回一个list
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 第一个部分-word embedding
class Embedding(nn.Module):
    def __init__(self, dim_word, vocab):
        """
        用来写word embedding层的代码部分,感觉embeeding层来说浮现起来没有什么特别的地方，都是这么写出来就行
        :param dim_word: 一个词嵌入的大小为多少
        :param vocab: 单词本的大小（一共有多少个单词）
        """
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab, dim_word, )  # 第一个参数是单词本大小，第二个是词嵌入维度大小
        self.dim_word = dim_word

    def forward(self, x):
        # 除以维度的根号，是为了维护输出的大小稳定
        x = self.embedding(x)/math.sqrt(self.dim_word)
        return x

# 第二个部分-position encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim_word, dropout, max_len=5000):
        """
        用于创建位置编码,和参考的一样，不用管这块
        :param dim_word:  word embedding的维度大小
        :param dropout: dropout的出发比率
        :param max_len: 每一个句子最大的长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 计算位置编码-这里就这么写得了，也不用太管位置编码的问题-如果按照原文的代码，应该来说就是这么写，也不需要管太多

        # 先得到一个和一句话长度（是最大长度）一样长的的，和我们得到的word embeeding的维度一样宽的一个zeor矩阵
        position_encoding = torch.zeros(max_len, dim_word)
        # 1.先得到了一个一维的tensor，里面的元素是从0-(max_len-1)，2.然后把他扩充到二维，size维 [max_len-1, 1]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是根据公式进行计算了，暂时就不管了
        div_term = torch.exp(torch.arange(0, dim_word, 2) *
                             -(math.log(10000.0) / dim_word))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.unsqueeze(0)
        # 使用self.register_buffer会让"position_encoding"就是就是等于position_encoding
        # 其实也就是 self.position_encoding = position_encoding
        # 但是区别在于，使用register_buffer在保存参数的时候也会把这个东西保存下来，这样就不用每一次都要求p_e了
        self.register_buffer('position_encoding', position_encoding)

    def forward(self, x):
        x = x + Variable(self.position_encoding[:, :x.size(1)], requires_grad=False)
        # 其实这里我不太动为什么还要对word embedding进行一个dropout，有道理但也没有那么有道理，把某一一个word embeeding的特征去掉
        x = self.dropout(x)
        return x


# 第三个部分encoder部分，我的思路是-单头->多头->feedford->layerNorm
# 然后把一个完整的encoder写出来

# 3.1 单头注意力
class SingleAttention(nn.Module):
    def __init__(self, word_dim, h, mask=None, dropout=None):
        """
        原来我以为weighting matrix是由nn.Variable变得，但是其实是由nn.linear构成的
        在我的设计中，我希望在单头的时候生成weighting matrix，而源码是在多头处写的
        在单头中写出三个nn.linear层进行变换（没有成为一个list，为了直观）

        :param word_dim: 词嵌入的大小
        :param h: 分为几个头，只有知道了分成几个头在知道这个每一个子空间的映射是多少
        :param mask:    为了与encoder部分统一，所以先把mask给弄上来
        :param dropout:     dropout为概率
        """
        super(SingleAttention, self).__init__()
        self.word_dim = word_dim
        self.mask = mask        # 这个应该是矩阵mask
        # 我这里每一个单头都会生成一个dropout，而源码是多头生成了一个dropout对象，然后所有单头一起用一个，可能更剩内存之类的吧
        # 这里为了清楚，还是每一单头单独弄一个，因为有时候传dropout概率，有时又是对象，很容易弄混乱
        self.dropout = nn.Dropout(dropout)
        self.h = h
        # 每一个变换矩阵都单独写,就是为了清楚, input是word embedding的大小，output是子空间分割之后的大小
        # 源码比较迷离扑朔，这里写出来会很直观
        # 需要转成int，即使 是 int/int，得到是float
        self.d_k = int(self.word_dim/self.h)
        self.linear_q = nn.Linear(in_features=self.word_dim, out_features=self.d_k, dtype=torch.float32)
        self.linear_k = nn.Linear(self.word_dim, self.d_k, dtype=torch.float32)
        self.linear_v = nn.Linear(self.word_dim, self.d_k, dtype=torch.float32)

    def forward(self, x):
        # x [batch_size, 一句话的最大长度(称为max-length), 词嵌入的dim]
        # qkv [batch_size, max_length, qkv的维度大小]
        # 让一句话的长度保持一致是在整个模型开始之前做的
        x = x.to(torch.float32)
        query = self.linear_q(x)
        key = self.linear_k(x)
        value = self.linear_v(x)
        # [batch_size, max_len, word_dim/h]把后两个维度转一下相乘
        score = torch.matmul(query, key.transpose(-2, -1))
        # 这里的应该是要除以子空间维度的根号
        score = score / math.sqrt(self.d_k)
        if self.mask is not None:
            # 要在算softmax之前进行mask的遮掩才对
            # 这个语法是当mask中为0的部分，就把对应位置的元素换成-1e9次方，这样经过下面的softmax就跟0一样了
            # 注意一下，不直接成为0是因为 -10^9次方比0要小，这样才会在softmax中变成真的0
            score = score.masked_fill(self.mask == 0, -1e9)
        # 进行softmax #[batch_size, 1， probability of scores]
        p_atten = F.softmax(score, dim=-1)

        # 然后再看是否要dropout
        if self.dropout is not None:
            p_atten = self.dropout(p_atten)     # 这里的dropout是object，不是dropout的概率
        # 把所有的softmax权重过的value加起来   [batch_size, 1, word_dim/h]
        atten_sum = torch.matmul(p_atten, value)
        return atten_sum, p_atten

# 3.2 多头注意力
class MiltiHeadAttention(nn.Module):
    def __init__(self, h, dim_word, dropout=0.1, mask=None):
        """
        这里是写多头的
        :param h: 分成几个头
        :param dim_word: 词嵌入的唯独大小
        :param dropout: 概率
        :param mask: 掩膜
        """
        super(MiltiHeadAttention, self).__init__()
        # 不能整除应该是在最后的拼接部分比较麻烦，但是可以写麻烦的代码解决这个问题的
        assert dim_word & h == 0, "word embedding dimension不能被h整除"
        self.h = h
        self.dim_word = dim_word
        self.linear = nn.Linear(dim_word, dim_word)    # 用于最后子空间拼接出来的矩阵变换
        self.atten = None
        self.dropout = dropout
        self.mask = mask

    def forward(self, x):
        batch_size = x.size(0)

        singleAtten = nn.ModuleList()
        for i in range(self.h):
            singleAtten.append(SingleAttention(self.dim_word, self.h, self.mask, self.dropout))

        attention_sum_list = []
        p_atten_list = []
        for sA in singleAtten:
            # 这样单头做完的结果存在一个list中
            temp = sA(x)
            attention_sum_list.append(temp[0])
            p_atten_list.append(temp[1])

        # [batch_size, max_len, word_dim/h] -> [batch_size, max_len, word_dim]
        result_attention = torch.cat(attention_sum_list, dim=-1)  # 在最后一维度上串联
        self.atten = torch.cat(p_atten_list, dim=-1)    # 不知道这个有什么用，源码写了我也先写上
        # 这个就是最后经过线性变换之后得到的z，多头返回这个z
        z = self.linear(result_attention)
        return z

# 3.3 feedforward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, word_dim, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.word_dim = word_dim
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        # 前向的两个层
        self.w_1 = nn.Linear(word_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, word_dim)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

# 3.4 layerNorm, 似乎nn.layerNorm就可以，但是我看源码是自己写的一个
class LayerNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 根据features的形状初始化两个参数张量a2，和b2，第一初始化为1张量，
        # 也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，
        # 这两个张量就是规范化层的参数。因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，
        # 因此就需要有参数作为调节因子，使其即能满足规范化要求，又能不改变针对目标的表征，
        # 最后使用nn.parameter封装，代表他们是模型的参数
        self.a_2 = nn.Parameter(torch.ones(feature_size))
        self.b_2 = nn.Parameter(torch.zeros(feature_size))
        # 把eps传到类中，超级小的保证分母不是0的一个数
        self.eps = eps

    def forward(self, x):
        # 输入参数x代表来自上一层的输出，在函数中，首先对输入变量x求其最后一个维度的均值，
        # 并保持输出维度与输入维度一致，接着再求最后一个维度的标准差，然后就是根据规范化公式，
        # 用x减去均值除以标准差获得规范化的结果。
        # 最后对结果乘以我们的缩放参数，即a2,*号代表同型点乘，即对应位置进行乘法操作，加上位移参b2，返回即可
        mean = x.mean(-1, keepdim=True) # 求均值，都对着最后一个维度
        std = x.std(-1, keepdim=True)   # 求标准差，对着最后一个维度
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 3.5 one encoder
class Encoderlayer(nn.Module):
    """
    这里实现完整的一层encoder,不考虑词嵌入和位置编码，这个放到总的里面
    """
    def __init__(self, dim_word, h, d_ff, dropout, mask=None):
        super(Encoderlayer, self).__init__()
        self.dim_word = dim_word
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout
        self.mask = mask

        self.norm1 = LayerNorm(dim_word)
        self.norm2 = LayerNorm(dim_word)
        self.multi = MiltiHeadAttention(h, self.dim_word, dropout, mask)
        self.feedFord = PositionwiseFeedForward(self.dim_word, self.d_ff, self.dropout)

    def forward(self, x):
        # 按照图进行一层encoder的输出
        attention = self.multi(x)
        x1 = self.norm1(x + attention)
        x2 = self.feedFord(x1)
        x2 = self.norm2(x1 + x2)
        return x2

## 开始写decoder的部分

# 生成decoder那部分向后遮掩的mask
def subsequent_mask(size):
    # 生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，他最后两维形成了以恶搞方阵
    atten_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')

    # 最后将numpy类型转化为torch中的tensor，内部做一个1- 的操作。这个其实是做了一个三角阵的反转，
    # subsequent_mask中的每个元素都会被1减。
    # 如果是0，subsequent_mask中的该位置由0变成1. 如果是1，subsequent_mask中的该位置由1变成0
    return torch.from_numpy(subsequent_mask) == 0

# one decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, h, dim_word, d_ff, dropout, tgt_length):
        super(DecoderLayer, self).__init__()
        self.h = h
        self.dim_word = dim_word
        self.dropout = dropout
        self.d_ff = d_ff
        self.tgt_length = tgt_length
        self.mask = subsequent_mask(self.tgt_length)

        self.self_atten = MiltiHeadAttention(self.h, self.dim_word, self.dropout, self.mask)
        self.encDoc_atten = MiltiHeadAttention(self.h, self.dim_word, self.dropout)
        self.norm1 = LayerNorm(self.dim_word)
        self.norm2 = LayerNorm(self.dim_word)
        self.norm3 = LayerNorm(self.dim_word)
        self.feedforward = PositionwiseFeedForward(self.dim_word, self.d_ff, self.dropout)

    def forward(self, x, decoder_info):
        """
        :param x: x为输入的字（mask蒙上版本）或者是上一个decoder输出的key
        :param decoder_info: 由decoder最后一层编码而成的key的信息
        :return:
        """
        # 按照顺序慢慢写，需要残差的就起一个名字，不需要的就覆盖掉
        x1 = self.norm1(x+self.self_atten(x))
        x2 = self.norm2(x1 + self.encDoc_atten(x1 + decoder_info))
        x3 = self.norm3(x2 + self.feedforward(x2))
        return x3


# 最后的输出层
class Generator(nn.Module):
    """
    最后一个输出时候的线性和softmax, 然后至少要变成one-hot编码
    """
    def __init__(self, dim_model, vocab):
        super(Generator, self).__init__()
        # 首先就是使用nn中的预定义线性层进行实例化，得到一个对象self.proj等待使用
        # 这个线性层的参数有两个，就是初始化函数传进来的两个参数：d_model，vocab_size
        self.proj = nn.Linear(dim_model, vocab)

    def forward(self, x):
        # 前向逻辑函数中输入是上一层的输出张量x,在函数中，
        # 首先使用上一步得到的self.proj对x进行线性变化,然后使用F中已经实现的log_softmax进行softmax处理。
        x = F.log_softmax(self.proj(x), dim=-1)
        # 我要在generator中就把他变成one-hot编码输出
        # 找到每行中最大值的索引
        max_indices = torch.argmax(x, dim=-1)

        return max_indices