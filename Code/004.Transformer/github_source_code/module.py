import numpy as np  # 导入NumPy库，用于进行矩阵运算和数据处理
import torch  # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn  # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time  # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘制图表和可视化
import seaborn  # 导入Seaborn库，用于绘制统计图形和美化图表

## 最刚开始的word embedding和位置编码

class Embeddings(nn.Module):
    def __init__(self, dim_word_embedding, vocab):
        """
        类的初始化函数，是用来吧一个单词变成词嵌入的
        在transformer的最刚开始要把一个冰冷的单词变成生动的word embedding
        :param dim_model: 词嵌入的维度大小
        :param vocab: 词表的大小-我的理解是，比如一个文本，里面有多少个不同的字就会有多少个vocab
        """
        super(Embeddings, self).__init__()
        # 调用nn中预先定义的Embedding层，获得一个词嵌入对象,使用这个nn中预先定义的word embedding模型获得一个单词的
        # 第一个参数是代表了在某个训练任务中，一共存在多少个单词，第二个参数代表了 得到的word embedding的维度大小是多少
        # 这个embedding层也是会被训练的
        self.embedding = nn.Embedding(vocab, dim_word_embedding)
        self.dim_model = dim_word_embedding

    def forward(self, x):
        x = self.embedding(x)
        # 乘以词嵌入的维度的根号，为了保证输出稳定
        x = x * math.sqrt(self.dim_model)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim_word_embedding, dropout, max_len=5000):
        """
        用于创建位置的编码
        :param dim_word_embedding: word embedding的维度大小
        :param dropout: dropout的出发比率
        :param max_len: 每一个句子最大的长度
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码-这里就这么写得了，也不用太管位置编码的问题-如果按照原文的代码，应该来说就是这么写，也不需要管太多

        # 先得到一个和一句话长度（是最大长度）一样长的的，和我们得到的word embeeding的维度一样宽的一个zeor矩阵
        position_encoding = torch.zeros(max_len, dim_word_embedding)
        # 1.先得到了一个一维的tensor，里面的元素是从0-(max_len-1)，2.然后把他扩充到二维，size维 [max_len-1, 1]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是根据公式进行计算了，暂时就不管了
        div_term = torch.exp(torch.arange(0, dim_word_embedding, 2) *
                             -(math.log(10000.0) / dim_word_embedding))
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

def clones(module, N):
    """
    要复制一个module出来，方便集体操作，产生N个module出来，装到一个list里面
    :param module: 要被复制的module
    :param N: 要复制几个
    :return: 返回一个list
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

## encoder的代码编写部分

class Encoder(nn.Module):

    def __init__(self, layer, N):
        """
        把一个完整的编码器的一层(包括多头和前向)传进来，然后克隆N份，叠加在一起，得到了完整的Encoder，
        由于Transformer的输入和输出的大小都一样，所以可以写一个Encoder直接把他们全部都堆叠在一起，而不用逐个逐个的调整参数
        :param layer: 复制encoder的一层
        :param N: 复制N个
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)  # 这里进行复制
        self.norm = LayerNorm(layer.size) # 弄一个layerNorm-这个layerNorm是底下规定的layerNorm，在nn.LayerNorm应该也可以

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)     # 这里相当于encoder都结束了，最后做一个layerNorm

class Sublayeronnection(nn.Module):
    """
    实现子层的连接结构的类, forward中的sublayer是代表多头层或者是feed forward，由于他们都是经过一个多头或者前向，然后layerNorm，
    所以就在这里实现了一个类去做，可以在EncoderLayer类中看出来
    """
    def __init__(self, size, dropout):
        super(Sublayeronnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 原paper的方案
        sublayer_out = sublayer(x)
        x_norm = self.norm(x + self.dropout(sublayer_out))

        # sublayer_out = sublayer(x)
        # sublayer_out = self.dropout(sublayer_out)
        # x_norm = x + self.norm(sublayer_out)
        return x_norm

class EncoderLayer(nn.Module):
    """
    encoder是有两层子层构成的，self-attention和feed-forward
    """
    def __init__(self, size, self_atten, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_atten = self_atten        # 代表attention layer
        self.feed_forward = feed_forward    # 代表feed forward layer
        self.sublayer = clones(Sublayeronnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # attention sub layer
        x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, mask))
        # feed forward sub layer
        z = self.sublayer[1](x, self.feed_forward)
        return z

def attention(query, key, value, mask = None, dropout = None):
    """
    用来计算attention值的 单头注意力机制
    """
    # 首先取query的最后一维的大小，对应词嵌入维度
    d_k = query.size(-1)
    # 按照attention的公式，把query*key的转置，再除以缩放系数得到注意力张量score
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)    # 如果mask中元素为0，那么score对应的位置就是-1e9

    # 得到score的最后一维度进行softmax操作，得到注意力张量
    p_atten = F.softmax(scores, dim=-1)

    # 之后判断是否使用fropout进行随机置0
    if dropout is not None:
        p_atten = dropout(p_atten)

    # 得到注意力与key相乘的结果
    attention_sum = torch.matmul(p_atten, value)

    return attention_sum, p_atten

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, dim_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除
        # 这是因为我们之后要给每个头分配等量的词特征，也就是embedding_dim/head
        assert dim_model % h == 0
        # 得到的每个头获得分割词向量维度d_k
        self.d_k = dim_model // h
        self.h = h

        # 创建linear层，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim x embedding_dim，
        # 然后使用，为什么是四个呢，这是因为在多头注意力中，Q,K,V各需要一个，最后拼接的矩阵还需要一个，因此一共是四个
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)
        # self.attention为None，这个代表最后得到的注意力张量的变量名字，有了结果在赋值，现在是none
        self.atten = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 前向逻辑函数，它输入参数有四个，前三个就是注意力机制需要的Q,K,V
        # 最后一个是注意力机制中可能需要的mask掩码tensor，默认是None
        if mask is not None:
            # same mask applied to all h heads
            # 使用unsqueeze扩展维度，代表多头中的第n个头
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 首先利用zip将输入QKV与三个线性层组到一起，然后利用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结构进行维度重塑，
        # 多加了一个维度h代表头，这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值，然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维，这样我们就得到了每个头的输入
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # 得到每个头的输入后，接下来就是将他们传入到attention中，
        # 这里直接调用我们之前实现的attention函数，同时也将mask和dropout传入其中
        x, self.atten = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，
        # 我们需要将其转换为输入的形状以方便后续的计算，因此这里开始进行第一步处理环节的逆操作，
        # 先对第二和第三维进行转置，然后使用contiguous方法。这个方法的作用就是能够让转置后的张量应用view方法，
        # 否则将无法直接使用，所以，下一步就是使用view重塑形状，变成和输入形状相同。
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        # 最后使用线性层列表中的最后一个线性变换得到最终的多头注意力结构的输出
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, d_ff)
        self.w_2 = nn.Linear(d_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

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

def subsequent_mask(size):
    # 生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，他最后两维形成了以恶搞方阵
    atten_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')

    # 最后将numpy类型转化为torch中的tensor，内部做一个1- 的操作。这个其实是做了一个三角阵的反转，
    # subsequent_mask中的每个元素都会被1减。
    # 如果是0，subsequent_mask中的该位置由0变成1. 如果是1，subsequent_mask中的该位置由1变成0
    return torch.from_numpy(subsequent_mask) == 0

## decoder部分

class Decoder(nn.Module):
    def __init__(self, layer, N):
        # 初始化的参数有连个，第一个是解码器层layer，第二个是解码器的个数N
        super(Decoder, self).__init__()
        # 首先使用clones方法克隆了N个layer，然后实例化一个layernorm，
        # 因为数据走过了所有的解码器层后最后要做规范化处理。
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # forward函数中的参数有4个，
        # x代表目标数据的嵌入表示，memory是编码器层的输出，
        # source_mask，target_mask代表源数据和目标数据的掩码张量，然后就是对每个层进行循环，
        # 当然这个循环就是变量x通过每一个层的处理，得出最后的结果，再进行一次规范化返回即可。
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_atten, src_atten, feed_forward, dropout):
        # 初始化函数的参数有5个，分别是size，代表词嵌入的维度大小，
        # 同时也代表解码器的尺寸，第二个是self_attn，多头自注意力对象，也就是说这个注意力机制需要Q=K=V，
        # 第三个是src_attn,多头注意力对象，这里Q!=K=V，
        # 第四个是前馈全连接层对象，
        # 最后就是dropout置0比率
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_atten = self_atten
        self.src_atten = src_atten
        self.feed_forward = feed_forward
        self.sublayer = clones(Sublayeronnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # forward函数中的参数有4个，分别是来自上一层的输入x，来自编码器层的语义存储变量memory，
        # 以及源数据掩码张量和目标数据掩码张量，将memory表示成m之后方便使用。
        m = memory
        # 将x传入第一个子层结构，第一个子层结构的输入分别是x和self-attn函数，
        # 因为是自注意力机制，所以Q,K,V都是x，最后一个参数时目标数据掩码张量，
        # 这时要对目标数据进行遮掩，因为此时模型可能还没有生成任何目标数据。
        # 比如在解码器准备生成第一个字符或词汇时，我们其实已经传入了第一个字符以便计算损失，
        # 但是我们不希望在生成第一个字符时模型能利用这个信息，因此我们会将其遮掩，同样生成第二个字符或词汇时，
        # 模型只能使用第一个字符或词汇信息，第二个字符以及之后的信息都不允许被模型使用。
        x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, tgt_mask))
        # 接着进入第二个子层，这个子层中常规的注意力机制，q是输入x;k,v是编码层输出memory，
        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄露，
        # 而是遮蔽掉对结果没有意义的padding。
        x = self.sublayer[1](x, lambda x: self.src_atten(x, m, m, src_mask))
        # 最后一个子层就是前馈全连接子层，经过它的处理后就可以返回结果，这就是我们的解码器结构
        return self.sublayer[2](x, self.feed_forward)

class Generator(nn.Module):
    """
    最后一个输出时候的线性和softmax
    """
    def __init__(self, dim_model, vocab):
        super(Generator, self).__init__()
        # 首先就是使用nn中的预定义线性层进行实例化，得到一个对象self.proj等待使用
        # 这个线性层的参数有两个，就是初始化函数传进来的两个参数：d_model，vocab_size
        self.proj = nn.Linear(dim_model, vocab)

    def forward(self, x):
        # 前向逻辑函数中输入是上一层的输出张量x,在函数中，
        # 首先使用上一步得到的self.proj对x进行线性变化,然后使用F中已经实现的log_softmax进行softmax处理。
        return F.log_softmax(self.proj(x), dim=-1)







