#  最后的结果是模型不收敛，这应该是因为decoder有小问题导致的，但是我相信我写出来的代码和注释可以让大部分新手快速了解transformer
#   在了解了我的代码之后，可以看看源码的写法，可以更好的帮助入门coding部分
#   使用这个simple test是因为：1）这个simple test确实简单容易参考， 参考文章 https://blog.csdn.net/m0_47779101/article/details/128087403?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171428426016800197089781%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171428426016800197089781&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-128087403-null-null.142^v100^pc_search_result_base4&utm_term=transformer%E7%9A%84%E6%BA%90%E7%A0%81&spm=1018.2226.3001.4187
#                           2）源码的代码太长了，也懒得看，只是最为入门练习感觉看源码的难度很大
#                           3） 我来写transformer是因为我想要来做transformer使用在vision相关的文章是怎么进行ViT,Swin Transformer之类的，所以快速练习是我想要的
#   写代码真的很对理解文章模型有帮助，虽然我的代码有一点点小问题，但是我感觉对于理解方便是很好

import numpy as np  # 导入NumPy库，用于进行矩阵运算和数据处理
import torch  # 导入PyTorch库，用于构建神经网络及相关操作
import torch.nn as nn  # 导入PyTorch神经网络模块，用于构建神经网络层
import torch.nn.functional as F  # 导入PyTorch神经网络函数库，用于激活函数、损失函数等
import math, copy, time  # 导入数学库、复制库和时间库，用于各种数学计算、复制操作和计时
from torch.autograd import Variable  # 从PyTorch自动微分库中导入Variable类，用于构建自动微分计算图
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘制图表和可视化
from model_muyi import Transform
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_batch(sentences):
    # 把文本转成词表索引
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    # 把索引转成tensor类型
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def train(model, sentences, epochs):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(epochs):
        optimizer.zero_grad()
        # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        outputs = model(enc_inputs, dec_inputs)
        # output:[batch_size x tgt_len,tgt_vocab_size]
        outputs = outputs.to(torch.float32)
        target_batch = target_batch.to(torch.float32)
        loss = criterion(outputs, target_batch).requires_grad_(True)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        print(outputs)
        print(target_batch)

        loss.backward()
        optimizer.step()

    return model




if __name__ == "__main__":
    # P为填充， S为start，E为end
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # length of source
    tgt_len = 5  # length of target

    ## 模型参数
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    dropout = 0.1

    model = Transform(d_model, src_len, tgt_len, dropout, h=n_heads, N=n_layers, d_ff=d_ff,
                      src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    print(model)
    # model.enable_input_require_grads()
    train(model, sentences, epochs=100)


