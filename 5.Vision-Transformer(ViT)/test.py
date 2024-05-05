import torch
from torch import nn

# def sss(a):
#     return a+100
#
#
# a = [1, 2, 3, 4, 5, 6]
# c = map(sss, a)
#
# for i in c:
#     print(i)


a = nn.Parameter(torch.randn(1, 50, 790))
print(a.size())