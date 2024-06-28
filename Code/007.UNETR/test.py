import torch

print(torch.cuda.is_available())

x = torch.randn((1, 10, 128, 128, 128))
print(x.size())
con = torch.nn.Conv3d(10, 20, 4, 4)
x1 = con(x)
print(x1.size())