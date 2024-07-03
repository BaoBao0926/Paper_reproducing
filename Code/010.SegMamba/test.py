import torch

a = torch.arange(35)
# print(a.shape)
# print(a.shape[2:], type(a.shape[2:]))
# n_tokens = a.shape[2:].numel()
# print(n_tokens, type(n_tokens))

b = a.chunk(5, dim=-1)
c = torch.stack(b, dim=-1)
d = c.flatten(-2)
vv = 1
