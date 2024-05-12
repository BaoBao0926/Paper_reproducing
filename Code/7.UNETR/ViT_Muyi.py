import torch
from torch import nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    # 如果t是tuple直接返回，如果t不是tuple，返回一个tuple (t, t)
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class FeedFordward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super(FeedFordward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),          # 与transformer不一样，先进行layerNorm
            # 然后进入MLP内部
            nn.Linear(dim, hidden_dim),     # 先进行一层MLP，从输入的维度 转换为 hidden_dim
            nn.GELU(),                  # 使用非线性层，ViT中用的都是GELU()
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), # 最后来一个MLP，从hidden_dim->输入维度，保证维度统一
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        # 这里其实怪怪的， 我的理解是，为了使用一个大矩阵去加速计算，(使用一堆小矩阵，容易思考和写代码，但是不利于GPU计算)
        # 所以这里的做法是，先把词嵌入维度大小的弄出来，然后在分开，这里的inner_dim其实是词嵌入的大小
        # 如果一个patch是16*16， 那么inner_dim也就是16*16*3=768
        # 也就是head_dim代表单头的dim是多少，inner_dim代表外面的词嵌入dim为多少
        inner_dim = dim_head * heads
        # 如果只有一个head，相当于没有进行多头，只进行了单头
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads      # 多头注意力有几个头
        # scale是attention计算中，把score除以dim的根号，这里是写成了乘以 所以是负的，维护数值稳定
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)   # LayerNorm层，需要知道一层的维度
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_out = nn.Sequential(    # 这里的线性层是把 单头的key -> 最后的key
            nn.Linear(inner_dim, dim),  # 在transformer中也是有的
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        # 如果是单头，那么就使用Identity()，这个层并没有什么实际的用处，类似于占位，保持结构统一

    def forward(self, x):
        # x [batch_size, cls+patch的数量, 词嵌入的数量]
        x = self.norm(x)    # 还是，先进行Norm
        # qkv [batch_size, cls+patch的数量， 3*单头的词嵌入维度]， 变成3倍的，也就是qkv三个
        qkv = self.to_qkv(x)    # 得到了qkv的大矩阵
        # 现在的qkv是一个元组，里面有三个元素 each[batch_size, cls+patch的数量， 单头的词嵌入维度]
        qkv = qkv.chunk(3, dim=-1)

        # 分别得到了qkv
        # map()函数，第一个参数是function，这里用的是lambda函数，第二个参数是可迭代的数据容器,最后得到第二个参数对应的经过function的结果
        # 前面的lambda参数是输入一个t，对t进行einops的维度变换，einops是很好用的变换tensor维度的一个包，我也是看了ViT才知道，这个包很容易用，搜一下就会了
        # 把传入的t，t其实就是qkv的某一个，qkv的每一个的维度是[b n (h*d)]--[batch_size, cls+patch数量, 词嵌入的维度]，
        # 而这里词嵌入的维度应该要被分成h个单头的维度作为子空间进行计算，也就是多头在干的事情，所以这行代码要把h个单头qkv给变换出来，并且按照矩阵的样子，所以多了一个维度
        # 变换之后的维度 [batch_size, cls+patch数量，head的数量， 单头词嵌入的维度]，单头词嵌入维度 = 词嵌入维度（16*16*3）/ head数量(8)

        # 在 https://www.bilibili.com/video/BV1cS4y1M7wo?vd_source=80b346be9e1c1a93109688bf064e5be1 这里中提到了一点
        # transformer实际的运行是每一个子空间都应该需要单独的一个线性层 让 词嵌入->单头的词嵌入，但是这里由于没有decoder，decoder的输入会有encoder的部分，所以transformer是不可以的
        # 所以 ViT可以用一个整体的MLP去共同得到,这里的代码也很简洁，容易懂。如果觉得这个有问题，其实我感觉还是可以弄h个线性层对应的生成
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # Self-attention中的 K和V的计算
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)    # 对最后维度进行softmax
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)     # attention最后的公式 现在的维度 [batch_size,cls+patch数量,h,单头的维度]
        out = rearrange(out, 'b h n d -> b n (h d)')    # 使用rearrange合并，[batch_size, clas+path数量, 词嵌入的维度]
        out = self.to_out(out)      # 最后经过一个线性层，得到最后的输出
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):      # 残差是在forward中写的
            self.layers.append((
                nn.ModuleList([
                    Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    FeedFordward(dim, mlp_dim, dropout=dropout)
                ])
            ))

    def forward(self, x):
        values = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            values.append(x)

        return values


class UNETR_PatchEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        # number of patch
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size    # patch的大小宽和高
        self.embed_dim = embed_dim      # 输出的embedding的大小
        # 把一个patch变成一个embedding的做法是，使用3D卷积进行，得到的一个就是embedding
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        # 位置编码瞬间就在这里加上去了，这里的位置编码是可训练参数，没有用ViT里面的posem_sincos_2d那玩意
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads,
                 mlp_dim, channel=3, dim_head=64, dropout=0., cube_size=(128, 128, 128)):
        """
        *的作用是 *后面的所有参数都必须要使用关键字传参
        :param image_size: 图片的大小，默认为256
        :param patch_size: 一个patch的大小，默认为16
        :param dim: 一个patch展开的大小（1024）
        :param depth: transformer的深度，重复几次encoder，默认为8
        :param heads: 多头自注意力中使用几个头
        :param mlp_dim: feedforward中 hidden layer的维度是多少
        :param channel: 图片是几个通道的
        :param dim_head: 每一个单头的维度是多少
        :param dropout: 用于训练的dropout比率是多少
        :param cube_size: 3D图片的大小
        """
        super(ViT, self).__init__()
        # 先得到img的大小和patch的大小
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, "The image dim must be divisible by patch dim"
        # 获得一个patch展开之后的大小
        patch_dim = channel*patch_width*patch_height

        # # 把img->变成patch的样子,这里是ViT对于标准的3通道RGB图片的做法，现在用于UNETR里面的，需要修改一下
        # self.to_patch_embedding = nn.Sequential(
        #     # [batch_size, channel, img_height, img_width]->[b_size, patch number, patch的维度(16*16*3)]
        #     Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim)
        # )
        # 使用UNETR的to-patch-embedding
        self.to_patch_embedding = UNETR_PatchEmbedding(
            input_dim=channel,
            embed_dim=dim,
            cube_size=cube_size,
            patch_size=patch_size,
            dropout=dropout
        )

        # 这个位置编码是我看到的一个，这里似乎不太对，但是我的理解是，这里的位置编码是可训练的参数， 放在这里不管了
        # self.pos_embedding = nn.Parameter(torch.randn(1, (image_height/patch_height)*(image_width/patch_width)+1, dim))
        self.Dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head,
                                       mlp_dim=mlp_dim, dropout=dropout)

        self.to_latent = nn.Identity()


    def forward(self, img):
        # 两个获得device的方法一样，但是我感觉第二个更好一点
        # device = img.device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        x = self.to_patch_embedding(img)
        x = self.Dropout(x)
        x = self.transformer(x)

        return x


if __name__ == "__main__":

    v = ViT(
        image_size=128,
        patch_size=16,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        channel=1,
        cube_size=(128, 128, 128)
    )

    img = torch.randn(10, 1, 128, 128, 128)
    x = v(img)
    print(x[0].size())