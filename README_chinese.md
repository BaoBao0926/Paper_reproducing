## Hi, I am [Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io)

---

[English](https://github.com/BaoBao0926/Paper_reproducing) | [简体中文](https://github.com/BaoBao0926/Paper_reproducing/blob/main/README_chinese.md)

---



这个仓库是一个我复现过的文章的overview，我想把我复现出来的代码展示出来，并且给出一些我对这些项目或者paper简单的思考。并且由于很多时候我也是一个新手，所以我写了非常详细的注释帮助自己理解。有一些我看过我想记录一下的工作我放在这个[仓库](https://github.com/BaoBao0926/Paper_reading)


  <!--    -----------------------------------------1.CapsNet -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">1. Capsule Network</b> 2023/11
   </summary>   
   
   <br />
   
  1.CapsNet将常用的标量(这篇paper认为CNN中通常使用的矩阵都是标量，但有时我们可能会说这些是向量或矩阵)转换为vector，并提出了一种算法Dynamic Routing。在我看来，动态路由对于特征提取来说是非常强大的，至少它给特征提取提供了一个新的思路。

2.它使用了胶囊的概念。

但是训练CapsNet是昂贵的。此外，与现有模型相比，CapsNet在更一般和更复杂的数据集上表现出了不足，难以处理复杂的数据集

  I refer this [repository](https://github.com/gram-ai/capsule-networks) to write the code

  Paper: [Dynamic Routing Between Capsules](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html)

Architecture:
  
  <img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/001.Capsule%20Network/583dc5ed79e1282895f8cd937e3a17e.png" alt="Model" style="width: 500px; height: auto;"/>
 
  Dynamic Routing Algorithm:
  
  <img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/001.Capsule%20Network/93dd912e6da6c2b7ec1df004c736e8e.png" alt="Model" style="width: 500px; height: auto;"/>
  
</details>


  <!--    -----------------------------------------2. U-Net   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">2.U-Net</b> 2024/4/4
   </summary>   
   
   <br />
   
 U-Net用于segmentation任务。该架构相对简单，因此适合初学者开始学习如何处理分段任务。这个model被用于处理医学图片最刚开始的时候。
 
 我看过一些解释，由于医学图片（比如CT照片）的结构都是限制的，大体样子都是一样的，所以比较浅层的模型的效果可能更好。现在而言，绝大部分的分割，至少在medical image这边，都会使用的是U-Net的结构


   Paper: [U-Net-Based medical image segmentation](https://ncbi.longhoe.net/pmc/articles/PMC9033381/)

Architecture:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/002.U-Net/architecutre.png" alt="Model" style="width: 500px; height: auto;"/>
   
</details>


  <!--    -----------------------------------------  3.Learning without forgetting   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">3.Learning without forgetting</b> 2024/4/18
   </summary>   
   
   <br />
   

无遗忘学习(LwF)用于处理classification任务中的continual learniing任务。一些论文认为这篇文章是第一个系统定义continual learning(CL)的论文。在我看来，它确实给了CL很多启示。

就其方法而言，我认为是将知识精馏(Knowledge Distinallation KD)应用于CL领域的最简单的方法。这个项目非常适合想要使用KD学习持续学习的新初学者。

此外，它的学习方法是在一个数据集中不断地学习一个类。以CUB-200数据集为例，它将一次学习一个类别。通常，我们可能会认为一次学习一个数据集的所有类别。

我在这个项目的复现代码中给出很详细的注释. 我参考了这个[仓库](https://github.com/ngailapdi/LWF), 我的代码结构与它的并不太一样，可能他的代码更高效，但我认为我的代码更容易理解，我加了许多注释


Paper: [Learning without Forgetting](https://ieeexplore.ieee.org/abstract/document/8107520)

Original Repository: [here](https://github.com/lizhitwo/LearningWithoutForgetting)

Architecture:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/003.Learning-without-forgetting/architecture.png" alt="Model" style="width: 600px; height: auto;"/>

Algorithm:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/003.Learning-without-forgetting/algorithm.png" alt="Model" style="width: 500px; height: auto;"/>

</details>



  <!--    ----------------------------------------- 4.Transformer   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">4.Transformer </b>  2024/4/25
   </summary>   
   
   <br />
   
有很多论文和存储库来解释它。我也学习了这些见解。

我之所以知道这一点，是因为在2021年变压器被用于计算机视觉(Vision transformer ViT)。因此，我学习了应该用在NLP中的Transformer。

I learn Transformer by this [blog](https://blog.csdn.net/benzhujie1245com/article/details/117173090?spm=1001.2014.3001.5506), offering very detailed explanation.

I refer this [repository](https://github.com/datawhalechina/dive-into-cv-pytorch) 's code to write my code. 我给出了许多详细的解释，并重新构建了代码框架，以便新手(也包括我自己)更容易学习，然后可以理解源代码是做什么的。

Paper: [Attention is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

The architecture:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/004.Transformer/358b56267a5fde9e4c42fae0f31a635.png" alt="Model" style="width: 350px; height: auto;"/>


</details>


  <!--    ----------------------------------------- 5.Vision Transformer   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">5.Vision Transformer</b> 2024/5/5
   </summary>   
   
   <br />
   
在2021年，一个团队使用了几乎不变的Transformer用于图像分类，这给了人们一个想法，原来用于NLP的Transformer也可以用于计算机视觉。这在视野上是一个巨大的进步。基于transoframer的模型已经打破了许多记录。它证明了变压器可以在CV中使用，如果在规模上，变压器甚至可以表现得更好。基于这篇文章，大量的文章诞生了去提高它

如果写过了Transformer，那么Vision Transformer也是很简单的，因为基本上和Transformer一样，而且没有decoder，只是多了一个Patch embedding

I learn ViT through this [bilibili vedio](https://www.bilibili.com/video/BV15P4y137jb?vd_source=80b346be9e1c1a93109688bf064e5be1) and this [one](https://www.bilibili.com/video/BV1Uu411o7oY?p=2&vd_source=80b346be9e1c1a93109688bf064e5be1), this [blog](https://blog.csdn.net/qq_51957239/article/details/132912677?spm=1001.2014.3001.5506).

Writing code refers to this [bilibili vedio](https://www.bilibili.com/video/BV1Uu411o7oY?p=2&vd_source=80b346be9e1c1a93109688bf064e5be1) and this [repository](https://github.com/lucidrains/vit-pytorch) and the [authrity repository](https://github.com/google-research/vision_transformer)

Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

The architecture: 

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/005.Vision-Transformer(ViT)/87c2a66be6f2a38f76d2a158fe79f28.png" alt="Model" style="width: 700px; height: auto;"/>

</details>



 <!--    ----------------------------------------- 6.Swin Transformer   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">6.Swin Transformer</b> 2024/5/11
   </summary>   
   
   <br />

Swin变压器是基于视觉变压器(Vision Transformer, ViT)的一项工作，解决了图像分辨率大、计算复杂度高的问题。这几乎是一项里程碑式的工作，打破了无数计算机视觉任务的记录。证明了Swim Transformer可以作为transformer的通用骨干。

它的代码很好，我从中学到了很多。我鼓励每个人都复制这段代码，它一定能给你很多洞察力，提高你的编码能力。

在它的论文和许多资源中，它说最好有一个预先训练。我只是在FOOD101上训练游泳变压器(就像一个简单的实验)。我发现了三个问题:1)训练非常困难，需要很大的计算成本(在此之前，我只是训练CNN而不是基于Transformer的模型)。2)从头开始训练网络会有很差的初始结果3)超参数，即学习率非常重要。这些都是我的发现，可能是错的。


The source I refer: a bilibili [vedio](https://www.bilibili.com/video/BV13L4y1475U?vd_source=80b346be9e1c1a93109688bf064e5be1) to explain paper, 
a bilibili [vedio](https://www.bilibili.com/video/BV1zT4y197Fe?p=2&vd_source=80b346be9e1c1a93109688bf064e5be1) to explain to code, a CSDN [blog](https://blog.csdn.net/qq_45848817/article/details/127105956?ops_request_misc=&request_id=&biz_id=102&utm_term=Swim%20transformer%E4%BB%8B%E7%BB%8D&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-127105956.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187) to explain the Swim Transformer,
a CSDN [blog](https://blog.csdn.net/beginner1207/article/details/138034012?ops_request_misc=&request_id=&biz_id=102&utm_term=Droppath&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-138034012.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187) to introduce Dropath(it is my first time to see this),

Original paper: [Swin transformer: Hierarchical vision transformer using shifted windows](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper)

Official repository: [here](https://github.com/microsoft/Swin-Transformer)

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/006.Swin-Transformer/1fec248384cc012c87ac288d50e980f.png" alt="Model" style="width: 700px; height: auto;"/>

</details>



 <!--    ----------------------------------------- 7.Unet-Transformer (UNETR)   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">7.Unet-Transformer (UNETR)</b> 2024/5/12
   </summary>   
   
   <br />

在视觉转换器(Vision Transformer, ViT)工作的基础上，提出了一种用于医学三维图像处理的UNEt-TRansformer (UNETR)工作。整个架构类似于U-net，编码器被ViT取代。

这是我第一次看到如何处理3D图像。处理3D是完全不同的。通常使用torch.nn.Conv3d, 最不同的是图像维度大小, 3D图像的尺寸类似于(batch_size, one image channel, height(frame), height, width)。以视频为例:如果有10个视频，每个视频由20帧组成，RGB图像(3通道)，224*224像素，则为(10,3,20,224)

还有一个基于这个和Swin-Transformer的作品，叫做Swin-UNETR，应该是非常相似的。

官方库中的代码使用monai库，可以为代码更改建议提供快速跟踪，并展示前沿的研究思想。但是在我的代码中，我使用了自己复制的ViT代码来复制UNETR。

我认为如果你已经实现了ViT或者想要使用monai库，实现UNETR并不是一件困难的事情。

训练这种基于变压器的网络是一种计算成本。我使用我的计算机(仅CPU)运行图像大小为(2,1,128,128,128)的转发部分，这大约需要一分钟。没有好的GPU，很难得到结果。这也是我第一次直观地感受到Transtranser消耗了多少计算资源。

Original paper: [Unetr: Transformers for 3d medical image segmentation](https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html)

Official repository: [here](https://github.com/Project-MONAI/research-contributions/tree/main)

Refered repository: [here](https://github.com/tamasino52/UNETR/blob/main/unetr.py)


<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/raw/main/Code/007.UNETR/model.png" alt="Model" style="width: 700px; height: auto;"/>


</details>



  <!--    ----------------------------------------- 8.Mamba   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">8.Mamba</b> 2024/6/22
   </summary>   
   
   <br />

从结果和表现的角度来看，曼巴似乎可以撼动transformer的地位。曼巴可以比变形金刚略胜一筹，但计算速度要快得多。它似乎是变形金刚的替代品。随着Transformer的发展，缺点之一是时间复杂度为O(n^2)。随着模型越来越大，问题变得越来越严重。而Mamba是O(n)，可以很好的解决这个问题。

另一点是，Transformer的自我关注机制实际上没有任何理论支持，它似乎只是模块的拼凑(尽管它似乎有意义)。但曼巴是由状态空间模型理论(State Space Model)支持的，这是我在本科Y3所学到的。这使曼巴语具有更高的可解释性。在某种程度上，曼巴与RNN/LSTM有着非常相似的想法。它们是一种正向流，从前一个输入到下一个输入。

总之，我认为Mamba有很多优势，它在诞生之初就可以比transformer做得更好，而且它的出现有望极大地促进该领域的发展，至少使用SSM的想法是伟大的。

曼巴的论文非常抽象。幸运的是，许多博客和视频试图解释它，这给了我很多见解。

Paper: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

Official Repository: [here](https://github.com/state-spaces/mamba/tree/main) 

I recommend this [CSND blog](https://blog.csdn.net/v_JULY_v/article/details/134923301?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171905345716800182784276%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=171905345716800182784276&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2)

I recommend these BiliBili videos: [1](https://www.bilibili.com/video/BV1vF4m1F7KG?vd_source=80b346be9e1c1a93109688bf064e5be1), [2](https://www.bilibili.com/video/BV1KH4y1W7cm?vd_source=80b346be9e1c1a93109688bf064e5be1), [3](https://www.bilibili.com/video/BV1gy411Y7xa?vd_source=80b346be9e1c1a93109688bf064e5be1), [4](https://www.bilibili.com/video/BV1hf421D7km?vd_source=80b346be9e1c1a93109688bf064e5be1) and [5](https://www.bilibili.com/video/BV1Xn4y1o7TE?vd_source=80b346be9e1c1a93109688bf064e5be1). After seeing these videos, I get a lots of insights and know what Mamba is.

虽然有很多资料来解释什么是曼巴，我认为曼巴的代码和架构不是很清楚，这些资料并没有把重点放在代码上。但是我找到了这个[repository](https://github.com/johnma2006/mamba-minimal)，它提供了最小的实现。看了这段代码后，我基本上知道了曼巴代码是什么。在我重新编写的代码中，我给出了详细的注释来解释每个部分。

- mamba_minimal.py is the work of the [repository](https://github.com/johnma2006/mamba-minimal) mentioned above.

- mamba_minimal_muyi.py is what I reproduced and give detailed comments.

- mamba_main is official full implementation and I give some comments.

I put some import picture here:

The whole architecture demo:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/008.Mamba/pictures/whole_architecture.png" alt="Model" style="width: 700px; height: auto;"/>

The formula for delta,A,B,C,D:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/008.Mamba/pictures/formula.png" alt="Model" style="width: 700px; height: auto;"/>

The algorithm for SSM:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/008.Mamba/pictures/algorithm.png" alt="Model" style="width: 700px; height: auto;"/>

The Mamba block architecture:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/008.Mamba/pictures/architecture.png" alt="Model" style="width: 700px; height: auto;"/>

</details>




 <!--    ----------------------------------------- 9.Vision Mamba(Vim)   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">9.Vision Mamba(Vim)</b> 2024/6/25
   </summary>   
   
   <br />

与变形金刚和视觉变形金刚的关系非常相似，Vision Mamba(Vim)在曼巴的基础上也有类似的想法。Vim有潜力成为新CV领域的通用支柱。性能和速度都高于变形金刚。

此外，我有一个想法，由于Mamba可以处理非常长的文本序列(例如数百万像素)，因此无论图像有多少像素，图像都不太可能达到数百万个补丁。因此，在处理图像时，Vim不应该忘记太多以前的补丁内容(Vim是一个时序模型)。因此，将图像作为时间序列数据处理不会降低性能。Vision transfrmer不会降低性能，因为它是并行的，每个patch都是同时计算的。当然实际处理的时候会下面介绍的创新2，多视角的去看图片

Vision Mmaba有两大创新:

1.曼巴在计算机视觉领域的应用。

2.使用双向SSM，导致了大量类似的工作。


I only see this Bilibili [video](https://www.bilibili.com/video/BV1hf421D7km?vd_source=80b346be9e1c1a93109688bf064e5be1). I know Vim when I learn Mamba. This is not too hard because it is very similar with the relationship of Transformer and Vision Transformer.

The official repository is [here](https://github.com/hustvl/Vim). 

The paper: [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)

至于代码，我没有看到任何可以帮助人们理解的代码。在我重新编码的代码中，我制作了一个玩具版本(非常简单的一个，类似于mamba_minimal)。我也给出了一个非常详细的注释在源代码的Vision Mamba。在源代码中，我发现有些地方似乎是错误的:当进行双向SSM时，它使用两个Vim块，一个用于向前，另一个用于向后。这与纸上描述的结构不符。我也在下面展示了real architecture。


The Vision Mamba architecture:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/009.Vision%20Mamba(Vim)/architecture.png" alt="Model" style="width: 800px; height: auto;"/>

The real Vim architecture in code:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/009.Vision%20Mamba(Vim)/real_architecture.png" alt="Model" style="width: 650px; height: auto;"/>


The Vision Mmaba algorithm:

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/009.Vision%20Mamba(Vim)/algorithm.png" alt="Model" style="width: 350px; height: auto;"/>

</details>



 <!--    ----------------------------------------- 10.SegMamba   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">10.SegMamba</b> 2024/7/3
   </summary>   
   
   <br />

贡献:
- 采用U-Net架构
- 第一层是Stem Convolutional Network, kernel size为7 * 7 * 7,padding为3 * 3 * 3,stride为2 * 2 * 2。在第一段中提到，一些研究发现利用large kernel改进视场，从高分辨率的三维图像中提取大范围信息是有用的。
  - 实际上，这个Stem卷积层类似于Patch Embedding。
- Mamba块被TSMamba块取代，如图2所示。
- 解码器是基于CNN的

至于代码:

- 这篇文章改写了Mamba。但我认为nslices在片间方向上有一些错误。
  - xz: [B, L, D] and nslices设置为[64, 32, 16, 8]
  - 例如，如果xz为[1,2,3,4.....35]和nslice = 5。实现后，xz变为[0,7,14,21,28,1,8…]
  - 表示interval = token总数/ nslice => nslice = token总数/interval = H * W * D/H * W = D
  - 因此，我们应该将nslics设置为D，而不是固定数字
- 与U-Mamba、VM-UNet和nnMamba的代码相比，此代码相对简单。

我看到一些关于视觉曼巴和医学图像分割的论文。在这篇论文中，我同时看到了U-Mamba, nnMamba和VM-UNet。除了VM-UNet之外，这三篇论文都没有使用补丁嵌入，而是使用了Stem Convlution。

 The Paper: [SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation](https://arxiv.org/pdf/2401.13560)

 The official repository: [Here](https://github.com/ge-xing/SegMamba)

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/SegMamba.png" alt="Model" style="width: 800px; height: auto;"/>



</details>




 <!--    ----------------------------------------- 11.UltraLight VM-UNet   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">11.UltraLight VM-UNet</b> 2024/7/9
   </summary>   
   
   <br />


贡献:

- 这项工作的最大贡献是轻量级模型。与[Light MUNet](https://arxiv.org/pdf/2403.05246)(or参见此[repository](https://github.com/BaoBao0926/Paper_reading/blob/main/VisionMamba_3DSegmentation_medicalImage_Chinese.md))相比，它的参数减少了87%，只有0.049M参数和0.06GFLOPs。提出的PVM层是一个即插即用模块，这是非常好的。
- 整体架构为U-Net架构，下采样层为maxpooling。编码器使用3层ConV Block和3层PVM Layer。解码器是对称的，也是3层卷积，3层PVM层。中间的跳跃连接使用SAB和CAB(空间注意桥和通道注意桥)
  - 编码器部分:共6层。前三层是Conv Layer。最后4层为PVM层
  - 连接部分，由共享参数的SAB和CAB组成
    - SAB(x) = x + x * Conv2d(k=7)([MaxPool(x);AvgPool (x)))
    - CAB(x) = x + x * Sigmoid(FC(GAP(x)))
  - 解码器部分:与编码器对称，由3个Conv层和3个PVM层组成
- PVM层:
  - 核心思想如图3所示。我们把通道分成四个部分，在每个部分上执行一个Mamba操作(从代码上看，每个通道组都是同一个曼巴)，这样可以节省很多参数，最后把它们放在一起
  - 在我没有放在这里的图4中，如果直接对C通道的数量执行Mamba，需要x参数，那么对C/2执行两次Mamba，只需要2*0.251(两个C/2是单独的曼巴)。对于4 * C/4，只需要0.063 * 4个参数
  - 整体外观非常简单，参数很少，效果还不错，虽然不是最好的，ISIC2017 DSC SE是最好的，PH^2是最好的，ISIC2018是DSC和ACC上最好的
- 代码中的实现细节
  - 首先，关于CAB的实施，我们可以看到，Fig.2中的CAB实际上还有一个阶段，这是我以前没有见过的。实际上，6个阶段的输出应该放在一起，然后通过相应的线性层映射到各自的维度，所以这里实际上是综合了每个阶段的信息
  - 对于skip connection，从图2中可以看出，每一stage都要经过SAB CAB，但实际上并非如此。根据代码，阶段6不经过SAB CAB，甚至跳过连接。其实有点阶段作为bottleneck的感觉，这绝对不是代码错误，因为上面提到的CAB是所有stage组合在一起，但代码实际上只是前5个stage组合在一起
  - maxpooling with stride=2 and size=2用于下采样
  - 编码器卷积是所有尺寸=3，步幅=1，填充=1
  - 解码器的最后一个卷积实际上是一个分割头，输出num_class, size=1。另外两个解码器size=3, stride=1, padding=1

至于我的代码，我添加了两个超参数来控制分割通道的数量以及是否使用相同的曼巴。我也让第六阶段通过了CAB和SAB。这段代码非常清晰，易于阅读。


Datasets:

    - ISIC2017
    - ISIC2018
    - PH^2，from external validation

  The Paper, published in 2024.3.29: [UltraLight VM-UNet:Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation](https://arxiv.org/pdf/2403.20035)

   The official repository: [Here](https://github.com/wurenkai/UltraLight-VM-UNet)

<img src="https://github.com/BaoBao0926/Paper_reproducing/blob/main/Code/011.UltraLight%20VM-UNet/UltraLight%20VM-UNet.png" alt="Model" style="width: 00px; height: auto;"/>

</details>








