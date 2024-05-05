## Hi, I am [Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io)

---

[English](https://github.com/BaoBao0926/Overview-of-Reproduced-Project/tree/main) | [简体中文](https://github.com/BaoBao0926/BaoBao0926.github.io/blob/main/README_CHINESE.md)

---



这个仓库是一个我复现过的文章的overview，我想把我复现出来的代码展示出来，并且给出一些我对这些项目或者paper简单的思考。并且由于很多时候我也是一个新手，所以我写了非常详细的注释帮助自己理解，我想这也许对一些新手同样有用。


  <!--    -----------------------------------------1.CapsNet -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">1. Capsule Network</b>
   </summary>   
   
   <br />
   
  1.CapsNet将常用的标量(这篇paper认为CNN中通常使用的矩阵都是标量，但有时我们可能会说这些是向量或矩阵)转换为vector，并提出了一种算法Dynamic Routing。在我看来，动态路由对于特征提取来说是非常强大的，至少它给特征提取提供了一个新的思路。

2.它使用了胶囊的概念。

但是训练CapsNet是昂贵的。此外，与现有模型相比，CapsNet在更一般和更复杂的数据集上表现出了不足，难以处理复杂的数据集

  I refer this [repository](https://github.com/gram-ai/capsule-networks) to write the code

  Paper: [Dynamic Routing Between Capsules](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html)
   
</details>


  <!--    -----------------------------------------2. U-Net   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">2.U-Net</b>
   </summary>   
   
   <br />
   
 U-Net用于segmentation任务。该架构相对简单，因此适合初学者开始学习如何处理分段任务。这个model被用于处理医学图片最刚开始的时候，我看过一些解释，由于医学图片（比如CT照片）的结构都是限制的，大体样子都是一样的，所以比较浅层的模型的效果可能更好


   Paper: [U-Net-Based medical image segmentation](https://ncbi.longhoe.net/pmc/articles/PMC9033381/)
</details>


  <!--    -----------------------------------------  3.Learning without forgetting   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">3.Learning without forgetting</b>
   </summary>   
   
   <br />
   

无遗忘学习(LwF)用于处理classification任务中的continual learniing任务。一些论文认为这篇文章是第一个系统定义continual learning(CL)的论文。在我看来，它确实给了CL很多启示。

就其方法而言，我认为是将知识精馏(Knowledge Distinallation KD)应用于CL领域的最简单的方法。这个项目非常适合想要使用KD学习持续学习的新初学者。

此外，它的学习方法是在一个数据集中不断地学习一个类。以CUB-200数据集为例，它将一次学习一个类别。通常，我们可能会认为一次学习一个数据集的所有类别。

我在这个项目的复现代码中给出很详细的注释. 我参考了这个[仓库](https://github.com/ngailapdi/LWF), 我的代码结构与它的并不太一样，可能他的代码更高效，但我认为我的代码更容易理解，我加了许多注释


Paper: [Learning without Forgetting](https://ieeexplore.ieee.org/abstract/document/8107520)

Original Repository: [here](https://github.com/lizhitwo/LearningWithoutForgetting)

</details>



  <!--    ----------------------------------------- 4.Transformer   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">4.Transformer</b>
   </summary>   
   
   <br />
   
There are a lots of paper and repostories to expain it. I also need learn these insights.

The reason why I learn this is that in 2021 transformer is used in Computer Vision(Vision Transformer ViT). Therefore, I learned Transformer, which should be used in NLP.

I learn Transformer by this [blog](https://blog.csdn.net/benzhujie1245com/article/details/117173090?spm=1001.2014.3001.5506), offering very detailed explanation.

I refer this [repository](https://github.com/datawhalechina/dive-into-cv-pytorch) 's code to write my code. I give many detailed explanation and I re-constructure the code skeleton so that it is easier for new comer(also for myself) to learn, and then can understand what source code is doing.

Paper: [Attention is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

</details>


  <!--    ----------------------------------------- 4.Vision Transformer   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">5.Vision Transformer</b>
   </summary>   
   
   <br />
   
In 2021, a team used almost unchanged Transformer used in image classification, which give people an idea that Transformer orinigal used in NLP can also be used in Computer Vision. This is a huge improvement in Vision field. Many records have been broken by Transofrmer-based model. It prove transformer can be used in CV and if at scale, Transformer can even performer better. Based on this work, a lot of work has been born.

If you can write the code of Transformer, Vision Transformer(ViT) is also easy for you because there is not decoder. 

I learn ViT through this [bilibili vedio](https://www.bilibili.com/video/BV15P4y137jb?vd_source=80b346be9e1c1a93109688bf064e5be1) and this [one](https://www.bilibili.com/video/BV1Uu411o7oY?p=2&vd_source=80b346be9e1c1a93109688bf064e5be1), this [blog](https://blog.csdn.net/qq_51957239/article/details/132912677?spm=1001.2014.3001.5506).

Writing code refer to this [bilibili vedio](https://www.bilibili.com/video/BV1Uu411o7oY?p=2&vd_source=80b346be9e1c1a93109688bf064e5be1) and this [repository](https://github.com/lucidrains/vit-pytorch) and the [authrity repository](https://github.com/google-research/vision_transformer)

Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

</details>
