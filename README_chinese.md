## Hi, I am [Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io)

---

[English](https://github.com/BaoBao0926/Overview-of-Reproduced-Project/tree/main) | [简体中文](https://github.com/BaoBao0926/BaoBao0926.github.io/blob/main/README_CHINESE.md)

---

这个仓库是一个我复现过的文章的overview，我想把我复现出来的代码展示出来，并且给出一些我对这些项目或者paper简单的思考和建议。

1.Capsule Network
1.CapsNet将常用的标量(这篇paper认为CNN中通常使用的矩阵都是标量，但有时我们可能会说这些是向量或矩阵)转换为vector，并提出了一种算法Dynamic Routing。在我看来，动态路由对于特征提取来说是非常强大的，至少它给特征提取提供了一个新的思路。

2.它使用了胶囊的概念。

但是训练CapsNet是昂贵的。此外，与现有模型相比，CapsNet在更一般和更复杂的数据集上表现出了不足。

2.U-Net
U-Net用于segmentation任务。该架构相对简单，因此适合初学者开始学习如何处理分段任务。

3.Learning without forgetting
无遗忘学习(LwF)用于处理classification任务中的continual learniing任务。一些论文认为这篇文章是第一个系统定义continual learning(CL)的论文。在我看来，它确实给了CL很多启示。

就其方法而言，我认为是将知识精馏(KD)应用于CL领域的最简单的方法。这个项目非常适合想要使用KD学习持续学习的新初学者。

此外，它的学习方法是在一个数据集中不断地学习一个类。以CUB-200数据集为例，它将一次学习一个类别。通常，我们可能会认为一次学习一个数据集的所有类别。

我在这个项目的复现代码中给出很详细的注释.
