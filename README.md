## Hi, I am Muyi Bao

---

[English](https://github.com/BaoBao0926/Overview-of-Reproduced-Project) | [简体中文](https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/README_chinese.md)

---


This depository is to give an overview for the projects reproduced by me and also I want to show my thoughts to these projects and papers.

### 1.Capsule Network

The idea of Capsule network is very novel. 

1.Change commonly used scalars (this paper think the matrixes normally used in CNN are all scalar, but sometimes we may think these are vectors or matrixs) into vectors and hence proposing a algorithm, Dynamic Routing. In my opinion, the Dynamic routing is powerful for feature extraction, at least it gives a new idea to extract features. 

2.It keeps using a idea of capsules.

But training CapsNet is costly. Additionaly, compared with nowadays model, CapsNet shows its inability to more general and complex datasets.

### 2.U-Net

U-Net is used in segmentation task. The architecture is relatively simple, therefore suitable for new begineers to start learning how to deal with segmentation task.

### 3.Learning without forgetting

Learning withou forgetting (LwF) is used to deal with continual learning task in classification task. Some papers regard this paper as the first paper to systematically define continual learning (CL). In my opinion, it indead gives a lots of insights to CL. 

As to its metholodogy, it can be regared as the most simple way to use Knowledge Distillation (KD) into CL area. This project is very suitable for new begineers who want to learn continual learning using KD.

Additionally, the way of its CL is continually learn one class in one dataset. Taking CUB-200 dataset as example, it will learn one category on one time. Normally, we may think learn all categories of one dataset on one time.

I give very detailed comments in this project.
