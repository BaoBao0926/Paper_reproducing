## Hi, I am [Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io)

---

[English](https://github.com/BaoBao0926/Overview-of-Reproduced-Project) | [简体中文](https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/README_chinese.md)

---


This depository is to give an overview for the projects reproduced by me and also I want to show my thoughts to these projects and papers. My code often has very detailed comments(some because I am also a new comer so that I need to make detailed comments to let myself understand).


  <!--    -----------------------------------------1.CapsNet -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">1. Capsule Network</b> 2023/11
   </summary>   
   
   <br />
   
  The idea of Capsule network is very novel and interesting

  1.Change commonly used scalars (this paper think the matrixes normally used in CNN are all scalar, but sometimes we may think these are vectors or matrixs) into vectors and hence proposing a algorithm, Dynamic Routing. In my opinion, the Dynamic routing is powerful for feature extraction, at least it gives a new idea to extract features. 

  2.It keeps using a idea of capsules.

  But training CapsNet is costly. Additionaly, compared with nowadays model, CapsNet shows its inability to more general and complex datasets. It is very hard to deal with complex datasets.

  I refer this [repository](https://github.com/gram-ai/capsule-networks) to write the code

  Paper: [Dynamic Routing Between Capsules](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html)
   
</details>


  <!--    -----------------------------------------2. U-Net   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">2.U-Net </b> 2024/4/4
   </summary>   
   
   <br />
   
  U-Net is used in segmentation task. The architecture is relatively simple, therefore suitable for new begineers to start learning how to deal with segmentation task. 

  It is used in medical field at first. I see a explanation that because the structure of medical images is constraint, relatively shallower model may work better.

   Paper: [U-Net-Based medical image segmentation](https://ncbi.longhoe.net/pmc/articles/PMC9033381/)
</details>


  <!--    -----------------------------------------  3.Learning without forgetting   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">3.Learning without forgetting </b>2024/4/18
   </summary>   
   
   <br />
   
  Learning withou forgetting (LwF) is used to deal with continual learning task in classification task. Some papers regard this paper as the first paper to systematically define continual learning (CL). In my opinion, it indead gives a lots of insights to CL. 

As to its metholodogy, it can be regared as the most simple way to use Knowledge Distillation (KD) into CL area. This project is very suitable for new begineers who want to learn continual learning using KD.

Additionally, the way of its CL is continually learn one class in one dataset. Taking CUB-200 dataset as example, it will learn one category on one time. Normally, we may think learn all categories of one dataset on one time.

I give very detailed comments in this project. I referred to this [project](https://github.com/ngailapdi/LWF). But the implementation way is different. I am not sure which one is better. But I think my code is very clear.

Paper: [Learning without Forgetting](https://ieeexplore.ieee.org/abstract/document/8107520)

Original Repository: [here](https://github.com/lizhitwo/LearningWithoutForgetting)
</details>



  <!--    ----------------------------------------- 4.Transformer   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">4.Transformer </b> 2024/4/25
   </summary>   
   
   <br />
   
There are a lots of paper and repostories to expain it. I also need learn these insights.

The reason why I learn this is that in 2021 transformer is used in Computer Vision(Vision Transformer ViT). Therefore, I learned Transformer, which should be used in NLP.

I learn Transformer by this [blog](https://blog.csdn.net/benzhujie1245com/article/details/117173090?spm=1001.2014.3001.5506), offering very detailed explanation.

I refer this [repository](https://github.com/datawhalechina/dive-into-cv-pytorch) 's code to write my code. I give many detailed explanation and I re-constructure the code skeleton so that it is easier for new comer(also for myself) to learn, and then can understand what source code is doing.

Paper: [Attention is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

</details>


  <!--    ----------------------------------------- 5.Vision Transformer   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">5.Vision Transformer </b>  2024/5/5
   </summary>   
   
   <br />
   
In 2021, a team used almost unchanged Transformer used in image classification, which give people an idea that Transformer orinigal used in NLP can also be used in Computer Vision. This is a huge improvement in Vision field. Many records have been broken by Transofrmer-based model. It prove transformer can be used in CV and if at scale, Transformer can even performer better. Based on this work, a lot of work has been born.

If you can write the code of Transformer, Vision Transformer(ViT) is also easy for you because there is not decoder. 

I learn ViT through this [bilibili vedio](https://www.bilibili.com/video/BV15P4y137jb?vd_source=80b346be9e1c1a93109688bf064e5be1) and this [one](https://www.bilibili.com/video/BV1Uu411o7oY?p=2&vd_source=80b346be9e1c1a93109688bf064e5be1), this [blog](https://blog.csdn.net/qq_51957239/article/details/132912677?spm=1001.2014.3001.5506).

Writing code refer to this [bilibili vedio](https://www.bilibili.com/video/BV1Uu411o7oY?p=2&vd_source=80b346be9e1c1a93109688bf064e5be1) and this [repository](https://github.com/lucidrains/vit-pytorch) and the [authrity repository](https://github.com/google-research/vision_transformer)

Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

</details>


   <!--    ----------------------------------------- 6.Swin Transformer   -------------------------------------------------------  -->
<details> 
   <summary>
   <b style="font-size: larger;">6.Swin Transformer</b> 2024/5/9
   </summary>   
   
   <br />
   


The source I refer: a bilibili [vedio](https://www.bilibili.com/video/BV13L4y1475U?vd_source=80b346be9e1c1a93109688bf064e5be1) to explain paper, 
a bilibili [vedio](https://www.bilibili.com/video/BV1zT4y197Fe?p=2&vd_source=80b346be9e1c1a93109688bf064e5be1) to explain to code, a CSDN [blog](https://blog.csdn.net/qq_45848817/article/details/127105956?ops_request_misc=&request_id=&biz_id=102&utm_term=Swim%20transformer%E4%BB%8B%E7%BB%8D&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-127105956.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187) to explain the Swim Transformer,
a CSDN [blog](https://blog.csdn.net/beginner1207/article/details/138034012?ops_request_misc=&request_id=&biz_id=102&utm_term=Droppath&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-138034012.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187) to introduce Dropath(it is my first time to see this),

Original paper: [Swin transformer: Hierarchical vision transformer using shifted windows](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper)

Official repository: [here](https://github.com/microsoft/Swin-Transformer)

</details>




