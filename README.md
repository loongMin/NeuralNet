## 基础学习
+ Bengio所著的《Deep Learning》
[原文官网](https://www.deeplearningbook.org/)
[原文翻译](https://github.com/exacity/deeplearningbook-chinese)
+ 数学之美-吴军
+ andrew的课程 
[网课](https://mooc.study.163.com/university/deeplearning_ai#/c)
[习题和答案讲解](https://blog.csdn.net/weixin_36815313/article/details/105728919)
	1. 神经网络和深度学习
	2. 改善深层次神经网络
	3. 卷积神经网络
	4. 序列模型
	
+ 李宏毅-2021 
[课程](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)
[资料](https://app6ca5octe2206.pc.xiaoe-tech.com/detail/p_6049e1c6e4b05a6195befd56/6)
	

+ 论坛与社区

	[Techbeat](https://www.techbeat.net/)
	
	[ZihaoZhao](https://www.zhihu.com/column/c_1102212337087401984)

	


## Papers
知乎博主总结：https://zhuanlan.zhihu.com/p/65177442
   
[arxiv](https://arxiv.org/list/cs/recent)
[nips](https://papers.nips.cc/paper/)
### Convolution Net - object recognize
YOLO

AlexNet

VGG-16

LeNet
    
Region proposal net, PRN
    R-CNN
 
  
   
### Face Recognize
FaceNet

DeepFace


### Object tracking



### Style transform




## Others

模型：
[deep_sort](https://github.com/ZQPei/deep_sort_pytorch), 
[mm](https://github.com/open-mmlab/mmtracking)



    




----------------------------------------------------------------------------------------
## 资料索引：
ROC曲线在二十世纪八十年代后期被引入机器学习[Spackman, 1989]
AUC则是从九十年代中期起在机器学习领域广为使用[Bradley,1997].
[Hand and Till,2001]将ROC曲线从二分类任务推广到多分类任务.
[Fawcett,2006]综述了ROC曲线的用途.

[Drummond and Holte,2006]发明了代价曲线.
代价敏感学习[Elkan,2001;Zhou and Liu,2006]专门研究非均等代价下的学习。

自助采样法,[Efron and Tibshirani, 1993]

[Dietterich,1998]指出了常规k折交叉验证法存在的风险,并提出了5*2折交叉验证法.
[Demsar, 2006]讨论了对多个算法进行比较检验的方法.

[Geman et al.,1992]针对回归任务给出了偏差-方差-协方差分解，后来被简称为偏差-方差分解。
但仅基于均方误差的回归任务中推导，对分类任务，由于0/1损失函数的跳变性,理论上推导出偏差-方差分解很困难。
已有多种方法可通过试验队偏差和方差进行估计[Kong and Dietterich,1995;Kohavi and Wolpert,1996; Breiman,1996;Friedman,1997;Domingos,2000].