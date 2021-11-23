# 资料
### 基础学习
+ Bengio所著的《Deep Learning》
[原文官网](https://www.deeplearningbook.org/)
[原文翻译](https://github.com/exacity/deeplearningbook-chinese)
		
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

+ 附加资料

	数学之美-吴军


### 前沿
前沿论文：
	知乎博主总结：https://zhuanlan.zhihu.com/p/65177442

模型：
[deep_sort](https://github.com/ZQPei/deep_sort_pytorch), 
[mm](https://github.com/open-mmlab/mmtracking)

数据集：
+ Kaggle预测房价：[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
    
    
    
    
    
    
    
# 周报
### week1 DONE：

1. 学习完了吴恩达网课的第一部分：神经网络和深度学习
2. 手推了以m个样本同时传播的全连接神经网络的前向传播，和后项求导的向量化公式，了解了向量化后的网路每个向量的每一纬度的意义！
3. python 实现了一个可用性很强的全链接网络的类，设计之初，它可以自定义每一层的宏参数，包括（项目见week1目录）：
    + 每层神经元数及他们使用的激活函数（relu，leakyrelu，sigmoid）
    + 神经层的层数
    + 损失函数：ems，l(a, y) = -(y*log(a) + (1-y)*log(1-a))
    + 传播时的样本数
    项目状态：项目基本写完，主要问题在于不熟悉python的各种容器，数据在容器间的跳转产生了bug，在调试中
    
    map：
    映射函数访问不到self，在类中，self不能作为参数
    映射序列只能是行向量，并且numpy的形状变换用flatten，不要用reshape    



### PLANE:
+ week2主题：感性的认识神经网络，学习优化的方法并优化week1的网络，到kaggle上跑结果
    1. 学习下一个课程：改善深层次神经网络
    2. 看书：deep learning
    3. 复习学过的课程：《机器学习》里各种对模型评估的方法
    4. 用kaggle上的多个数据集（如房价预测数据集）跑设计完成的全链接网络模型，尝试各种优化的方法
    
+ week3主题：学习网络模型和视觉特征
    1. 主要学习，吴恩达的课程中：《卷积神经网络》和《序列模型》
    2. 复习《计算机视觉》课程中，提取图形特征的方法，以及相关一些算法
    3. 手撕算法
    
+ week4主题：开始读前沿论文