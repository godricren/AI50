## Machine Learning 机器学习相关概念
`Machine` 在这里指的是平台，系统，代码等，不是计算机。  
`Learning` 代码在经历了某些过程以后，性能会得到提升。这个过程就叫做学习  

### 机器学习的项目流程
1. 从宏观角度分析问题，搞定输入什么，输出什么；比如：翻译器输入中文，输出英文。放假预测，输出房子的相关信息，输出价格。人脸检测，输入人脸图片，输出人脸
2. 按照数据和输出构建数据集；
   * 一行一个样本，一列一个特征，前面放特征，最后放样本
   * 特征就是输入，标签就是输出
3. ***找一个***机器学习算法，完成从输入到输出的***映射***
   * 选择一种算法
   * 把数据给算法学习
   * 完成模型的训练
   * 对模型进行评估
4. 部署应用
5. 模型迭代升级
### algorithm & model
* algorithm 算法，计算机执行一个任务时，具体的步骤
* model 模型，算法的代码实现
### 传统算法（rule -based algorithm） vs 人工智能算法(data based algorithm)
* rule based algorithm 
* data based algorithm
## K-Nearest Neighbors
```python

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib


X,y = load_iris(return_X_y=True)
X_train, y_train, X_test, y_text = train_test_split(X,y,
                                                    train_size=0.2,
                                                    random_state=42,
                                                    shuffle=True)
clf = KNeighborsClassifier()
clf.fit(X=X_train,y=y_train)
y_pred = clf.predict(X=X_test)

joblib.dump(value=clf,filename="clf.aura")
```
### KNN算法原理
* 算法：计算机解决问题的步骤
* KNN K-Nearest Neighbors K个最近的邻居,KNN是惰性计算 规则+数据  
* 核心理念：鸟随鸾凤飞腾远，人伴贤良品德高  
推理流程：（如何给一朵花分类）
1.  找出这朵花最近的K个邻居
    * 距离的度量：欧式空间 - 距离的计算、向量视角 - 余弦相似度
2. K个最近的邻居进行投票，选出类别出现次数最多的类

## 概率与数理统计
概率与数理统计主要是辅助建模的，比较典型的有`高斯贝叶斯`
### 概率
概率`probability`是事件的固有属性，用频率`frequency`来代替，当实验次数越多时，频率就越趋近于概率  
在实际项目中，数据量很大，概率就直接用频率替代即可
#### 概率的特性
* 非负性
* 规范性
#### 概率的计算
##### 离散型变量
在有限个状态中选择一个，**注意：有限个不一定很少**  
对于离散变量的编码常见两种：
* zero index
* one hot

##### 连续型变量
有无数个采样结果，比如，温度，严格意义上来说，对于连续量上的某个单点的概率都是0    
**概率密度函数**    
`probability density funcation` PDF 简单来说概率密度函数，是概率的导函数，也就是说，概率是概率密度函数的积分，比如正态分布

在实际应用的模型中，我们主要是来比较概率的大小，并不在意概率具体的值具体是多少，所以直接使用概率密度函数的值来代替就可以了。这样就可以求连续型变量，某个点的概率值了。

**条件概率**

P(A) A发生的概率

P(A|B) 在B发生的条件下A发生的概率

应对条件概率的法宝就是重新划分样本空间，也就是说把不满足条件的样本删除，再重新计算概率
$$
P(A|B)=\frac{P(AB)}{P(B)}
$$
 变换得到贝叶斯公式
$$
P(A|B)\frac{P(B|A)P(A)}{P(B)}
$$


或
$$
P(y_i|X)=\frac{P(X|y_i)P(y_i)}{P(X)}
$$


$X$ 是一个输入样本的特征 $[x_1,x_2,...,x_n]$

$y_i$ 第$i$类
$$
P(y_i|X)=P(X|y_i)P(y_i)
$$
$P(X|y_i)$就是概率密度函数的取值，把其他的类型都踢出去，

$P(y_i)$ 就是第$i$类的取值 先验概率，是常量



高斯贝叶斯

连续型变量

**基本统计量**

* 均值
* 标准差
  * 求均值
  * 求每个数和均值的差
  * 把差取平方
  * 再把平方取均值
  * 再把结果取平方根

协方差

两列数据的变化趋势是否相同

协方差的计算
$$
E((X-E(X))\times (Y-E(Y))
$$


皮尔逊相关系数

Person相关系数，协方差归一化，除以他们的标准差，从-1，1，-1严格负相关，1严格正相关，0，不相关

## 信息论和决策树算法

### 1. 信息量的大小

* 小概率事件发生了，信息量很大
* 大概率事件发生了，信息量很小

### 2. entropy 信息熵

熵是一个系统的混乱程度，熵越大，系统越混乱，反之亦然
$$
H(X) = -\sum_{i=1}^{n}p(x_i)logp(x_i)
$$
**基尼系数**
$$
gini=\sum_{i=1}^{n}p\times(1-p)
$$
基尼系数就是信息熵的工程化简，内涵一样，但是计算代价要低很多



