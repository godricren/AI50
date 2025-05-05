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
核心理念：鸟随鸾凤飞腾远，人伴贤良品德高  
推理流程：
1. 

