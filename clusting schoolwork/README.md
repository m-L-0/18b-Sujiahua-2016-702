

**导入数据**

```
from sklearn.preprocessing import normalize      
from sklearn.datasets import load_iris
import networkx as nx
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
iris = load_iris()
Matrix=np.array(iris.data)
A=np.zeros((150,150))
```

**鸢尾花数据集画成图的形式**

选取sepal width和sepal length这两个特征将鸢尾花的数据表示出来

![](F:\机器学习\实训\聚类\1.png)

![](F:\机器学习\实训\聚类\iris1.png)

**确定一个合适的阈值**   求取鸢尾花数据集的相似度，相似度>0.95的 两个样本之间添加一条黑色实体边 相似度小于0.9的红色虚边 其他的白色的虚边

**求取邻接矩阵**  

```

Matrix=np.array(iris.data)
A=np.zeros((150,150))
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        A[i,j]=(np.exp(-0.5*np.dot((Matrix[i]-Matrix[j]),(Matrix[i]-Matrix[j]))))
```

**计算距离**

```
def cal_dis_mat(Matrix, data):
        # 计算距离平方的矩阵
    dis_mat = np.zeros((150, 150))  # 生成一个n×n的零矩阵
    for i in range(150):
        for j in range(i + 1, 150):  # 距离矩阵是对称阵，所以没必要j也从0循环到n-1
            dis_mat[i, j] = (Matrix[i] - Matrix[j]).dot((Matrix[i] - Matrix[j]))  # 计算两个向量之间距离的平方赋值给dis_mat
            dis_mat[j, i] = dis_mat[i, j]  # 距离矩阵是对称阵
    return dis_mat
```

**求取度矩阵 拉普拉斯矩阵**

```
Du = np.diag(A.sum(axis=1))  # 计算度数矩阵
La= Du - A  # 计算拉普拉斯矩阵
L=normalize(La, axis=0, norm='max')  # 计算归一化的对称拉普拉斯矩阵
```

**进行聚类**

```
# 初始化质心
def init_centre(data,k):
    centre_idx = np.random.choice(data.shape[0], size=k)#随机选择K个质心
    centres = [data[i] for i in centre_idx]
    return np.array(centres)
# 根据质心划分簇
def split_cluster(data, centres):
    clusters = [[] for i in range(centres.shape[0])]
    for i in range(data.shape[0]):
        dist = np.square(data[i]-centres).sum(axis=1)
        idx = np.argmin(dist)
        clusters[idx].append(i)
    return np.array(clusters)
# 更新质心
def update_centre(clusters, data):
    n_features = data.shape[1]
    k = clusters.shape[0]
    centres = np.zeros((k,n_features))
    for i, cluster in enumerate(clusters):
        centre = np.mean(data[cluster],axis=0)
        centres[i] = centre
    return centres
```

**聚类结果可视化**

![](F:\机器学习\实训\聚类\iris2.png)

这个图像看着比较糟心，:(

**调整参数**

调整K-means的参数 调整最大迭代次数和收敛值

把最大迭代次数调整到了2500，收敛值调成了0.001

**分簇正确率(其中一次)**

第一种花分类正确的有50个

第二种花分类正确的有46个

第三种花分类正确的有38个

正确率134/150

87.333333%

因为一些类别相同的花，距离簇心有一点远，还有一些不同类别的花，距离这个簇心比较近，所以结果就会出现聚类效果不太好。

苏家华

2016011702