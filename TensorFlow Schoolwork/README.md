

**K近邻算法**

输入：样本集 D={(x1,y1),(x2,y2)...,(xm,ym)}D={(x1,y1),(x2,y2)...,(xm,ym)}; 
   聚类簇数 kk. 
输出：簇划分C={C1,C2,...,Ck}C={C1,C2,...,Ck}
对未知类别属性的数据集中的每个点依次执行以下操作： 
(1) 计算已知类别数据集中的点与当前点之间的距离；

(2) 按照距离递增次序排序；  

(3) 选取与当前点距离最小的k个点；  

(4) 确定前k个点所在类别的出现频率； 

 (5) 返回前k点出现频率最高的类别作为当前的预测分类。 



**1划分数据集和测试集**



```
iris =datasets.load_iris()
 # 洗乱数据
indices = np.arange(len(iris.data))
np.random.shuffle(indices)
# 分割训练数据集和测试数据集
split_index = int(len(iris.data) * 0.2)
```

**2设计模型**

```
def knn(tr_value, tr_target, te_value, te_target, k):
    hit = 0
    for i in range(len(te_value)):
        # 利用numpy提供的大矩阵运算简洁地计算当前数据与所有测试数据之间的欧氏距离
        one2n = np.tile(te_value[i], (int(len(tr_value)), 1))
        distance = (((tr_value - one2n) ** 2).sum(axis=1)) ** 0.5
 
        count = {}
        # 根据距离从小到大排列训练数据的下标
        sorted_distance = distance.argsort()
        # 统计前k个训练数据中哪个标签出现次数最多
        # print(sorted_distance)
        for j in range(k):
            # print(distance[sorted_distance[j]])
            tmp_tag = tr_target[sorted_distance[j]]
            # print(tmp_tag)
            if tmp_tag in count.keys():
                count[tmp_tag] += 1
            else:
                count[tmp_tag] = 1
 
        # 排序后选取出现次数最多的标签作为当前测试数据的预测结果
        tag_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
        print(te_target[i], tag_count[0][0])
```

**验证模型**

使用验证集检测模型性能

![](F:\机器学习\实训\TensorFlow Schoolwork\image\acc.png)

使用验证集调整超参数

```
k_range = range(1,30)
k_scores = []
for k in k_range:
    print(k)
    clf = XGBClassifier(learning_rate= 0.28, min_child_weight=0.7, max_depth=21,
                        gamma=0.2, n_estimators = k ,seed=1000)

    X_Train = tr_value
    Y_Train = tr_target

    X, y = (X_Train, Y_Train)
    
    #交叉验证，循环跑，cv是每次循的次数
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    #得到平均值
    k_scores.append(scores.mean())

print(k_scores)
#画出图像
plt.plot(k_range, k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross validated accuracy")
plt.show()
```

![](F:\机器学习\实训\TensorFlow Schoolwork\image\pingjia.png)

```
姓名：苏家华
学号：2016011702
```

