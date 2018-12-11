# 高光谱数据分类

**导入需要的模块**

```
import matplotlib.pyplot as plt  
import numpy as np
import scipy.io as sio
import spectral
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import re
```

**导入训练集**

```
a=[]
path="./train/"
for i in os.listdir(path):
    print(i)
    A=sio.loadmat(path+i)[i.split('.')[0]]
    a.append(A)
```

**划分训练集、测试集**

```
X_train,X_valid,y_train,y_valid = train_test_split(data,label,test_size=0.25,shuffle=True)
```

**构建分类器**

```
svc = SVC(C=10,gamma=0.01,class_weight='balanced')  #构建分类器，设定参数
svc.fit(X_train,y_train)
y_pred = svc.predict(X_valid)
```

**使用GridSearchCV选择参数**

```
svr = svm.SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4, 8], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
clf = GridSearchCV(svr, parameters, scoring='f1')
clf.fit(X_train, y_train)
print('The parameters of the best model are: ')
print(clf.best_params_)
```

**导入要进行验证的数据**

```
test_data = sio.loadmat("./data_test_final.mat")['data_test_final']
test_data = np.array(test_data, dtype=np.float64)
x_test = scaler.transform(test_data)
y_test = svc.predict(x_test)
```

**输出预测结果**

```
data = pd.DataFrame(y_test)
data.to_csv("./test_labels.csv")
```

```
苏家华
2016011702
```

