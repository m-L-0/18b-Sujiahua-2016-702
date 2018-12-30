from sklearn.preprocessing import normalize      
from sklearn.datasets import load_iris
import networkx as nx
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

iris = load_iris()
location = pd.read_csv("./location.txt",sep=" ",header=None)
#setosa_sepal_len = iris.data[:50, 0]
#setosa_sepal_width = iris.data[:50, 1]
#
#versi_sepal_len = iris.data[50:100, 0]
#versi_sepal_width = iris.data[50:100, 1]
#
#vergi_sepal_len = iris.data[100:, 0]
#vergi_sepal_width = iris.data[100:, 1]
#
#pyplot.scatter(setosa_sepal_len, setosa_sepal_width, marker = 'o', c = 'b',  s = 30, label = 'Setosa')
#pyplot.scatter(versi_sepal_len, versi_sepal_width, marker = 'o', c = 'r',  s = 50, label = 'Versicolour')
#pyplot.scatter(vergi_sepal_len, vergi_sepal_width, marker = 'o', c = 'y',  s = 35, label = 'Virginica')
#pyplot.xlabel("sepal length")
#pyplot.ylabel("sepal width")
#pyplot.title("sepal length and width scatter")
#pyplot.legend(loc = "upper right")                                                                                                                                                
data=np.array(iris.data)
Matrix=[]
Matrix=np.array(iris.data)
A=np.zeros((150,150))
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        A[i,j]=(np.exp(-0.5*np.dot((Matrix[i]-Matrix[j]),(Matrix[i]-Matrix[j]))))
#计算距离
dis_mat = np.zeros((150, 150))  # 生成一个n×n的零矩阵
for i in range(150):
    for j in range(i + 1, 150):  # 距离矩阵是对称阵，所以没必要j也从0循环到n-1
        dis_mat[i, j] = A[i,j]#((Matrix[i] - Matrix[j]).dot((Matrix[i] - Matrix[j])))  # 计算两个向量之间距离的平方赋值给dis_mat
        dis_mat[j, i] = dis_mat[i, j]  # 距离矩阵是对称阵

def q_neighbors(A,q=10):
    n = []
    for i in range(len(A)):
        inds = np.argsort(A)#进行排序
        inds = inds[-16:-1]#选择q个
        n.append(inds)
    return np.array(n)

qnn = q_neighbors(A)
plt.figure(figsize=(8,8))
G = nx.Graph() 
pos = np.array(location)
# 向图G添加节点和边
G.add_nodes_from([i for i in range(150)])
for i in range(150):
    for j in range(150):
        # i和J互为近邻，二者之间才有边
        if(i in qnn[j] and j in qnn[i]):
            G.add_edge(i,j,weight=A[i,j])
# 画出节点           
nx.draw_networkx_nodes(G, pos, node_color='black', node_size=40, node_shape='o')
# 将图G中的边按照权重分组
#lin=[]
#lin=A
edges_list1=[]
edges_list2=[]
edges_list3=[]
for (u,v,d) in G.edges(data='weight'):
    if d > 0.98:
        edges_list1.append((u,v))
    elif d< 0.9:
        edges_list2.append((u,v))
    else:
        edges_list3.append((u,v))
# 按照分好的组，以不同样式画出边
nx.draw_networkx_edges(G, pos, edgelist=edges_list1, width=1, alpha=0.7, edge_color='black', style='solid')
nx.draw_networkx_edges(G, pos, edgelist=edges_list2, width=1, alpha=0.7, edge_color='red', style='dashed')
nx.draw_networkx_edges(G, pos, edgelist=edges_list3, width=1, alpha=0.7, edge_color='white', style='dashed')
#plt.savefig("iris_graph.png")
plt.show() 

Xiang = None  # 图的相似性矩阵
La = None  # 图的拉普拉斯矩阵
Du = None  # 图的度矩阵
cluster = None  #簇的数量
Ding = None  # 图中顶点的数量或者数据集中样本的数量
centers = None 
Ding = data.shape[0]  # 获取数据集中有多少数据
Xiang=A  # 计算相似度矩阵
Du = np.diag(Xiang.sum(axis=1))  # 计算度数矩阵
La = Du - Xiang  # 计算拉普拉斯矩阵
     

L=normalize(La, axis=0, norm='max')  # 计算归一化的对称拉普拉斯矩阵


def spectral_clustering(X,k):
    w, v = np.linalg.eig(L)
    inds = np.argsort(w)[:k]
    Vectors = v[:, inds]
    normalizer = np.linalg.norm(Vectors, axis=1)  # 规一化
    normalizer = np.repeat(np.transpose([normalizer]), k, axis=1)
    Vectors = Vectors / normalizer
    return Vectors
# 对谱聚类得出的数据进行K-means聚类
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
# 得到聚簇后的类别标签
def get_cluster_labels(clusters, data):
    y_pred = np.zeros(data.shape[0], dtype=np.int32)
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred
# 对整个数据集data进行Kmeans聚类，返回其聚类的标签
def k_means(data,k,max_iter,epsilon):
    # 从所有样本中随机选取k个样本作为初始的聚类中心
    centres = init_centre(data,k)  
    # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
    for _ in range(max_iter):
        # 将所有数据进行归类，归类规则就是将该样本归类到与其最近的中心
        clusters = split_cluster(data,centres)
        former_centres = centres
        # 计算新的聚类中心
        centres = update_centre(clusters, data)
        # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
        diff = centres - former_centres
        if diff.any() < epsilon:
            break
    return get_cluster_labels(clusters, data)
# 谱聚类以及K-means聚类结果
sp_data = spectral_clustering(data,3)
pred = k_means(sp_data,3,2500,0.001)

#聚类结果可视化 
plt.figure(figsize=(10,10))
N = nx.Graph()
# 给图N添加顶点和边
N.add_nodes_from([i for i in range(150)])

qnn = q_neighbors(A)
for i in range(150):
    for j in range(150):
        if(i in qnn[j] and j in qnn[i]):
            N.add_edge(i,j, weight=A[i,j])
#按簇标签将节点分组         
nodes_list1=[i for i in range(150) if pred[i] == 0]
nodes_list2=[i for i in range(150) if pred[i] == 1]
nodes_list3=[i for i in range(150) if pred[i] == 2]
# 按照权重将边分组
edges_list1=[]
edges_list2=[]
edges_list3=[]
for (u,v,d) in G.edges(data='weight'):
    if d > 0.95:
        edges_list1.append((u,v))
    elif d < 0.85:
        edges_list2.append((u,v))
    else:
        edges_list3.append((u,v))
# 画出不同节点
nx.draw_networkx_nodes(N, pos, node_size=30, nodelist=nodes_list1, node_shape='o')
nx.draw_networkx_nodes(N, pos, node_size=30, nodelist=nodes_list2, node_shape='^')
nx.draw_networkx_nodes(N, pos, node_size=30, nodelist=nodes_list3, node_shape='s')
# 画出边
nx.draw_networkx_edges(G, pos, edgelist=edges_list1, width=1, alpha=0.7, edge_color='black', style='solid')
nx.draw_networkx_edges(G, pos, edgelist=edges_list2, width=1, alpha=0.7, edge_color='white', style='dashed')
nx.draw_networkx_edges(G, pos, edgelist=edges_list3, width=1, alpha=0.7, edge_color='green', style='solid')
#plt.savefig("sp_iris_graph.png")
plt.show()
iris_label = pd.read_csv("./iris.csv",header=None)
label = np.array(iris_label[5])
cnt = 0
for i in range(150):
    if label[i] == pred[i]:
        cnt += 1
acc = cnt / 150
print(acc)