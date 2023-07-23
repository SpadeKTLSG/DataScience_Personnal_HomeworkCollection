# -*- coding: utf-8 -*-
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch  # 层次聚类

from scipy.spatial.distance import pdist, squareform  # pair-wise distance样本距离计算，pdist返回向量形式，squareform返回矩阵形式
from sklearn import preprocessing, neighbors, cluster, metrics, manifold, decomposition
import warnings

warnings.filterwarnings('ignore')  # 不显示warning信息
np.set_printoptions(precision=5, suppress=True)  # 设置浮点数显示位数，尽量不用科学计数法表示

# 读取2016_world_happiness.csv文件，并展示前五个样本
wh = pd.read_csv('2016_world_happiness.csv')
# print(wh.shape)
# print(wh.head())
# print(wh.columns)

# 开始数据分析
# 使用相关性分析和描述性统计等相关方法进行数据探索

# !画出关键因素的相关矩阵
wh_key = wh.drop(['Country', 'Region', 'Happiness Rank', 'Happiness Score',
                  'Lower Confidence Interval', 'Upper Confidence Interval', 'Dystopia Residual'], axis=1)
corrmat = wh_key.corr()  # 得到相关性矩阵
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corrmat, square=True)  # 画出热力图
sns.pairplot(wh_key, kind='reg', diag_kind='kde', diag_kws=dict(shade=True), size=4)  # 画出散点图
# plt.show()
# 从上面的两个图可以看出，Family与Economy (GDP per Capita)呈正相关关系(另外还有其他的正相关或负相关关系)
# 由于关键因素的取值代表了在对Happiness Score做预测时的系数，所以只能代表两者对Happiness Score的权重一致，并不能说明两者的相关关系，
# 所以对于该数据集，做相关分析是没有任何意义的，除非拿到原始的关键因素的取值，而不是拟合之后的权重


#  !各个区域的幸福指数差异
# print(wh.iloc[wh['Country'] == 'China'].values)
# print(wh.iloc[wh['Country'] == 'Japan'].values)
#  中国的幸福指数排名处于中等，日本的幸福指数排名处于中等偏上的水平，两者相差30名。
# 
f, ax = plt.subplots(figsize=(8, 8))
ax = sns.swarmplot(y="Region", x="Happiness Score", data=wh)
ax = sns.boxplot(y="Region", x="Happiness Score", data=wh)
# plt.show()

# 从平均幸福指数来看，撒哈拉以南的非洲地区的幸福程度最低，南亚地区的幸福程度排在倒数第二，西欧及北美这两个以资本主义国家为主的地区幸福程度较高。
# 同时，Middle East and Northern Africa中东和北非地区的幸福指数差异最大，我们猜测中东和北非这部分石油国与战乱国混合而成的区域
# ，由石油带来的大量财富与战争带来的国家分裂导致了国家之间幸福感的巨大差异。


# !不同区域关键因素对幸福指数的影响差异
# 由于各国的幸福指数是不同的，为了得出不同国家的每个关键因素对幸福指数的影响，我们应该将各个关键因素贡献值除以其幸福指数，得到关键因素贡献比例值。

for row in range(wh.shape[0]):  # 除以幸福指数
    wh_key.iloc[row] = wh_key.iloc[row] / wh['Happiness Score'].iloc[row]
reg_df = pd.DataFrame(wh['Region'], columns=['Region'])  # 将Series转换为DataFrame
wh_norm = pd.concat([reg_df, wh_key], axis=1)  # 将wh_key与reg_df拼接
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 30))  # 画出六张图
for col, ax in zip(wh_key.columns, axes.flatten()):
    sns.swarmplot(x=col, y='Region', data=wh_norm, ax=ax)
    sns.boxplot(x=col, y='Region', data=wh_norm, ax=ax)
#     
#     
# plt.show()

# 从上面六张图我们可以看出，在低收入区域(从该数据集无法得出收入状况，这里的“低收入”是一个经验判断)如Sub-Saharan Africa和Southern Asia区域，
# 国家GDP对幸福指数的贡献相较于其他区域小，这些地区的主要幸福来源为自由、慷慨换来的幸福感。
# 高收入地区如Western Europe西欧和North America北美的幸福感来自于经济状况、家庭、健康状况、自由等各个方面，似乎它们在各个方面较其他区域都相当满意。
# 有趣的是，Australia and New Zealand澳洲和新西兰地区对政府的各项决策非常信任，较其他地区，它们从政府的各项福利和政策中收获的幸福感最强

# !对样本进行聚类
# 使用Kmeans、MiniBatchKMeans、SpectralClustering和DBSCAN四种方法进行聚类
# 首先去掉wh_norm数据框中的'Region'列，并将DataFrame转换为ndarray：
wh_array = wh_norm.drop('Region', axis=1).values

names = ['KMeans', 'MiniBatchKMeans', 'SpectralClustering', 'DBSCAN']


# 定义聚类的函数
def clustering(n_clusters=5, eps=.04):  # DBSCAN 无 n_clusters参数，最终簇的数目由 eps 决定
    labels = {}  # 存储聚类结果的类别标签
    centers = {}  # 存储各类中心点位置的数据
    preds = {}  # 当聚类对象不存在labels_属性，将预测结果存储在这个字典变量中

    # 建立各个聚类模型类的实例对象
    kmeans = cluster.KMeans(n_clusters=n_clusters)  # n_clusters 簇的数目
    minibatch_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)
    spectral = cluster.SpectralClustering(n_clusters=n_clusters,
                                          eigen_solver='arpack', affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=eps)

    algorithms = [kmeans, minibatch_means, spectral, dbscan]
    for name, algorithm in zip(names, algorithms):  # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组
        algorithm.fit(wh_array)
        if hasattr(algorithm, 'labels_'):  # 判断聚类对象是否存在labels_属性
            labels[name] = algorithm.labels_.astype(int)
        else:
            preds[name] = algorithm.predict(wh_array)
        if hasattr(algorithm, 'cluster_centers_'):  # 判断聚类对象是否存在cluster_centers_属性
            centers[name] = algorithm.cluster_centers_
    return labels, centers, preds


# ! 层次聚类
# 使用scipy.cluster.hierarrchy层次聚类方法进行聚类，这里的层次聚类为聚合式层次聚类，即初始时每个样本点看成一个簇。然后画出聚类结果的树状图。
# 通过wh_array数据矩阵得到连接矩阵(linkage matrix)
Z = sch.linkage(wh_array, method='ward', metric='euclidean')
# 通过cophenet函数来计算聚类结果的同表像相关系数(Cophenetic Correlation Coefficient)，该系数度量了树状图隐含的成对距离在多大程度上保留了原始数据的实际成对距离信息。
# 其值越接近1，层次聚类越好地保持了原始成对距离信息。

methods = ['single', 'complete', 'average', 'weighted', 'ward']
metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
           'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
           'jaccard', 'kulsinski', 'mahalanobis', 'matching',
           'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
           'sokalmichener', 'sokalsneath', 'sqeuclidean']

c_init = 0
for method in methods:
    for metric in metrics:
        try:  # 某些(method,metric)的组合理论上是不可行的，比如在连接方法为'ward'的情况下只能使用`euclidean`
            Z = sch.linkage(wh_array, method=method, metric=metric)
            c, coph_dists = sch.cophenet(Z, pdist(wh_array))
        except:
            pass
        if c > c_init:
            c_init = c
            best_para = (method, metric, c)  # 记录最佳参数组合
print(best_para)

# 使用最佳参数组合best_para进行层次聚类：
Z = sch.linkage(wh_array, method=best_para[0], metric=best_para[1])
# print(Z)

# 画出树状图

f, ax = plt.subplots(figsize=(20, 10))
plt.xlabel('sample index', fontsize=20)
plt.ylabel('distance', fontsize=20)
sch.set_link_color_palette(['g', 'y', 'c', 'k'])  # 设置色系
sch.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, color_threshold=0.12)  # 画树状图
plt.xticks(fontsize=8)
plt.yticks(fontsize=15)

# 如果我们要查看将所有样本聚为10个簇的结果:
f, ax = plt.subplots(figsize=(20, 10))  # 设置画布大小
plt.xlabel('sample index', fontsize=20)
plt.ylabel('distance', fontsize=20)
sch.set_link_color_palette(['g', 'y', 'c', 'k'])
sch.dendrogram(Z,
               truncate_mode='lastp',
               p=10,
               leaf_rotation=90,
               leaf_font_size=8,
               color_threshold=0.125,
               show_contracted=True, )
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.show()

# 提取层次聚类结果 
hierarchy_labels = sch.fcluster(Z, t=10, criterion='maxclust')  # t为聚类簇的数目
print(hierarchy_labels)
print(sch.fcluster(Z, t=0.15, criterion='distance'))
print(sch.fcluster(Z, t=3, depth=10))

# !利用距离矩阵实现聚类

dist_mat = squareform(pdist(wh_array))  # 通过pdist函数计算距离矩阵
cmap = sns.diverging_palette(130, 25, n=10, as_cmap=True)  # 生成散点图的颜色
sns.clustermap(dist_mat, cmap=cmap)

# 也可以直接通过array得到 clustermap，不过图形的上方是对特征进行的聚类
sns.clustermap(wh_norm.drop('Region', axis=1), cmap=cmap)  # 通过array得到 clustermap
# plt.show()

# !数据可视化

wh_array = preprocessing.StandardScaler().fit_transform(wh_norm.drop('Region', axis=1))
dimen_reduc_names = ['PCA', 'MDS', 'ISOMAP', 'LLE', 't-SNE', 'Spectral Embedding']
pca = decomposition.PCA(n_components=2)
mds = manifold.MDS(n_components=2, n_init=1, max_iter=100)
iso = manifold.Isomap(n_components=2, n_neighbors=2)
lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=3)
tsne = manifold.TSNE(n_components=2, init='pca', random_state=10)
spectral = manifold.SpectralEmbedding(n_components=2, random_state=10, eigen_solver='arpack')
dimen_reduc_algorithms = [pca, mds, iso, lle, tsne, spectral]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, name, algorithm in zip(range(len(dimen_reduc_names)), dimen_reduc_names, dimen_reduc_algorithms):  # 画出降维后的图形
    wh_2d = algorithm.fit_transform(wh_array)
    plt.subplot(2, 3, i + 1)
    plt.scatter(wh_2d[:, 0], wh_2d[:, 1], c='g')
    plt.title(name, fontsize=15)
    plt.xticks([])
    plt.yticks([])
# plt.show()
# 这六种降维方法并没有哪一种能够很明显地看出样本点聚集而成的集团


iso = manifold.Isomap(n_components=2, n_neighbors=2)  # 选择其中一种降维方法
wh_2d = iso.fit_transform(wh_array)  # 降维

# # 选择合适的颜色
sns.palplot(sns.color_palette("Set3", 10))
colors = np.array([x for x in sns.color_palette("Set3", 10)])
# 
# 画出最终簇的数目分别为2、6、10，以及使用四种不同聚类算法的结果：

n_clusters = range(2, 11, 4)  # 选择簇的数目
nrows = len(n_clusters)
ncols = len(names)
fig_counts = nrows * ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 15))  # 画布大小

for i, n_cluster in enumerate(n_clusters):
    labels, centers, preds = clustering(n_cluster, eps=1.6)  # 聚类
    for j, name in enumerate(names):
        label = labels[name]  # 提取标签
        label_transform = np.array([x - label.min() for x in label])  # 标签转换
        plt.subplot(3, 4, i * len(names) + j + 1)  # 画子图
        plt.scatter(wh_2d[:, 0], wh_2d[:, 1], color=colors[label_transform].tolist())  # 画散点图
        if name == 'DBSCAN':
            plt.title(name)
        else:
            plt.title(name + '\n' + 'n_cluster=' + str(n_cluster), fontsize=15)
        plt.xticks([])
        plt.yticks([])

# plt.show()


# !聚类结果分析
# 由以上的图可以看出，KMeans方法与MiniBatchKmeans方法的聚类结果大致相同(注意簇的颜色在不同的子图中没有对应关系)，
# 这个结果也非常容易理解。SpectralClustering方法与前两种方法存在较大的差异。DBSCAN是基于密度的聚类方法，簇内不同点散乱的分布在二维空间中。
label_rank = wh['Happiness Rank'].copy(deep=True)

# label_rank: 将变量'Region'进行按照等级编码
for i, element in enumerate(label_rank):
    label_rank[i] = (np.floor(i / 16))  # 16个国家为一组，每组的值相同
labels, centers, preds = clustering(n_clusters=10, eps=1.6)  # 聚类
labels['Happiness Ranks'] = label_rank
names.insert(0, 'Happiness Ranks')  # 在names列表最前面加上 'Regions'

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, name in enumerate(names):
    plt.subplot(1, 5, i + 1)  # 画子图
    plt.scatter(wh_2d[:, 0], wh_2d[:, 1], color=colors[labels[name]].tolist())  # 画散点图
    plt.title(name, fontsize=15)
    plt.xticks([])
    plt.yticks([])

names.remove('Happiness Ranks')  # 最后将'Regions'删除，防止重复运行时出错

# plt.show()

#  !聚类结果与 Region 的关系
label_region = wh_norm['Region'].copy(deep=True) 
for i, element in enumerate(pd.unique(wh_norm['Region'])):
    label_region.replace(element, i, inplace=True)  # label_region: 将类别变量'Region'进行数字编码之后得到的Series

labels, centers, preds = clustering(n_clusters=10, eps=1.6)
labels['Regions'] = label_region
names.insert(0, 'Regions')  # 在names列表最前面加上 'Regions'

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, name in enumerate(names):
    plt.subplot(1, 5, i + 1)
    plt.scatter(wh_2d[:, 0], wh_2d[:, 1], color=colors[labels[name]].tolist())
    plt.title(name, fontsize=15)
    plt.xticks([])
    plt.yticks([])

names.remove('Regions')  # 最后将 'Regions'删除，防止重复运行时出错
plt.show()
# 很容易从上图发现，属于同一区域的国家并不属于同一簇，故以上的聚类结果与区域之间并不存在很强的关系，
# 在同一区域的国家其幸福指数的权重差异可能由不同比重的关键因素造成。
# 综合以上分析，我们基本可以得出一个结论：为了提高一个国家的幸福指数，仿效同区域国家的改进措施，
# 有可能是不可取的。我们可以给出的建议是仿效同簇内的其他国家的改进措施。
