# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from cffi.backend_ctypes import xrange
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')  # 不显示warning信息
pd.options.display.width = 900  # Dataframe 显示宽度设置

# 读取文件
train = pd.read_csv('accord_sedan_training.csv')
print('The shape is', train.shape)
train.head(7)

# 将 train 划分为 x 和 y
train_x = train.drop('price', 1)
train_y = train['price']

# 4. 特征提取并构建线性回归模型
dv = DictVectorizer()
dv.fit(train.T.to_dict().values())  # one-hot 编码
print('Dimension is', len(dv.feature_names_), '\n\n', dv.feature_names_)

# One-Hot编码方法二：使用`pandas`的`get_dummies`函数实现
nomial_var = ['engine', 'trim', 'transmission']
multi_dummies = []  # 存储三个 DataFrame
train_x_dummies = train_x[['mileage', 'year']]
for col in nomial_var:
    dummies = pd.get_dummies(train_x[col], prefix=col)
    train_x_dummies = pd.concat([train_x_dummies, dummies], axis=1)  # 将编码结果与非编码特征水平拼接起来
print(train_x_dummies.head())

# 构建线性回归模型
train_x_array = dv.transform(train_x.T.to_dict().values())

print('\n\n')
LR = LinearRegression().fit(train_x_array, train_y)
print('得到了模型', '+'.join([format(LR.intercept_, '0.2f')] + [str('(') + str(b) + a + str(')') for a, b in zip(dv.feature_names_, LR.coef_)]))
# 得到了模型

# 由测试集拟合得到的模型，我们可以预测测试集中的价格，计算每个样本的绝对误差
pred_y = LR.predict(dv.transform(train.T.to_dict().values()))
train_error = abs(pred_y - train_y)  # 计算绝对误差
print('\n\n')
print('绝对误差数据的百分位数', np.percentile(train_error, [75, 90, 95, 99]))  # 计算绝对误差数据的百分位数

sns.boxplot(x=train_error, palette="Set2")
# 输出图1盒图观察离群值

# 在本案例中，我们设定置信水平为0.95，即认为超过95%百分位数的train_error为离群值。下面我们在二维空间中画出正常值（蓝色）与离群值（红色）：

outlierIndex = train_error >= np.percentile(train_error, 95)
inlierIndex = train_error < np.percentile(train_error, 95)

# 得到train_error最大的index值，即极端离群值
most_severe = train_error[outlierIndex].idxmax()

fig = plt.figure(figsize=(7, 7))
indexes = [inlierIndex, outlierIndex, most_severe]
color = ['#2d9ed8', '#EE5150', '#a290c4']
label = ['normal points', 'outliers', 'extreme outliers']
for i, c, l in zip(indexes, color, label):
    plt.scatter(train['mileage'][i],
                train_y[i],
                c=c,
                marker='^',
                label=l)
plt.legend(loc='upper right',
           frameon=True,
           edgecolor='k',
           framealpha=1,
           fontsize=12)
plt.xlabel('$mileage$')
plt.ylabel('$price$')
plt.grid('on')
sns.set_style('dark')
# 输出图二正常值（蓝色）与离群值（红色）

# 看看有多少离群值:
print(outlierIndex.value_counts())

# 上图结果也符合我们的经验理解，二手车的行驶公里数越高，它卖出去的价格就应该越低，所以对于处在右上和左下区域的点可能是一些离群值（对于同一款车而言）。比如左下区域的点，一些行驶里程数低，价格也比较低的车辆，有可能该车辆是事故车辆或者有损坏，而右上区域的离群值有可能是真实的离群值，相对来讲不容易有合理的解释，可能是输入失误或者胖手指输入造成。
# 通常情况下，为了避免不同尺度的影响。我们在进行线性回归模型拟合之前，需要对各个特征进行标准化。常见的标准化有z-score标准化、0-1标准化等，这里我们选择z-score标准化来观察标准化对离群值检测的影响。


# 利用 preprocessing.scale函数将特征标准化
columns = train_x_dummies.columns
train_x_zscore = pd.DataFrame(preprocessing.scale(train_x_dummies), columns=columns)
# train_y_zscore = pd.DataFrame(preprocessing.scale(pd.DataFrame(train_y,columns=['price'])),columns = ['price'])

# 线性模型拟合
LR_zscore = LinearRegression().fit(train_x_zscore.values, train_y)
print('\n\n')
print('+'.join([format(LR.intercept_, '0.2f')] + [str('(') + str(b) + a + str(')') for a, b in zip(dv.feature_names_, LR_zscore.coef_)]))

pred_y_zscore = LR_zscore.predict(train_x_zscore)
train_error_zscore = abs(pred_y_zscore - train_y)  # 计算绝对误差
print(np.percentile(train_error_zscore, [75, 90, 95, 99]))  # 计算绝对误差数据的百分位数
print('\n\n')
outlierIndex_zscore = train_error_zscore >= np.percentile(train_error_zscore, 95)
inlierIndex_zscore = train_error_zscore < np.percentile(train_error_zscore, 95)
diff = (outlierIndex_zscore != outlierIndex)  # diff 用于存储标准化前后的离群值检测结果不同的index
print(diff.value_counts())
print('\n\n')

# 画出标准化前后的检测差异点
fig = plt.figure(figsize=(7, 7))

# rep_inlierIndex为标准化前后都为正常值的index
rep_inlierIndex = (inlierIndex == inlierIndex_zscore)

indexes = [rep_inlierIndex, outlierIndex, outlierIndex_zscore]
color = ['#2d9ed8', '#EE5150', '#a290c4']
markers = ['^', '<', '>']
label = ['inliers', 'outliers before z-score', 'outliers after z-score']
for i, c, m, l in zip(indexes, color, markers, label):
    plt.scatter(train['mileage'][i],
                train_y[i],
                c=c,
                marker=m,
                label=l)
plt.xlabel('$mileage$')
plt.ylabel('$price$')
plt.grid('on')
plt.legend(loc='upper right',
           frameon=True,
           edgecolor='k',
           framealpha=1,
           fontsize=12)
sns.set_style('dark')

#
# 从结果可以看到，绝大多数样本的检测结果一致。有两个样本存在差别，其中一个样本在标准化之前会被检测为离群值，另外一个样本在标准化之后会被检测为离群值。虽然在本例中，标准化前后的检测效果差异不是很大，我们仍然建议在线性建模之前对特征进行标准化。

# 测试集的验证

test = pd.read_csv('accord_sedan_testing.csv')

datasets = [train, test]
color = ['#2d9ed8', '#EE5150']
label = ['training set', 'testing set']
fig = plt.figure(figsize=(7, 7))
for i, c, l in zip(range(len(datasets)), color, label):
    plt.scatter(datasets[i]['mileage'],
                datasets[i]['price'],
                c=c,
                marker='^',
                label=l)
plt.xlabel('$mileage$')
plt.ylabel('$price$')
plt.grid('on')
plt.legend(loc='upper right',
           frameon=True,
           edgecolor='k',
           framealpha=1,
           fontsize=12)
sns.set_style('dark')

pred_y_test = LR.predict(dv.transform(test.T.to_dict().values()))
test_error = abs(pred_y_test - test['price'])

# 使用分布图观察测试集误差
fig = plt.figure(figsize=(7, 7))
sns.distplot(test_error, kde=False)
plt.xlabel('$test\_error$')
plt.ylabel('$count$')
plt.grid('on')

# 找出极端离群值
most_severe_test = test_error.idxmax()
print(test.iloc[most_severe_test])

# 在测试集上使用LOF进行离群值检测
test = pd.read_csv('accord_sedan_testing.csv')
fig = plt.figure(figsize=(7, 7))
plt.scatter(test['mileage'],
            test['price'],
            c='#EE5150',
            marker='^',
            label='testing set')
plt.xlabel('$mileage$')
plt.ylabel('$price$')
plt.grid('on')
plt.legend(loc='upper right',
           frameon=True,
           edgecolor='k',
           framealpha=1,
           fontsize=12)
sns.set_style('dark')

# 定义函数计算第k可达距离
test_2d = test[['mileage', 'price']]
from sklearn.neighbors import NearestNeighbors
import numpy as np

neigh = NearestNeighbors()  # 默认为欧式距离
model = neigh.fit(test_2d)

data = test_2d
# dist为每个样本点与第k距离邻域内的点的距离（包括自身）,neighbor为第k距离邻域点的编号（包括自身）
dist, neighbor = neigh.kneighbors(test_2d, n_neighbors=6)

k_distance_p = np.max(dist, axis=1)

nums = data.shape[0]
lrdk_p = []
lof = []
indicy = []
for p_index in xrange(nums):
    rdk_po = []
    neighbor_p = neighbor[p_index][neighbor[p_index] != p_index]
    for o_index in neighbor_p:
        rdk_po.append(max(k_distance_p[o_index], int(dist[p_index][neighbor[p_index] == o_index])))
    lrdk_p.append(float(len(neighbor_p)) / sum(rdk_po))

for p_index in xrange(nums):
    lrdk_o = []
    neighbor_p = neighbor[p_index][neighbor[p_index] != p_index]
    for o_index in neighbor_p:
        lrdk_o.append(lrdk_p[o_index])
    lof.append(float(sum(lrdk_o)) / (len(neighbor_p) * (lrdk_p[p_index])))

fig = plt.figure(figsize=(7, 7))

for index, size in zip(xrange(nums), lof):
    if index in indicy:
        plt.scatter(data['mileage'][index],
                    data['price'][index],
                    s=np.exp(lof[index]) * 50,
                    c='#efab40',
                    alpha=0.6,
                    marker='o')
        plt.text(data['mileage'][index] - np.exp(lof[index]) * 50,
                 data['price'][index] - np.exp(lof[index]) * 50,
                 str(round(lof[index], 2)))
    else:
        plt.scatter(data['mileage'][index],
                    data['price'][index],
                    s=np.exp(lof[index]) * 50,
                    c='#5dbe80',
                    alpha=0.6,
                    marker='o')
        plt.text(data['mileage'][index] - np.exp(lof[index]) * 50,
                 data['price'][index] - np.exp(lof[index]) * 50,
                 str(round(lof[index], 2)),
                 fontsize=7)

plt.xlabel('mileage')
plt.ylabel('price')
plt.grid('off')

plt.show()
