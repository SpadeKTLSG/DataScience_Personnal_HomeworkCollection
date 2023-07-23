# -*- coding: gbk -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')  # 不显示warning信息
pd.set_option('display.max_columns', None)  # 显示所有列
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

df = pd.read_csv('bank_all.csv', sep=';')  # 读取最大数据集进行训练,可以看到我也准备了备用的数据集

'''
# print(df.shape)  # 查看数据集的维度: (45211, 17)
# print(df.head())  # 查看数据集的前5行
# print(df.tail())  # 查看数据集的后5行
# print(df.columns) # 查看数据集的列名
# print(df.index) # 查看数据集的索引
# print(df.info())  # 查看数据集的基本信息: 不赘述
# print(df.isnull().any())  # 查看数据集的值是否为空: 无缺失值

# print(df.dtypes) # 查看数据集的数据类型,列表如下:

# 大部分是object类型，后期需要转换成数值类型
# age           int64
# job          object
# marital      object
# education    object
# default      object
# balance       int64
# housing      object
# loan         object
# contact      object
# day           int64
# month        object
# duration      int64
# campaign      int64
# pdays         int64
# previous      int64
# poutcome     object
# y            object

# print(df['y'].value_counts())  # 查看数据集的标签分布

# age           10.618762
# balance     3044.765829
# day            8.322476
# duration     257.527812
# campaign       3.098021
# pdays        100.128746
# previous       2.303441
# dtype: float64'''

'''# 字段标签说明：
# 'age年龄', 'job职业', 'marital婚姻状况', 'education教育', 'default违约状况', 'balance余额', 'housing房贷',
# 'loan贷款', 'contact联系类型', 'day最后联系日', 'month最后联系月份', 'duration联系时长', 'campaign联系人数', 
# 'pdays联系过几天','previous之前联系的数量', 'poutcome之前的结果',
# 'y贷款了吗?'''

# !第一步:简单数据分析&预处理
# 计算标准差、最小值和最大值,描述性统计信息,并进行数据可视化
bankdescribe = df.describe()
bankstd = df.std()
bankmin = df.min()
bankmax = df.max()
df.hist(figsize=(10, 10))
plt.show()

# 首先将这个数据集拆分为训练集和测试集(yes/no)
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 在预测之前需要将数据集中的object类型转换为数值类型, 转换方法:将object类型转换为数值类型,并将数值类型转换为one-hot编码

# ?这里是一些重复的方法,打包成函数了
def SetDownPlz():
    global y_train, y_test, X_train, X_test, y_dummies_train, y_dummies_test
    # 1.将object类型转换为数值类型
    # 1.1 将job转换为数值类型
    job_mapping = {'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4, 'management': 5, 'retired': 6,
                   'self-employed': 7, 'services': 8, 'student': 9, 'technician': 10, 'unemployed': 11, 'unknown': 12}
    X_train['job'] = X_train['job'].map(job_mapping)
    X_test['job'] = X_test['job'].map(job_mapping)
    # 1.2 将marital转换为数值类型
    marital_mapping = {'divorced': 1, 'married': 2, 'single': 3, 'unknown': 4}
    X_train['marital'] = X_train['marital'].map(marital_mapping)
    X_test['marital'] = X_test['marital'].map(marital_mapping)
    # 1.3 将education转换为数值类型
    education_mapping = {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 4}
    X_train['education'] = X_train['education'].map(education_mapping)
    X_test['education'] = X_test['education'].map(education_mapping)
    # 1.4 将default转换为数值类型
    default_mapping = {'no': 1, 'yes': 2, 'unknown': 3}
    X_train['default'] = X_train['default'].map(default_mapping)
    X_test['default'] = X_test['default'].map(default_mapping)
    # 1.5 将housing转换为数值类型
    housing_mapping = {'no': 1, 'yes': 2, 'unknown': 3}
    X_train['housing'] = X_train['housing'].map(housing_mapping)
    X_test['housing'] = X_test['housing'].map(housing_mapping)
    # 1.6 将loan转换为数值类型
    loan_mapping = {'no': 1, 'yes': 2, 'unknown': 3}
    X_train['loan'] = X_train['loan'].map(loan_mapping)
    X_test['loan'] = X_test['loan'].map(loan_mapping)
    # 1.7 将contact转换为数值类型
    contact_mapping = {'cellular': 1, 'telephone': 2, 'unknown': 3}
    X_train['contact'] = X_train['contact'].map(contact_mapping)
    X_test['contact'] = X_test['contact'].map(contact_mapping)
    # 1.8 将month转换为数值类型
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5,
                     'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
                     'nov': 11, 'dec': 12}
    X_train['month'] = X_train['month'].map(month_mapping)
    X_test['month'] = X_test['month'].map(month_mapping)
    # 1.9 将poutcome转换为数值类型
    poutcome_mapping = {'failure': 1, 'other': 2, 'success': 3, 'unknown': 4}
    X_train['poutcome'] = X_train['poutcome'].map(poutcome_mapping)
    X_test['poutcome'] = X_test['poutcome'].map(poutcome_mapping)
    # 1.10 将y转换为数值类型
    y_mapping = {'no': 1, 'yes': 2}
    y_train = y_train.map(y_mapping)
    y_test = y_test.map(y_mapping)
    # 2.将数值类型转换为one-hot编码
    # 2.1 将job转换为one-hot编码
    job_dummies_train = pd.get_dummies(X_train['job'], prefix='job')
    job_dummies_test = pd.get_dummies(X_test['job'], prefix='job')
    X_train = pd.concat([X_train, job_dummies_train], axis=1)
    X_test = pd.concat([X_test, job_dummies_test], axis=1)
    X_train.drop('job', axis=1, inplace=True)
    X_test.drop('job', axis=1, inplace=True)
    # 2.2 将marital转换为one-hot编码
    marital_dummies_train = pd.get_dummies(X_train['marital'], prefix='marital')
    marital_dummies_test = pd.get_dummies(X_test['marital'], prefix='marital')
    X_train = pd.concat([X_train, marital_dummies_train], axis=1)
    X_test = pd.concat([X_test, marital_dummies_test], axis=1)
    X_train.drop('marital', axis=1, inplace=True)
    X_test.drop('marital', axis=1, inplace=True)
    # 2.3 将education转换为one-hot编码
    education_dummies_train = pd.get_dummies(X_train['education'], prefix='education')
    education_dummies_test = pd.get_dummies(X_test['education'], prefix='education')
    X_train = pd.concat([X_train, education_dummies_train], axis=1)
    X_test = pd.concat([X_test, education_dummies_test], axis=1)
    X_train.drop('education', axis=1, inplace=True)
    X_test.drop('education', axis=1, inplace=True)
    # 2.4 将default转换为one-hot编码
    default_dummies_train = pd.get_dummies(X_train['default'], prefix='default')
    default_dummies_test = pd.get_dummies(X_test['default'], prefix='default')
    X_train = pd.concat([X_train, default_dummies_train], axis=1)
    X_test = pd.concat([X_test, default_dummies_test], axis=1)
    X_train.drop('default', axis=1, inplace=True)
    X_test.drop('default', axis=1, inplace=True)
    # 2.5 将housing转换为one-hot编码
    housing_dummies_train = pd.get_dummies(X_train['housing'], prefix='housing')
    housing_dummies_test = pd.get_dummies(X_test['housing'], prefix='housing')
    X_train = pd.concat([X_train, housing_dummies_train], axis=1)
    X_test = pd.concat([X_test, housing_dummies_test], axis=1)
    X_train.drop('housing', axis=1, inplace=True)
    X_test.drop('housing', axis=1, inplace=True)
    # 2.6 将loan转换为one-hot编码
    loan_dummies_train = pd.get_dummies(X_train['loan'], prefix='loan')
    loan_dummies_test = pd.get_dummies(X_test['loan'], prefix='loan')
    X_train = pd.concat([X_train, loan_dummies_train], axis=1)
    X_test = pd.concat([X_test, loan_dummies_test], axis=1)
    X_train.drop('loan', axis=1, inplace=True)
    X_test.drop('loan', axis=1, inplace=True)
    # 2.7 将contact转换为one-hot编码
    contact_dummies_train = pd.get_dummies(X_train['contact'], prefix='contact')
    contact_dummies_test = pd.get_dummies(X_test['contact'], prefix='contact')
    X_train = pd.concat([X_train, contact_dummies_train], axis=1)
    X_test = pd.concat([X_test, contact_dummies_test], axis=1)
    X_train.drop('contact', axis=1, inplace=True)
    X_test.drop('contact', axis=1, inplace=True)
    # 2.8 将month转换为one-hot编码
    month_dummies_train = pd.get_dummies(X_train['month'], prefix='month')
    month_dummies_test = pd.get_dummies(X_test['month'], prefix='month')
    X_train = pd.concat([X_train, month_dummies_train], axis=1)
    X_test = pd.concat([X_test, month_dummies_test], axis=1)
    X_train.drop('month', axis=1, inplace=True)
    X_test.drop('month', axis=1, inplace=True)
    # 2.9 将poutcome转换为one-hot编码
    poutcome_dummies_train = pd.get_dummies(X_train['poutcome'], prefix='poutcome')
    poutcome_dummies_test = pd.get_dummies(X_test['poutcome'], prefix='poutcome')
    X_train = pd.concat([X_train, poutcome_dummies_train], axis=1)
    X_test = pd.concat([X_test, poutcome_dummies_test], axis=1)
    X_train.drop('poutcome', axis=1, inplace=True)
    X_test.drop('poutcome', axis=1, inplace=True)
    # 2.10 将y转换为one-hot编码
    # 将y转换为one-hot编码
    y_dummies_train = pd.get_dummies(y_train, prefix='y')
    y_dummies_test = pd.get_dummies(y_test, prefix='y')


def SeeseemyHoneyisGood():
    pass
    # 转换完毕,查看转换后的数据:没问题
    # print('转换后的数据：')
    # print(X_train.head())
    # print(X_test.head())
    # print(y_dummies_train.head())
    # print(y_dummies_test.head())
    # 检查二者是否对应:对应了
    # print('检查二者是否对应：')
    # print(X_train.shape)
    # print(y_dummies_train.shape)
    # print(X_test.shape)
    # print(y_dummies_test.shape)


SetDownPlz()
# SeeseemyHoneyisGood()

# !第二步:使用C4.5对营销结果进行预测
# C4.5是一种决策树算法，它是ID3算法的改进版，它的核心思想是以信息熵为准则选择特征，以信息增益比为准则进行特征划分，以递归的方式生成决策树。


clf = DecisionTreeClassifier(criterion='entropy')  # 创建决策树分类器
clf.fit(X_train, y_dummies_train)  # 拟合训练集

y_pred = clf.predict(X_test)  # 对测试集进行预测

accuracy = clf.score(X_test, y_dummies_test)  # 计算模型的准确性acc= 87.5%
print('模型的准确性：', accuracy)

# 通过可视化图表展示预测结果:

# 1.绘制混淆矩阵
cm = confusion_matrix(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))  # 计算混淆矩阵
print('混淆矩阵：', cm)

# 2.绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))  # 计算误判率和正确率
roc_auc = auc(fpr, tpr)  # 计算AUC,AUC（Area Under Curve）被定义为ROC曲线下与坐标轴围成的面积
print('AUC：', roc_auc)  # AUC=0.87

lw = 2  # 线宽=2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 画出ROC曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # 画出对角线
plt.xlim([0.0, 1.0])  # 设置x轴上下限
plt.ylim([0.0, 1.05])  # 设置y轴上下限
plt.xlabel('误判率')  # 设置x轴标签
plt.ylabel('正确率')  # 设置y轴标签
plt.title('预测模型的ROC曲线')  # 设置标题
plt.legend(loc="lower right")  # 设置图例位置
plt.show()  # 显示图表

# !第三步,使用SVM对营销结果进行预测
# SVM是一种二分类模型，它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机；SVM还包括核技巧，这使它成为实质上的无线性分类器。
# SVM的的学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题,其学习算法就是求解凸二次规划的最优化算法。


clf = svm.SVC(kernel='linear', C=1)  # 创建SVM分类器

y_dummies_train = np.array(y_dummies_train).ravel()  # y_dummies_train是一个二维数组，需要转换成一维数组
X_train = np.resize(X_train, (360,))  # 这里从36000变成360
y_dummies_train = np.resize(y_dummies_train, (360,))  # 这里从36000变成360
X_train = X_train.reshape(-1, 1)  # 由于只能接受有一个特征，所以需要reshape一下
y_dummies_train = y_dummies_train.reshape(-1, 1)  # 由于只能接受有一个特征，所以需要reshape一下

# 机器性能不足,无法在原样本量下继续训练,等待15分钟只能放弃;无奈这个SVM模型只能缩小训练集
# keeps Err : X has 51 features, but SVC is expecting 1 features as input 此问题始终无法解决


clf.fit(X_train, y_dummies_train)  # 拟合训练集

y_pred = clf.predict(X_test)  # 利用训练集的结果对测试集进行预测

accuracy = clf.score(X_test, y_dummies_test)  # 看看准确率如何
print('模型的准确性：', accuracy)

# 绘制混淆矩阵
cm = confusion_matrix(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))
print('混淆矩阵：', cm)

# AUC图表
fpr, tpr, thresholds = roc_curve(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))  # 计算误判率和正确率
roc_auc = auc(fpr, tpr)  # 计算AUC,AUC（Area Under Curve）被定义为ROC曲线下与坐标轴围成的面积
print('AUC：', roc_auc)  # AUC=0.87

# !第四步,使用adaboost对营销结果进行预测

# adaboost是一种迭代算法，其核心思想是针对同一个训练集训练不同的分类器，然后把这些分类器集成起来，
# 对新的数据进行分类时，每个分类器都对该数据进行分类，最后根据分类器的投票数进行判断。

clf = AdaBoostClassifier(n_estimators=100)  # 创建adaboost分类器

y_dummies_train = np.array(y_dummies_train).ravel()  # y_dummies_train是一个二维数组，需要转换成一维数组
X_train = np.resize(X_train, (360,))  # 这里从36000变成360
y_dummies_train = np.resize(y_dummies_train, (360,))
X_train = X_train.reshape(-1, 1)  # 由于只能接受有一个特征，所以需要reshape一下
y_dummies_train = y_dummies_train.reshape(-1, 1)  # 由于只能接受有一个特征，所以需要reshape一下

# 无法解决此问题: ValueError: X has 51 features, but AdaBoostClassifier is expecting 1 features as input.
# 实验宣告失败

clf.fit(X_train, y_dummies_train)  # 拟合训练集
y_pred = clf.predict(X_test)  # 利用训练集的结果对测试集进行预测
accuracy = clf.score(X_test, y_dummies_test)  # 看看准确率
print('模型的准确性：', accuracy)

# 展示预测结果: 绘制混淆矩阵

cm = confusion_matrix(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))
print('混淆矩阵：', cm)
fpr, tpr, thresholds = roc_curve(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))  # 计算误判率和正确率
roc_auc = auc(fpr, tpr)
print('AUC：', roc_auc)

# 图表...

# !第五步,比较结果并分析

# 详情见第一个模型分析,由于无法完成Part3,4任务目标,故无法进行比较分析.同时,由于机器性能限制,双方使用的数据量不同,无法进行比较分析.
# 部分实验宣告失败,但是学习到了很多知识,提供了训练的框架和思路, 完成了其中一种训练的方法.这也算是一种收获吧.
