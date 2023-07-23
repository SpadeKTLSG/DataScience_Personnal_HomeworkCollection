# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
df = pd.read_csv("bank.csv", sep=";")  # 导入相关的数据
print(df.shape)  # 查看数据的维度(4521, 17)
print(df.columns)  # 数据的列名 
print(df.info())  # 查看数据的信息 all non-null  dtypes: int64(7), object(10)
print(df.describe())  # 查看数据的描述性统计信息 见下图
print(df.isnull().sum())  # 查看数据的缺失值 none
print(df['y'].value_counts())  # 查看因变量的分布  no     4000  yes     521


# 在预测之前需要将数据集中的object类型转换为数值类型, 转换方法:将object类型转换为数值类型,并将数值类型转换为one-hot编码
def ChangeIntoNum(df1):
    df1['job'] = df1['job'].map(
        {'admin.': 1, 'unknown': 2, 'unemployed': 3, 'management': 4, 'housemaid': 5, 'entrepreneur': 6, 'student': 7, 'blue-collar': 8, 'self-employed': 9, 'retired': 10,
         'technician': 11, 'services': 12})
    df1['marital'] = df1['marital'].map({'married': 1, 'divorced': 2, 'single': 3})
    df1['education'] = df1['education'].map({'unknown': 1, 'secondary': 2, 'primary': 3, 'tertiary': 4})
    df1['default'] = df1['default'].map({'yes': 1, 'no': 2})
    df1['housing'] = df1['housing'].map({'yes': 1, 'no': 2})
    df1['loan'] = df1['loan'].map({'yes': 1, 'no': 2})
    df1['contact'] = df1['contact'].map({'unknown': 1, 'telephone': 2, 'cellular': 3})
    df1['month'] = df1['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
    df1['poutcome'] = df1['poutcome'].map({'unknown': 1, 'other': 2, 'failure': 3, 'success': 4})
    df1['y'] = df1['y'].map({'yes': 1, 'no': 2})
    return df1


def ChangeIntoOneHot(df2):
    df2 = pd.get_dummies(df2, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
    return df2


# 开始转换,之后将数据集分为训练集和测试集，其中训练集占80%，测试集占20%
df = ChangeIntoNum(df)
df = ChangeIntoOneHot(df)
X = df.drop('y', axis=1)
y = df['y']
X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.2)
# 复制一份训练集, 用于后续的多元线性回归
X_train2 = X_train1.copy()
y_train2 = y_train1.copy()

# !1建立多元线性回归模型
reg = LinearRegression()  # 建立多元线性回归模型
reg.fit(X_train1, y_train1)  # 使用训练集拟合模型

# 查看回归系数和截距
print("回归系数: ", reg.coef_)
print("截距: ", reg.intercept_)

y_pred1 = reg.predict(X_test)  # 使用测试集进行预测

# 评估模型的性能
mse1 = mean_squared_error(y_test, y_pred1)  # 均方误差
rmse1 = np.sqrt(mse1)  # 均方根误差
r21 = r2_score(y_test, y_pred1)  # R方值

print("\n\n线性回归模型:")
print("均方误差: ", mse1)
print("均方根误差 ", rmse1)
print("R方值: ", r21, "\n\n")

# 可视化预测值和真实值的散点图
fig = px.scatter(x=y_test, y=y_pred1, labels={'x': 'Actual', 'y': 'Predicted'})
fig.show()
# 可视化预测值和真实值的误差分布图
fig = px.histogram(x=y_test - y_pred1, labels={'x': 'Error'}, nbins=50)
fig.show()

# !2建立多项式回归模型
# 生成多项式特征，假设使用二次多项式
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train2)
X_test_poly = poly.transform(X_test)

# 建立多项式回归模型
reg2 = LinearRegression()
reg2.fit(X_train_poly, y_train2)  # 使用训练集拟合模型

y_pred2 = reg2.predict(X_test_poly)  # 使用测试集进行预测

# 评估模型的性能
mse2 = mean_squared_error(y_test, y_pred2)  # 均方误差
rmse2 = np.sqrt(mse2)  # 均方根误差
r22 = r2_score(y_test, y_pred2)  # R方值

print("多项式回归模型:")
print("均方误差: ", mse2)
print("均方根误差: ", rmse2)
print("R方值: ", r22, "\n\n")

# 可视化预测值和真实值的散点图
fig = px.scatter(x=y_test, y=y_pred2, labels={'x': 'Actual', 'y': 'Predicted'})
fig.show()
# 可视化预测值和真实值的误差分布图
fig = px.histogram(x=y_test - y_pred2, labels={'x': 'Error'}, nbins=50)
fig.show()

# !对比两者的性能

# 不同回归模型的RMSE和R-squared
fig = go.Figure()
fig.add_trace(go.Scatter(x=["线性", "多项式"], y=[rmse1, rmse2], mode="lines+markers", name="RMSE"))
fig.add_trace(go.Bar(x=["线性", "多项式"], y=[r21, r22], name="R方值"))
fig.update_layout(title="回归模型比较", xaxis_title="模型", yaxis_title="量度")
fig.show()

# 不同回归模型的预测值和真实值的散点图矩阵
df_compare = pd.DataFrame({"真实": y_test, "线性": y_pred1, "多项式": y_pred2})
fig = px.scatter_matrix(df_compare)
fig.update_layout(title="回归模型的散布矩阵")
fig.show()
