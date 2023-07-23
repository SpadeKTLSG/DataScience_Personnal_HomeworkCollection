# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pylab as pl

insurance = pd.read_csv("insurance.csv")
# print(insurance.head(10))

# 1 入手分析
print(insurance["charges"].describe())  # 查看关键元素charges的分布
# 因为平均值远大于中位数，所以这表明保险费用的分布是右偏的

pl.hist(insurance["charges"])
pl.xlabel('charges')
print('\n\n')
# 直方图


# 2 进一步分析
# 变量sex被划分成male和female两个水平，而变量smoker被划分成yes和no两个水平。从describe()的输出中，我们知道变量region有4个水平，但我们需要仔细看一看，它们是如何分布的。
print(insurance["sex"].describe())
print(insurance["smoker"].describe())
print(insurance["region"].describe())
print(insurance.region.value_counts())

# 这里，我们看到数据几乎均匀地分布在4个地理区域中。

# 3 
# 在使用回归模型拟合数据之前，有必要确定自变量与因变量之间以及自变量之间是如何相关的。相关系数矩阵（correlation matrix）提供了这些关系的快速概览。给定一组变量，它可以为每一对变量之间的关系提供一个相关系数。

print('\n\n')

print(insurance[["age", "bmi", "children", "charges"]].corr())  # 创建一个相关系数矩阵


# 该矩阵中的相关系数不是强相关的，但还是存在一些显著的关联。例如，age和bmi显示出中度相关，这意味着随着年龄（age）的增长，身体质量指数（bmi）也会增加。此外，age和charges，bmi和charges，以及children和charges也都呈现出中度相关。

# 使用虚拟编码技术来完成对类别变量的分析:
def dummycoding(dataframe):
    dataframe_age = dataframe['age']
    dataframe_bmi = dataframe['bmi']
    dataframe_children = dataframe['children']
    dataframe_charges = dataframe['charges']
    dataframe_1 = dataframe.drop(['age'], axis=1)
    dataframe_2 = dataframe_1.drop(['bmi'], axis=1)
    dataframe_3 = dataframe_2.drop(['children'], axis=1)
    dataframe_new = dataframe_3.drop(['charges'], axis=1)

    dataframe_new = pd.get_dummies(dataframe_new, prefix=dataframe_new.columns).astype(int)

    dataframe_new['age'] = dataframe_age
    dataframe_new['bmi'] = dataframe_bmi
    dataframe_new['children'] = dataframe_children
    dataframe_new['charges'] = dataframe_charges
    return dataframe_new


insurance_lm = dummycoding(insurance)

print(insurance_lm.head(10))

# 我们已经完成了对insurance中类型变量的虚拟编码。然后在回归模型中保留sex_female、smoker_no和region_northeast变量，使东北地区的女性非吸烟者作为参照组。


insurance_lm_y = insurance_lm['charges']
insurance_lm_X1 = insurance_lm.drop(['charges'], axis=1)
insurance_lm_X2 = insurance_lm_X1.drop(['sex_female'], axis=1)
insurance_lm_X3 = insurance_lm_X2.drop(['smoker_no'], axis=1)
insurance_lm_X = insurance_lm_X3.drop(['region_northeast'], axis=1)

print(insurance_lm_X.head(10))
print(insurance_lm_y.head(10))

print('\n\n')
regr = linear_model.LinearRegression()
regr.fit(insurance_lm_X, insurance_lm_y)

print('Intercept: %.2f'
      % regr.intercept_)
print('Coefficients: ')
print(regr.coef_)
print('Residual sum of squares: %.2f'
      % np.mean((regr.predict(insurance_lm_X) - insurance_lm_y) ** 2))
print('Variance score: %.2f' % regr.score(insurance_lm_X, insurance_lm_y))
print('\n\n')


pl.show()


#考虑到非线性关系，可以添加一个高阶项到回归模型中，把模型当做多项式处理。
insurance_lm['age2'] = insurance_lm['age']*insurance_lm['age']
print(insurance_lm['age2'].head(10))


# 然后，当我们建立改进后的模型时，我们将把age和age2都添加到回归模型中。
# 假设我们有一种预感，一个特征的影响不是累积的，而是当特征的取值达到一个给定的阈值后才产生影响。例如，对于在正常体重范围内的个人来说，BMI对医疗费用的影响可能为0，但是对于肥胖者（即BMI不低于30）来说，它可能与较高的费用密切相关。
# 
# 我们可以通过创建一个二进制指标变量来建立这种关系，即如果BMI大于等于30，那么设定为1，否则设定为0。该二元特征的 β
# 估计表示BMI大于等于30的个人相对于BMI小于30的个人对医疗费用的平均净影响。


print('\n\n')
insurance_lm['bmi30'] = 0
# 
# for i in range(0, 1338):
#     if insurance_lm['bmi'][i] >= 30 :
#         insurance_lm['bmi30'][i] = 1
#     else:
#         insurance_lm['bmi30'][i] = 0
# 
# insurance_lm['bmi30'].head(10)
# 当两个特征存在共同的影响时，这称为相互作用（interaction）。如果怀疑两个变量相互作用，那么可以通过在模型中添加它们的相互作用来检验这一假设，可以使用R中的公式语法来指定相互作用的影响。为了体现肥胖指标（bmi30）和吸烟指标（smoker）的相互作用，可以将bmi30 ∗
# smoker_yes也作为自变量放入模型。

insurance_lm['bmi30_smoker']=insurance_lm['bmi30']*insurance_lm['smoker_yes']
print(insurance_lm.head(15))
print('\n\n')

insurance_lm_y = insurance_lm['charges']
insurance_lm_X1 = insurance_lm.drop(['charges'], axis = 1)
insurance_lm_X2 = insurance_lm_X1.drop(['sex_female'], axis = 1)
insurance_lm_X3 = insurance_lm_X2.drop(['smoker_no'], axis = 1)
insurance_lm_X = insurance_lm_X3.drop(['region_northeast'], axis = 1)

regr = linear_model.LinearRegression()
regr.fit(insurance_lm_X, insurance_lm_y)

print('Intercept: %.2f'
      % regr.intercept_)
print('Coefficients: ')
print(regr.coef_)
print('Residual sum of squares: %.2f'
      % np.mean((regr.predict(insurance_lm_X) - insurance_lm_y) ** 2))
print('Variance score: %.2f' % regr.score(insurance_lm_X, insurance_lm_y))
print('\n\n')
# 分析该模型的拟合统计量有助于确定我们的改变是否提高了回归模型的性能。相对于我们的第一个模型，R方值从0.75提高到约0.87，我们的模型现在能解释医疗费用变化的87%。肥胖和吸烟之间的相互作用表明了一个巨大的影响，除了单独吸烟增加的超过13404美元的费用外，肥胖的吸烟者每年要另外花费19810美元，这可能表明吸烟会加剧（恶化）与肥胖有关的疾病。