# -*- coding: gbk -*- 
# 实验3 用支持向量机进行光学字符识别
# 使用UCI的光学字符识别数据集
import pandas as pd
from sklearn.svm import SVC

letters = pd.read_csv("letterecognition.csv")
print(letters.head(10))  # 展示前10行数据

# 数据集中的每个样本都是一个字母，每个字母都有16个特征，每个特征都是一个整数
print('\n\n\n')
print(letters["letter"].value_counts().sort_index())  # 统计每个字母的个数
print('\n\n\n')  # 可见，各个字符的样本数量分布相对均衡。现在，我们进一步观察每一个自变量的取值分布：

print(letters.iloc[:, 1:].describe())  # 查看每个自变量的取值分布
# 观察发现16个自变量的取值范围都在0~15之间，因此对于该数据集我们不需要对变量进行标准化操作。 此外，数据集作者已经将样本随机排列，所以也不需要我们对数据进行随机打散。 此处，我们直接取前14000个样本（70%）作为训练集，后6000个样本（30%）作为测试集。
# 设置训练集和测试集
letters_train = letters.iloc[0:14000, ]
letters_test = letters.iloc[14000:20000, ]

# 模型训练
# 使用sklearn.svm包中的相关类来实现来构建基于支持向量机的光学字符识别模型 ,选用 SVC 来进行模型构建,SVC 有两个主要的参数可以设置：核函数参数 kernel 和约束惩罚参数C

letter_recognition_model = SVC(C=1, kernel="linear")
letter_recognition_model.fit(letters_train.iloc[:, 1:], letters_train['letter'])

# 性能评估
# 首先，使用predict()函数得到上一节训练的支持向量机模型在测试集合上的预测结果，然后使用 sklearn.metrics中的相关函数对模型的性能进行评估。
from sklearn import metrics

letters_pred = letter_recognition_model.predict(letters_test.iloc[:, 1:])
print(metrics.classification_report(letters_test["letter"], letters_pred))
print(pd.DataFrame(metrics.confusion_matrix(letters_test["letter"], letters_pred),
                   columns=letters["letter"].value_counts().sort_index().index,
                   index=letters["letter"].value_counts().sort_index().index))

# 上述混淆矩阵中对角线的元素表示模型正确预测数，对角元素之和表示模型整体预测正确的样本数。 而非对角线元素上的值则可以反映模型在哪些类的预测上容易犯错，例如第P行第F列的取值为25，说明模型有25次将“P”字符错误地识别为“F”字符。直观来看，“P”和“F”相似度比较高，对它们的区分也更具有挑战性。 现在，让我们来通过这个来计算模型在测试集中的预测正确率。
agreement = letters_test["letter"] == letters_pred
print(agreement.value_counts())
print("Accuracy:", metrics.accuracy_score(letters_test["letter"], letters_pred))
# Output: 可见，我们的初步模型在6000个测试样本中，正确预测5068个，整体正确率（Accuaray）为84.47%。


# 模型性能提升 
# 对于支持向量机，有两个主要的参数能够影响模型的性能：一是核函数的选取，二是惩罚参数C的选择。 下面，我们期望通过分别尝试这两个参数来进一步改善模型的预测性能。

# 1 核函数的选取 
# 在 SVC 中，核函数参数kernel可选值为"rbf"（径向基核函数）、“poly”（多项式核函数）、"sigmoid"（sigmoid核函数）和"linear"（线性核函数）。我们的初始模型选取的是线性核函数，下面我们观察在其他三种核函数下模型正确率的改变。

kernels = ["rbf", "poly", "sigmoid"]
for kernel in kernels:
    letters_model = SVC(C=1, kernel=kernel)
    letters_model.fit(letters_train.iloc[:, 1:], letters_train['letter'])
    letters_pred = letters_model.predict(letters_test.iloc[:, 1:])
    print("kernel = ", kernel, ", Accuracy:",
          metrics.accuracy_score(letters_test["letter"], letters_pred))

    # 从结果可以看到，当选取RBF核函数时，模型正确率由84.47%提高到97.12%。 多项式核函数下模型正确率为94.32%。 sigmoid核函数下模型的正确率只有5左右%。

# 2 惩罚参数C的选取
# 分别测试 $C = 0.01,0.1,1,10,100$时字符识别模型正确率的变化。核函数选取径向基核函数（RBF）。
c_list = [0.01, 0.1, 1, 10, 100]
for C in c_list:
    letters_model = SVC(C=C, kernel="rbf")
    letters_model.fit(letters_train.iloc[:, 1:], letters_train['letter'])
    letters_pred = letters_model.predict(letters_test.iloc[:, 1:])
    print("C = ", C, ", Accuracy:",
          metrics.accuracy_score(letters_test["letter"], letters_pred))

    #可见，当惩罚参数C设置为10和100时，模型正确率进一步提升，分别达到96.21%和96.91%。
    

# 实验结束.
