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

warnings.filterwarnings('ignore')  # ����ʾwarning��Ϣ
pd.set_option('display.max_columns', None)  # ��ʾ������
plt.rcParams['font.sans-serif'] = ['SimHei']  # ����������ʾ���ı�ǩ
plt.rcParams['axes.unicode_minus'] = False  # ����������ʾ����

df = pd.read_csv('bank_all.csv', sep=';')  # ��ȡ������ݼ�����ѵ��,���Կ�����Ҳ׼���˱��õ����ݼ�

'''
# print(df.shape)  # �鿴���ݼ���ά��: (45211, 17)
# print(df.head())  # �鿴���ݼ���ǰ5��
# print(df.tail())  # �鿴���ݼ��ĺ�5��
# print(df.columns) # �鿴���ݼ�������
# print(df.index) # �鿴���ݼ�������
# print(df.info())  # �鿴���ݼ��Ļ�����Ϣ: ��׸��
# print(df.isnull().any())  # �鿴���ݼ���ֵ�Ƿ�Ϊ��: ��ȱʧֵ

# print(df.dtypes) # �鿴���ݼ�����������,�б�����:

# �󲿷���object���ͣ�������Ҫת������ֵ����
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

# print(df['y'].value_counts())  # �鿴���ݼ��ı�ǩ�ֲ�

# age           10.618762
# balance     3044.765829
# day            8.322476
# duration     257.527812
# campaign       3.098021
# pdays        100.128746
# previous       2.303441
# dtype: float64'''

'''# �ֶα�ǩ˵����
# 'age����', 'jobְҵ', 'marital����״��', 'education����', 'defaultΥԼ״��', 'balance���', 'housing����',
# 'loan����', 'contact��ϵ����', 'day�����ϵ��', 'month�����ϵ�·�', 'duration��ϵʱ��', 'campaign��ϵ����', 
# 'pdays��ϵ������','previous֮ǰ��ϵ������', 'poutcome֮ǰ�Ľ��',
# 'y��������?'''

# !��һ��:�����ݷ���&Ԥ����
# �����׼���Сֵ�����ֵ,������ͳ����Ϣ,���������ݿ��ӻ�
bankdescribe = df.describe()
bankstd = df.std()
bankmin = df.min()
bankmax = df.max()
df.hist(figsize=(10, 10))
plt.show()

# ���Ƚ�������ݼ����Ϊѵ�����Ͳ��Լ�(yes/no)
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ��Ԥ��֮ǰ��Ҫ�����ݼ��е�object����ת��Ϊ��ֵ����, ת������:��object����ת��Ϊ��ֵ����,������ֵ����ת��Ϊone-hot����

# ?������һЩ�ظ��ķ���,����ɺ�����
def SetDownPlz():
    global y_train, y_test, X_train, X_test, y_dummies_train, y_dummies_test
    # 1.��object����ת��Ϊ��ֵ����
    # 1.1 ��jobת��Ϊ��ֵ����
    job_mapping = {'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4, 'management': 5, 'retired': 6,
                   'self-employed': 7, 'services': 8, 'student': 9, 'technician': 10, 'unemployed': 11, 'unknown': 12}
    X_train['job'] = X_train['job'].map(job_mapping)
    X_test['job'] = X_test['job'].map(job_mapping)
    # 1.2 ��maritalת��Ϊ��ֵ����
    marital_mapping = {'divorced': 1, 'married': 2, 'single': 3, 'unknown': 4}
    X_train['marital'] = X_train['marital'].map(marital_mapping)
    X_test['marital'] = X_test['marital'].map(marital_mapping)
    # 1.3 ��educationת��Ϊ��ֵ����
    education_mapping = {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 4}
    X_train['education'] = X_train['education'].map(education_mapping)
    X_test['education'] = X_test['education'].map(education_mapping)
    # 1.4 ��defaultת��Ϊ��ֵ����
    default_mapping = {'no': 1, 'yes': 2, 'unknown': 3}
    X_train['default'] = X_train['default'].map(default_mapping)
    X_test['default'] = X_test['default'].map(default_mapping)
    # 1.5 ��housingת��Ϊ��ֵ����
    housing_mapping = {'no': 1, 'yes': 2, 'unknown': 3}
    X_train['housing'] = X_train['housing'].map(housing_mapping)
    X_test['housing'] = X_test['housing'].map(housing_mapping)
    # 1.6 ��loanת��Ϊ��ֵ����
    loan_mapping = {'no': 1, 'yes': 2, 'unknown': 3}
    X_train['loan'] = X_train['loan'].map(loan_mapping)
    X_test['loan'] = X_test['loan'].map(loan_mapping)
    # 1.7 ��contactת��Ϊ��ֵ����
    contact_mapping = {'cellular': 1, 'telephone': 2, 'unknown': 3}
    X_train['contact'] = X_train['contact'].map(contact_mapping)
    X_test['contact'] = X_test['contact'].map(contact_mapping)
    # 1.8 ��monthת��Ϊ��ֵ����
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5,
                     'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
                     'nov': 11, 'dec': 12}
    X_train['month'] = X_train['month'].map(month_mapping)
    X_test['month'] = X_test['month'].map(month_mapping)
    # 1.9 ��poutcomeת��Ϊ��ֵ����
    poutcome_mapping = {'failure': 1, 'other': 2, 'success': 3, 'unknown': 4}
    X_train['poutcome'] = X_train['poutcome'].map(poutcome_mapping)
    X_test['poutcome'] = X_test['poutcome'].map(poutcome_mapping)
    # 1.10 ��yת��Ϊ��ֵ����
    y_mapping = {'no': 1, 'yes': 2}
    y_train = y_train.map(y_mapping)
    y_test = y_test.map(y_mapping)
    # 2.����ֵ����ת��Ϊone-hot����
    # 2.1 ��jobת��Ϊone-hot����
    job_dummies_train = pd.get_dummies(X_train['job'], prefix='job')
    job_dummies_test = pd.get_dummies(X_test['job'], prefix='job')
    X_train = pd.concat([X_train, job_dummies_train], axis=1)
    X_test = pd.concat([X_test, job_dummies_test], axis=1)
    X_train.drop('job', axis=1, inplace=True)
    X_test.drop('job', axis=1, inplace=True)
    # 2.2 ��maritalת��Ϊone-hot����
    marital_dummies_train = pd.get_dummies(X_train['marital'], prefix='marital')
    marital_dummies_test = pd.get_dummies(X_test['marital'], prefix='marital')
    X_train = pd.concat([X_train, marital_dummies_train], axis=1)
    X_test = pd.concat([X_test, marital_dummies_test], axis=1)
    X_train.drop('marital', axis=1, inplace=True)
    X_test.drop('marital', axis=1, inplace=True)
    # 2.3 ��educationת��Ϊone-hot����
    education_dummies_train = pd.get_dummies(X_train['education'], prefix='education')
    education_dummies_test = pd.get_dummies(X_test['education'], prefix='education')
    X_train = pd.concat([X_train, education_dummies_train], axis=1)
    X_test = pd.concat([X_test, education_dummies_test], axis=1)
    X_train.drop('education', axis=1, inplace=True)
    X_test.drop('education', axis=1, inplace=True)
    # 2.4 ��defaultת��Ϊone-hot����
    default_dummies_train = pd.get_dummies(X_train['default'], prefix='default')
    default_dummies_test = pd.get_dummies(X_test['default'], prefix='default')
    X_train = pd.concat([X_train, default_dummies_train], axis=1)
    X_test = pd.concat([X_test, default_dummies_test], axis=1)
    X_train.drop('default', axis=1, inplace=True)
    X_test.drop('default', axis=1, inplace=True)
    # 2.5 ��housingת��Ϊone-hot����
    housing_dummies_train = pd.get_dummies(X_train['housing'], prefix='housing')
    housing_dummies_test = pd.get_dummies(X_test['housing'], prefix='housing')
    X_train = pd.concat([X_train, housing_dummies_train], axis=1)
    X_test = pd.concat([X_test, housing_dummies_test], axis=1)
    X_train.drop('housing', axis=1, inplace=True)
    X_test.drop('housing', axis=1, inplace=True)
    # 2.6 ��loanת��Ϊone-hot����
    loan_dummies_train = pd.get_dummies(X_train['loan'], prefix='loan')
    loan_dummies_test = pd.get_dummies(X_test['loan'], prefix='loan')
    X_train = pd.concat([X_train, loan_dummies_train], axis=1)
    X_test = pd.concat([X_test, loan_dummies_test], axis=1)
    X_train.drop('loan', axis=1, inplace=True)
    X_test.drop('loan', axis=1, inplace=True)
    # 2.7 ��contactת��Ϊone-hot����
    contact_dummies_train = pd.get_dummies(X_train['contact'], prefix='contact')
    contact_dummies_test = pd.get_dummies(X_test['contact'], prefix='contact')
    X_train = pd.concat([X_train, contact_dummies_train], axis=1)
    X_test = pd.concat([X_test, contact_dummies_test], axis=1)
    X_train.drop('contact', axis=1, inplace=True)
    X_test.drop('contact', axis=1, inplace=True)
    # 2.8 ��monthת��Ϊone-hot����
    month_dummies_train = pd.get_dummies(X_train['month'], prefix='month')
    month_dummies_test = pd.get_dummies(X_test['month'], prefix='month')
    X_train = pd.concat([X_train, month_dummies_train], axis=1)
    X_test = pd.concat([X_test, month_dummies_test], axis=1)
    X_train.drop('month', axis=1, inplace=True)
    X_test.drop('month', axis=1, inplace=True)
    # 2.9 ��poutcomeת��Ϊone-hot����
    poutcome_dummies_train = pd.get_dummies(X_train['poutcome'], prefix='poutcome')
    poutcome_dummies_test = pd.get_dummies(X_test['poutcome'], prefix='poutcome')
    X_train = pd.concat([X_train, poutcome_dummies_train], axis=1)
    X_test = pd.concat([X_test, poutcome_dummies_test], axis=1)
    X_train.drop('poutcome', axis=1, inplace=True)
    X_test.drop('poutcome', axis=1, inplace=True)
    # 2.10 ��yת��Ϊone-hot����
    # ��yת��Ϊone-hot����
    y_dummies_train = pd.get_dummies(y_train, prefix='y')
    y_dummies_test = pd.get_dummies(y_test, prefix='y')


def SeeseemyHoneyisGood():
    pass
    # ת�����,�鿴ת���������:û����
    # print('ת��������ݣ�')
    # print(X_train.head())
    # print(X_test.head())
    # print(y_dummies_train.head())
    # print(y_dummies_test.head())
    # �������Ƿ��Ӧ:��Ӧ��
    # print('�������Ƿ��Ӧ��')
    # print(X_train.shape)
    # print(y_dummies_train.shape)
    # print(X_test.shape)
    # print(y_dummies_test.shape)


SetDownPlz()
# SeeseemyHoneyisGood()

# !�ڶ���:ʹ��C4.5��Ӫ���������Ԥ��
# C4.5��һ�־������㷨������ID3�㷨�ĸĽ��棬���ĺ���˼��������Ϣ��Ϊ׼��ѡ������������Ϣ�����Ϊ׼������������֣��Եݹ�ķ�ʽ���ɾ�������


clf = DecisionTreeClassifier(criterion='entropy')  # ����������������
clf.fit(X_train, y_dummies_train)  # ���ѵ����

y_pred = clf.predict(X_test)  # �Բ��Լ�����Ԥ��

accuracy = clf.score(X_test, y_dummies_test)  # ����ģ�͵�׼ȷ��acc= 87.5%
print('ģ�͵�׼ȷ�ԣ�', accuracy)

# ͨ�����ӻ�ͼ��չʾԤ����:

# 1.���ƻ�������
cm = confusion_matrix(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))  # �����������
print('��������', cm)

# 2.����ROC����
fpr, tpr, thresholds = roc_curve(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))  # ���������ʺ���ȷ��
roc_auc = auc(fpr, tpr)  # ����AUC,AUC��Area Under Curve��������ΪROC��������������Χ�ɵ����
print('AUC��', roc_auc)  # AUC=0.87

lw = 2  # �߿�=2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # ����ROC����
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # �����Խ���
plt.xlim([0.0, 1.0])  # ����x��������
plt.ylim([0.0, 1.05])  # ����y��������
plt.xlabel('������')  # ����x���ǩ
plt.ylabel('��ȷ��')  # ����y���ǩ
plt.title('Ԥ��ģ�͵�ROC����')  # ���ñ���
plt.legend(loc="lower right")  # ����ͼ��λ��
plt.show()  # ��ʾͼ��

# !������,ʹ��SVM��Ӫ���������Ԥ��
# SVM��һ�ֶ�����ģ�ͣ����Ļ���ģ���Ƕ����������ռ��ϵļ���������Է�������������ʹ���б��ڸ�֪����SVM�������˼��ɣ���ʹ����Ϊʵ���ϵ������Է�������
# SVM�ĵ�ѧϰ���Ծ��Ǽ����󻯣�����ʽ��Ϊһ�����͹���ι滮�����⣬Ҳ�ȼ������򻯵ĺ�ҳ��ʧ��������С������,��ѧϰ�㷨�������͹���ι滮�����Ż��㷨��


clf = svm.SVC(kernel='linear', C=1)  # ����SVM������

y_dummies_train = np.array(y_dummies_train).ravel()  # y_dummies_train��һ����ά���飬��Ҫת����һά����
X_train = np.resize(X_train, (360,))  # �����36000���360
y_dummies_train = np.resize(y_dummies_train, (360,))  # �����36000���360
X_train = X_train.reshape(-1, 1)  # ����ֻ�ܽ�����һ��������������Ҫreshapeһ��
y_dummies_train = y_dummies_train.reshape(-1, 1)  # ����ֻ�ܽ�����һ��������������Ҫreshapeһ��

# �������ܲ���,�޷���ԭ�������¼���ѵ��,�ȴ�15����ֻ�ܷ���;�������SVMģ��ֻ����Сѵ����
# keeps Err : X has 51 features, but SVC is expecting 1 features as input ������ʼ���޷����


clf.fit(X_train, y_dummies_train)  # ���ѵ����

y_pred = clf.predict(X_test)  # ����ѵ�����Ľ���Բ��Լ�����Ԥ��

accuracy = clf.score(X_test, y_dummies_test)  # ����׼ȷ�����
print('ģ�͵�׼ȷ�ԣ�', accuracy)

# ���ƻ�������
cm = confusion_matrix(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))
print('��������', cm)

# AUCͼ��
fpr, tpr, thresholds = roc_curve(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))  # ���������ʺ���ȷ��
roc_auc = auc(fpr, tpr)  # ����AUC,AUC��Area Under Curve��������ΪROC��������������Χ�ɵ����
print('AUC��', roc_auc)  # AUC=0.87

# !���Ĳ�,ʹ��adaboost��Ӫ���������Ԥ��

# adaboost��һ�ֵ����㷨�������˼�������ͬһ��ѵ����ѵ����ͬ�ķ�������Ȼ�����Щ����������������
# ���µ����ݽ��з���ʱ��ÿ�����������Ը����ݽ��з��࣬�����ݷ�������ͶƱ�������жϡ�

clf = AdaBoostClassifier(n_estimators=100)  # ����adaboost������

y_dummies_train = np.array(y_dummies_train).ravel()  # y_dummies_train��һ����ά���飬��Ҫת����һά����
X_train = np.resize(X_train, (360,))  # �����36000���360
y_dummies_train = np.resize(y_dummies_train, (360,))
X_train = X_train.reshape(-1, 1)  # ����ֻ�ܽ�����һ��������������Ҫreshapeһ��
y_dummies_train = y_dummies_train.reshape(-1, 1)  # ����ֻ�ܽ�����һ��������������Ҫreshapeһ��

# �޷����������: ValueError: X has 51 features, but AdaBoostClassifier is expecting 1 features as input.
# ʵ������ʧ��

clf.fit(X_train, y_dummies_train)  # ���ѵ����
y_pred = clf.predict(X_test)  # ����ѵ�����Ľ���Բ��Լ�����Ԥ��
accuracy = clf.score(X_test, y_dummies_test)  # ����׼ȷ��
print('ģ�͵�׼ȷ�ԣ�', accuracy)

# չʾԤ����: ���ƻ�������

cm = confusion_matrix(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))
print('��������', cm)
fpr, tpr, thresholds = roc_curve(y_dummies_test.values.argmax(axis=1), y_pred.argmax(axis=1))  # ���������ʺ���ȷ��
roc_auc = auc(fpr, tpr)
print('AUC��', roc_auc)

# ͼ��...

# !���岽,�ȽϽ��������

# �������һ��ģ�ͷ���,�����޷����Part3,4����Ŀ��,���޷����бȽϷ���.ͬʱ,���ڻ�����������,˫��ʹ�õ���������ͬ,�޷����бȽϷ���.
# ����ʵ������ʧ��,����ѧϰ���˺ܶ�֪ʶ,�ṩ��ѵ���Ŀ�ܺ�˼·, ���������һ��ѵ���ķ���.��Ҳ����һ���ջ��.
