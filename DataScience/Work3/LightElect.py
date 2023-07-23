# -*- coding: gbk -*- 
# ʵ��3 ��֧�����������й�ѧ�ַ�ʶ��
# ʹ��UCI�Ĺ�ѧ�ַ�ʶ�����ݼ�
import pandas as pd
from sklearn.svm import SVC

letters = pd.read_csv("letterecognition.csv")
print(letters.head(10))  # չʾǰ10������

# ���ݼ��е�ÿ����������һ����ĸ��ÿ����ĸ����16��������ÿ����������һ������
print('\n\n\n')
print(letters["letter"].value_counts().sort_index())  # ͳ��ÿ����ĸ�ĸ���
print('\n\n\n')  # �ɼ��������ַ������������ֲ���Ծ��⡣���ڣ����ǽ�һ���۲�ÿһ���Ա�����ȡֵ�ֲ���

print(letters.iloc[:, 1:].describe())  # �鿴ÿ���Ա�����ȡֵ�ֲ�
# �۲췢��16���Ա�����ȡֵ��Χ����0~15֮�䣬��˶��ڸ����ݼ����ǲ���Ҫ�Ա������б�׼�������� ���⣬���ݼ������Ѿ�������������У�����Ҳ����Ҫ���Ƕ����ݽ��������ɢ�� �˴�������ֱ��ȡǰ14000��������70%����Ϊѵ��������6000��������30%����Ϊ���Լ���
# ����ѵ�����Ͳ��Լ�
letters_train = letters.iloc[0:14000, ]
letters_test = letters.iloc[14000:20000, ]

# ģ��ѵ��
# ʹ��sklearn.svm���е��������ʵ������������֧���������Ĺ�ѧ�ַ�ʶ��ģ�� ,ѡ�� SVC ������ģ�͹���,SVC ��������Ҫ�Ĳ����������ã��˺������� kernel ��Լ���ͷ�����C

letter_recognition_model = SVC(C=1, kernel="linear")
letter_recognition_model.fit(letters_train.iloc[:, 1:], letters_train['letter'])

# ��������
# ���ȣ�ʹ��predict()�����õ���һ��ѵ����֧��������ģ���ڲ��Լ����ϵ�Ԥ������Ȼ��ʹ�� sklearn.metrics�е���غ�����ģ�͵����ܽ���������
from sklearn import metrics

letters_pred = letter_recognition_model.predict(letters_test.iloc[:, 1:])
print(metrics.classification_report(letters_test["letter"], letters_pred))
print(pd.DataFrame(metrics.confusion_matrix(letters_test["letter"], letters_pred),
                   columns=letters["letter"].value_counts().sort_index().index,
                   index=letters["letter"].value_counts().sort_index().index))

# �������������жԽ��ߵ�Ԫ�ر�ʾģ����ȷԤ�������Խ�Ԫ��֮�ͱ�ʾģ������Ԥ����ȷ���������� ���ǶԽ���Ԫ���ϵ�ֵ����Է�ӳģ������Щ���Ԥ�������׷��������P�е�F�е�ȡֵΪ25��˵��ģ����25�ν���P���ַ������ʶ��Ϊ��F���ַ���ֱ����������P���͡�F�����ƶȱȽϸߣ������ǵ�����Ҳ��������ս�ԡ� ���ڣ���������ͨ�����������ģ���ڲ��Լ��е�Ԥ����ȷ�ʡ�
agreement = letters_test["letter"] == letters_pred
print(agreement.value_counts())
print("Accuracy:", metrics.accuracy_score(letters_test["letter"], letters_pred))
# Output: �ɼ������ǵĳ���ģ����6000�����������У���ȷԤ��5068����������ȷ�ʣ�Accuaray��Ϊ84.47%��


# ģ���������� 
# ����֧������������������Ҫ�Ĳ����ܹ�Ӱ��ģ�͵����ܣ�һ�Ǻ˺�����ѡȡ�����ǳͷ�����C��ѡ�� ���棬��������ͨ���ֱ�����������������һ������ģ�͵�Ԥ�����ܡ�

# 1 �˺�����ѡȡ 
# �� SVC �У��˺�������kernel��ѡֵΪ"rbf"��������˺���������poly��������ʽ�˺�������"sigmoid"��sigmoid�˺�������"linear"�����Ժ˺����������ǵĳ�ʼģ��ѡȡ�������Ժ˺������������ǹ۲����������ֺ˺�����ģ����ȷ�ʵĸı䡣

kernels = ["rbf", "poly", "sigmoid"]
for kernel in kernels:
    letters_model = SVC(C=1, kernel=kernel)
    letters_model.fit(letters_train.iloc[:, 1:], letters_train['letter'])
    letters_pred = letters_model.predict(letters_test.iloc[:, 1:])
    print("kernel = ", kernel, ", Accuracy:",
          metrics.accuracy_score(letters_test["letter"], letters_pred))

    # �ӽ�����Կ�������ѡȡRBF�˺���ʱ��ģ����ȷ����84.47%��ߵ�97.12%�� ����ʽ�˺�����ģ����ȷ��Ϊ94.32%�� sigmoid�˺�����ģ�͵���ȷ��ֻ��5����%��

# 2 �ͷ�����C��ѡȡ
# �ֱ���� $C = 0.01,0.1,1,10,100$ʱ�ַ�ʶ��ģ����ȷ�ʵı仯���˺���ѡȡ������˺�����RBF����
c_list = [0.01, 0.1, 1, 10, 100]
for C in c_list:
    letters_model = SVC(C=C, kernel="rbf")
    letters_model.fit(letters_train.iloc[:, 1:], letters_train['letter'])
    letters_pred = letters_model.predict(letters_test.iloc[:, 1:])
    print("C = ", C, ", Accuracy:",
          metrics.accuracy_score(letters_test["letter"], letters_pred))

    #�ɼ������ͷ�����C����Ϊ10��100ʱ��ģ����ȷ�ʽ�һ���������ֱ�ﵽ96.21%��96.91%��
    

# ʵ�����.
