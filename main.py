
import numpy as np

from sklearn import linear_model, datasets

##数据
from data_prepare import get_metrics


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
##import matplotlib as mpl
##mpl.rcParams["font.sans-serif"] = ["SimHei"]
##mpl.rcParams["axes.unicode_minus"] = False

raw_data = np.genfromtxt('data.csv', dtype="float", skip_header = 0, delimiter="," )
data = raw_data[1:,2:]

##划分数据集
trainLen = int(data.shape[0]*0.8)#训练集和测试集8:2
np.random.shuffle(data)
trainData = data[:trainLen, :-1]
trainLabel = data[:trainLen,-1]
testData = data[trainLen+1:, :-1]
testLabel = data[trainLen+1:, -1]

##############数据标准化################
scaler = MinMaxScaler()
scaler.fit(trainData)
trainData = scaler.transform(trainData)
testData = scaler.transform(testData)


########################################################################
######训练模型
rf = RandomForestClassifier().fit(trainData,trainLabel)#随机森林
lr = LogisticRegression().fit(trainData,trainLabel)#逻辑回归
knn = KNeighborsClassifier().fit(trainData,trainLabel)#K近邻
########################################################################


########################################################################
####预测结果及性能展示

print('RF算法预测分类指标：')
predict_result = rf.predict(testData) #预测结果
get_metrics(true_labels=testLabel,predicted_labels=predict_result)#性能指标展示

sns.set()
f,ax=plt.subplots()
c_rf = confusion_matrix(testLabel, predict_result, labels=[0,1])
print(c_rf)



print('LR算法预测分类指标：')
predict_result = lr.predict(testData) #预测结果
get_metrics(true_labels=testLabel,predicted_labels=predict_result)#性能指标展示

sns.set()
f,ax=plt.subplots()
c_lr = confusion_matrix(testLabel, predict_result, labels=[0,1])
print(c_lr)



print('KNN算法预测分类指标：')
predict_result = knn.predict(testData) #预测结果
get_metrics(true_labels=testLabel,predicted_labels=predict_result)#性能指标展示

sns.set()
f,ax=plt.subplots()
c_knn = confusion_matrix(testLabel, predict_result, labels=[0,1])
print(c_knn)


