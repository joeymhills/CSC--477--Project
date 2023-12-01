import numpy as np
from sklearn import svm, datasets, neighbors, preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

f = open("heart+disease/processed.cleveland.data", "r")



#divide data into features

# Age; sex; chest pain type (angina, abnang, notang, asympt)
# Trestbps (resting blood pres); cholesteral; fasting blood sugar < 120
# (true or false); resting ecg (norm, abn, hyper); max heart rate;
# exercise induced angina (true or false); oldpeak; slope (up, flat, down)
# number of vessels colored (???); thal (norm, fixed, rever). Finally, the
# class is either healthy (buff) or with heart-disease (sick).

data = np.loadtxt("heart+disease/processed.cleveland.data", delimiter=",")
#sort data by last colmumn

data = data[data[:,-1].argsort()]

features = data[:, :-1]

scaler = MinMaxScaler()
data[:, :-1] = scaler.fit_transform(features)

target = data[:,13]

def count_class(num):
    count = 0
    for i in range(len(data)):
        if data[i,-1] == num:
            count += 1
    return count

trainData = []
trainTarget = []
testData = []
testTarget = []


#CODE USED FOR OUTLIER DETECTION
#outliers were manually removed after detection thus this code is no longer needed
'''
lof = LocalOutlierFactor()
yhat = lof.fit_predict(data)
print(yhat)
idx = 0
removeList = []
for curr in yhat:
    if curr == -1:
       removeList.append(idx)
    idx += 1
print(removeList)
'''

#divide data into training and testing sets
#training set is 80% of data, testing set is 20%
 #Class 0 data separation
for i in range(0,126):
    trainData.append(data[i])
    trainTarget.append(target[i])
    curr = data[i]
    print("class 0", data[i,13])
for i in range(127,158):
    testData.append(data[i])
    testTarget.append(target[i])
    print("class 0", data[i,13])
 #Class 1 data separation
for i in range(159, 210):
    trainData.append(data[i])
    trainTarget.append(target[i])
    print("class 1", data[i,13])
for i in range(211, 213):
    testData.append(data[i])
    testTarget.append(target[i])
    print("class 1", data[i,13])
#Class 2 data separation
for i in range(220,240):
    trainData.append(data[i])
    trainTarget.append(target[i])
    print("class 2", data[i,13])
for i in range(241,247):
    testData.append(data[i])
    testTarget.append(target[i])
    print("class 2", data[i,13])
#Class 3 data separation
for i in range(256,273):
    trainData.append(data[i])
    trainTarget.append(target[i])
    print("class 3", data[i,13])
for i in range(274,281):
    testData.append(data[i])
    testTarget.append(target[i])
    print("class 3", data[i,13])
#Class 4 data separation
for i in range(282,289):
    trainData.append(data[i])
    trainTarget.append(target[i])
    print("class 4", data[i,13])
for i in range(290,293):
    testData.append(data[i])
    testTarget.append(target[i])
    print("class 4", data[i,13])





#Training Model with SVM
s = svm.SVC(kernel = 'linear')
s.fit(trainData, trainTarget)
decisions = s.predict(testData)


# Parameter Tuning for KNN n value

#List Hyperparameters that we want to tune.
n_neighbors = list(range(1,30))
#Convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors)
#Create new KNN object
knn_2 = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=10)
#Fit the model
best_model = clf.fit(data, target)
#Print The value of best Hyperparameters
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])


#Using PCA
pca = PCA(n_components=2)
pca.fit(trainData)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

trainDataPCA = pca.transform(trainData)
testDataPCA = pca.transform(testData)

#Training Model with KNN  (UPDATED TO USE PCA)
n_neighbors = 23 # K in KNN classifier
nn = neighbors.KNeighborsClassifier(n_neighbors)
nn.fit(trainDataPCA, trainTarget) #Training is done
pr = nn.predict(testDataPCA) # testing
print(pr)

def accuracy(decisions, testTarget):
    correct = 0
    for i in range(len(decisions)):
        if decisions[i] == testTarget[i]:
            correct += 1
    return correct/len(decisions)

print("accuracy of svm is: ", accuracy(decisions, testTarget))
print("accuracy of knn is: ", accuracy(pr, testTarget))

# CLASSIFICATION REPORT

#y_pred = LogisticRegression(testData)
#print(classification_report(testTarget, y_pred))
#roc_auc_score(testTarget, y_pred)




