import numpy as np
from sklearn import svm, datasets, neighbors
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
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


age = data[:,0]
sex = data[:,1]
chest_pain = data[:,2]
resting_blood_pres = data[:,3]
cholesteral = data[:,4]
fasting_blood_sugar = data[:,5]
resting_ecg = data[:,6]
max_heart_rate = data[:,7]
exercise_induced_angina = data[:,8]
oldpeak = data[:,9]
slope = data[:,10]
num_vessels_colored = data[:,11]
thal = data[:,12]

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

#divide data into training and testing sets
#training set is 80% of data, testing set is 20%

 #Class 0 data separation
for i in range(0,129):
    trainData.append(data[i])
    trainTarget.append(target[i])
    curr = data[i]
for i in range(130,163):
    testData.append(data[i])
    testTarget.append(target[i])
 #Class 1 data separation
for i in range(164, 207):
    trainData.append(data[i])
    trainTarget.append(target[i])
for i in range(208, 219):
    testData.append(data[i])
    testTarget.append(target[i])
#Class 2 data separation
for i in range(220,248):
    trainData.append(data[i])
    trainTarget.append(target[i])
for i in range(249,255):
    testData.append(data[i])
    testTarget.append(target[i])
#Class 3 data separation
for i in range(256,284):
    trainData.append(data[i])
    trainTarget.append(target[i])
for i in range(285,290):
    testData.append(data[i])
    testTarget.append(target[i])
#Class 4 data separation
for i in range(291,299):
    trainData.append(data[i])
    trainTarget.append(target[i])
for i in range(300,303):
    testData.append(data[i])
    testTarget.append(target[i])


# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(trainData)
print(yhat)
idx = 0
removeList = []
for curr in yhat:
    if curr == -1:
       removeList.append(idx)
    idx += 1
#Just need to remove the elements at indices found in removeList

#for curr in removeList:
#    trainData.pop(curr)


#Training Model with SVM
s = svm.SVC(kernel = 'linear')
s.fit(trainData, trainTarget)
decisions = s.predict(testData)

#Training Model with KNN
n_neighbors = 7 # K in KNN classifier
nn = neighbors.KNeighborsClassifier(n_neighbors)
nn.fit(trainData, trainTarget) #Training is done
pr = nn.predict(testData) # testing
print(pr)

def accuracy(decisions, testTarget):
    correct = 0
    for i in range(len(decisions)):
        if decisions[i] == testTarget[i]:
            correct += 1
    return correct/len(decisions)

print("accuracy of svm is: ", accuracy(decisions, testTarget))


print("accuracy of knn is: ", accuracy(pr, testTarget))



