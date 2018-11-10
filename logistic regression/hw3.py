import math
import numpy
import matplotlib.pyplot as plt
import operator
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import random

#return w of logistic regression
def LR(x, y, stepSize, eps):
    #add constant term into x
    column = numpy.ones(x.shape[0])
    x = numpy.column_stack((x, column))

    #initiate w
    w = numpy.random.random((x.shape[1],1))
    
    #initiate d
    d = numpy.zeros((x.shape[1],1))
    #compute d
    for i in range(x.shape[0]):
        e = math.exp(x[i].dot(w))
        d += (y[i]*x[i] - x[i]*e/(1+e)).reshape(-1,1)

    #update w
    while d.T.dot(d)>eps:
        #update w
        w += stepSize * d
        
        #initiate d
        d = numpy.zeros((x.shape[1],1))
        LL = 0
        #compute d
        for i in range(x.shape[0]):
            e = math.exp(x[i].dot(w))
            d += (y[i]*x[i] - x[i]*e/(1+e)).reshape(-1,1)
        
    return w


#compute the predicted probability
def PredictProbability(x, w):
    #the predicted probability
    p = numpy.zeros((x.shape[0],1))
    
    #add constant term into x
    column = numpy.ones(x.shape[0])
    x = numpy.column_stack((x, column))

    #compute the predicted probability
    for i in range(x.shape[0]):
        e = math.exp(x[i].dot(w))
        p[i] = e/(1+e)

    return p


#plot ROC curve
def ROCplot(predictedProbability, y):
    roc = []

    #compute FPR and TPR
    #iterate through the Thresholds
    for i in range(y.size):
        TP=0
        FP=0
        TN=0
        FN=0
        #compute the FPR and TPR for this Threshold
        for j in range(y.size):
            if predictedProbability[j]<predictedProbability[i]:
                if y[j]==0:
                    TN += 1
                else:
                    FN += 1
            else:
                if y[j]==1:
                    TP += 1
                else:
                    FP += 1    

        roc.append([FP/(FP+TN), TP/(TP+FN)])
    
    #add FPR and TPR with Threshold=1
    TP=0
    FP=0
    TN=0
    FN=0
    for j in range(y.size):
        if predictedProbability[j]<1:
            if y[j]==0:
                TN += 1
            else:
                FN += 1
        else:
            if y[j]==1:
                TP += 1
            else:
                FP += 1

    roc.append([FP/(FP+TN), TP/(TP+FN)])

    #sort the (FPR, TPR)
    roc.sort(key=operator.itemgetter(0,1))
    roc=numpy.array(roc)

    #draw the graph
    plt.plot(roc[:,0], roc[:,1])
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


#compute accuracy
def Accuracy(predictedProbability, y, threshold):
    TP=0
    TN=0
    for i in range(y.size):
        if predictedProbability[i]<threshold:
            if y[i]==0:
                TN += 1
        else:
            if y[i]==1:
                TP += 1

    return (TP+TN)/y.size


#load data
print ("Loading data...")
data = load_breast_cancer()
x = data.data
y = data.target

#normalize inputs
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

#split data into training and testing sets
print ("Preparing for both training and testing datasets...")
index = [i for i in range(y.size)]
random.shuffle(index)

xTrain = []
yTrain = []
xTest = []
yTest = []
for i in range(int(y.size/3)):
    xTest.append(x[index[i]])
    yTest.append(y[index[i]])

xTest = numpy.array(xTest).reshape(-1, x.shape[1])
yTest = numpy.array(yTest).reshape(-1, 1)

for i in range(int(y.size/3), y.size):
    xTrain.append(x[index[i]])
    yTrain.append(y[index[i]])

xTrain = numpy.array(xTrain).reshape(-1, x.shape[1])
yTrain = numpy.array(yTrain).reshape(-1, 1)

#train the model
print ("Training the model...")
w = LR(xTrain, yTrain, 0.1, 0.0049)
#predict the probability
yPredict = PredictProbability(xTest, w)
print ("Accuracy with threshold = 50%: ", Accuracy(yPredict, yTest, 0.5))
#generate ROC curve
ROCplot(yPredict, yTest)
