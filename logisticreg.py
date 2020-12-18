import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def sigmoid(x):
    sig=1/(1+np.exp(-x))
    return sig

def prediction(weight, x):
    label=[]
    prod=np.dot(weight,x)
    hyp=sigmoid(prod)
    for i in range(0,len(hyp[0])):
        if (hyp[0][i]>0.5):
            label.append(1)
        elif (hyp[0][i]<= 0.5):
            label.append(-1)
    return label

def logistic(X_train,X_test,y_train,y_test,alpha):
    w = np.random.randn(1, 5) #5 atrributes and 5 weights
    for i in range(1, 1500): #100 epochs
        prod=np.dot(w,X_train)
        y_pred=prediction(w,X_train) #-1 or +1
        val=-np.multiply(y_train, prod)
        J=np.sum(np.log(1+np.exp(val)))
        numerator=-np.multiply(y_train,np.exp(val))
        denominator=1+np.exp(val)
        res=numerator/denominator
        gradient=np.dot(X_train,res.T)
        w=w-alpha*gradient.T #update weight
        print("Epoch", i, "Loss", J, "Training Accuracy: ", accuracy_score(y_train[0], y_pred) * 100)
    testprediction=prediction(w, X_test)
    print("Test Accuracy :", accuracy_score(y_test[0],testprediction)*100)

if __name__ == '__main__':
    df = pd.read_csv('./Cleaned_data_withheading.csv')
    df = df.drop(labels='Age', axis=1) #dropping column age due to missing values
    X = np.array(df)[:, :-1]
    y = np.array(df)[:, -1:]
    y[y <= 0] = -1

    p = len(X)
    ones = np.ones((p, 1)) #adding column of ones for bias
    X = np.column_stack((X, ones))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    X_train = np.transpose(X_train)
    X_test = np.transpose(X_test)
    y_train = np.transpose(y_train)
    y_test = np.transpose(y_test)

    alpha = 0.008 #learning rate
    logistic(X_train, X_test, y_train, y_test, alpha)