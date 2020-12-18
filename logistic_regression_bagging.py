import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
from sklearn.model_selection import ShuffleSplit

def sigmoid(x):
    sig=1/(1+np.exp(-x))
    return sig
	
def predict_example(w,x):
    prod=np.dot(w,x)
    hyp=sigmoid(prod)
    if (hyp > 0.5):
        return 1
    else:
        return -1 

def logistic(X_train,y_train,alpha,num_epochs):
    w = np.random.randn(1, 5) #5 atrributes and 5 weights
    for i in range(1, num_epochs): #100 epochs
        prod=np.dot(w,X_train)
        y_pred=[predict_example(w,x) for x in X_train.T] #-1 or +1
        val=-np.multiply(y_train, prod)
        J=np.sum(np.log(1+np.exp(val)))
        numerator=-np.multiply(y_train,np.exp(val))
        denominator=1+np.exp(val)
        res=numerator/denominator
        gradient=np.dot(X_train,res.T)
        w=w-alpha*gradient.T #update weight
        #print("Epoch", i, "Loss", J, "Training Accuracy: ", accuracy_score(y_train[0], y_pred) * 100)
    return w

def predict_ensemble(W,x):
    y_pred = []
    for i in range(len(x)):
        y_temp = []
        for w in W:
            y_temp.append(predict_example(w,x[i]))
        y_pred.append(max(set(y_temp), key=y_temp.count))
    return y_pred 

def create_subset(x,y,subset_indices):
    
    r,c = np.shape(x)
    x_subset = np.empty((1,c),dtype = int)
    y_subset = np.empty((1,1),dtype = int)
    
    for i in subset_indices:
        x_subset = np.vstack((x_subset,x[i]))
        y_subset = np.vstack((y_subset,y[i]))
        
    return x_subset[1:,:],y_subset[1:,:]

def bagging(x,y,num_learners,num_epochs):
    W_ensemble = []
    for i in range(num_learners):
        set_indices = random.choices(range(len(x)),k = len(x))
        x_set,y_set = create_subset(x,y,set_indices)
        W = logistic(x.T,y.T,0.001,num_epochs)
        W_ensemble.append(W)
    return W

def pre_processing(x,y):

    y[y <= 0] = -1
    p = len(x)
    ones = np.ones((p, 1))
    x = np.column_stack((x, ones))

    return x,y

def accuracy_metrics(ytrue,ypred):

    tp = [0]
    fp = [0]
    
    for i in range(len(ytrue)):
        if (ypred[i] == 1):
            if (ytrue[i] == 1):
                y = tp[-1] + 1
                tp.append(y)
                fp.append(fp[-1])
            else:
                y = fp[-1] + 1
                fp.append(y)
                tp.append(tp[-1])
        else:
            tp.append(tp[-1])
            fp.append(fp[-1])
    if(max(tp) == 0):
        tpr = [0]*(len(tp))
    else:    
        tpr = [i/max(tp) for i in tp]
        
    if(max(fp) == 0):
        fpr = [0]*len(fp)
    else:    
        fpr = [i/max(fp) for i in fp]

    return tpr,fpr

if __name__ == '__main__':
    df = pd.read_csv('./Cleaned_data_withheading.csv')
    df = df.drop(labels = 'Age', axis = 1)
    X = np.array(df)[:,:-1]
    y = np.array(df)[:,-1]
    cv = ShuffleSplit(n_splits=10,test_size = 0.15)
    for num_learners in range(10,101,10):  
        legends = []
        num_epochs = 1000
        graph_title = "Bagging: logistic regression \n"+"Number of learners " + str(num_learners)
        plt.figure(num_learners/10)
        plt.title(graph_title)
        accuracy = []
        precision = []
        recall = []
        for i, (train, test) in enumerate(cv.split(X, y)):
            X_train,y_train = pre_processing(X[train],y[train])
            X_test,y_test = pre_processing(X[test],y[test])
            W = bagging(X_train,y_train,10,500)
            y_pred = predict_ensemble(W,X_test)
            accuracy.append(100*accuracy_score(y_test, y_pred))
            (tp,fp),(fn,tn) = confusion_matrix(y_test,y_pred)
            precision.append(100*tp/(tp+fp))
            recall.append(100*tp/(tp+fn))
            tpr,fpr = accuracy_metrics(y_test,y_pred)
            plt.plot(fpr,tpr)
            legends.append("Iteration " + str(i+1))
        plt.legend(legends)
        #plt.savefig(fname = "Logistic regression bagged "+str(num_learners))
        print("Number of learners %2d" %(num_learners))
        print("Average accuracy: %4.2f, Average precision: %4.2f, Average recall: %4.2f" %((sum(accuracy)/len(accuracy)),(sum(precision)/len(precision)),(sum(recall)/len(recall))))
        print("Maximum accuracy: %4.2f, Maximum precision: %4.2f, Maximum recall: %4.2f" %(max(accuracy),max(precision),max(recall)))
        print()