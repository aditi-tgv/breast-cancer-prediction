import numpy as np
import pandas as pd
import random
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix


def entropy(x):
    unique_elements,count = np.unique(x,return_counts=True)
    h = 0
    for c in count:
        h += -(c/len(x))*np.log2(c/len(x))
    
    return h
	
def assign_label(p,r):
    res = 0
    prob = 0
    
    for i in range(len(p)):
        res += r[i]
        prob += p[i]*(1-p[i])
    if prob == 0:
        return res
    else:
        return res/prob

def partition(x,y,p,r,attr_value_pair):
    (_,c) = np.shape(x)
    a,v = attr_value_pair
    xTrue = np.empty((1,c),dtype = int)
    xFalse = np.empty((1,c),dtype = int)
    yTrue = np.empty((1,1),dtype = int)
    yFalse = np.empty((1,1),dtype = int)
    pTrue = np.empty((1,1))
    pFalse = np.empty((1,1))
    rTrue = np.empty((1,1))
    rFalse = np.empty((1,1))
    for i in range(len(x)):
        if x[:,a][i] == v:
            xTrue = np.vstack((xTrue,x[i]))
            yTrue = np.vstack((yTrue,y[i]))
            pTrue = np.vstack((pTrue,p[i]))
            rTrue = np.vstack((rTrue,r[i]))
        else:
            xFalse = np.vstack((xFalse,x[i]))
            yFalse = np.vstack((yFalse,y[i]))
            pFalse = np.vstack((pFalse,p[i]))
            rFalse = np.vstack((rFalse,r[i]))  
            
    return xTrue[1:,:],yTrue[1:,:],xFalse[1:,:],yFalse[1:,:],pTrue[1:,:],pFalse[1:,:],rTrue[1:,:],rFalse[1:,:]

def normalized_residual(x,r,v):
    norm_r = 0
    c = 0
    for i in range(len(x)):
        if x[i] == v:
            norm_r += abs(r[i])
            c += 1
    return float(norm_r/c)

def attributeValuePairs(x):
    
    r,c =  np.shape(x)
    attr_val = []
    for i in range(c):
        unique_elements = np.unique(x[:,i])
        for v in unique_elements:
            attr_val.append((i,v))
        
    return attr_val

def make_tree(x,y,p,r,depth = 0,max_depth = 5):
    
    d = depth
    m = max_depth
    
    if (entropy(y) == 0):
        return assign_label(p,r)
    else:
        if(d == m):
            return assign_label(p,r)
        else:
            
            d += 1
            dec_tree = {}
            attr_value = attributeValuePairs(x)
            norm_residual = []
            for a,v in attr_value:
                norm_residual.append(normalized_residual(x[:,a],r,v))
            
            attribute,value = attr_value[norm_residual.index(max(norm_residual))]
            
            xTrue,yTrue,xFalse,yFalse,pTrue,pFalse,rTrue,rFalse = partition(x,y,p,r,(attribute,value))
            dec_tree[(attribute,value,False)] = make_tree(xFalse,yFalse,pFalse,rFalse,d,m)
            dec_tree[(attribute,value,True)] = make_tree(xTrue,yTrue,pTrue,rTrue,d,m)
            
            return dec_tree


def predict_example(x, tree):

    for k in tree:
        a,v = k[:2]
        
    if(x[a] == v):
        if(type(tree[(a,v,True)]) == dict):
            return predict_example(x,tree[(a,v,True)])
        else:
            return tree[(a,v,True)]
    else:
        if(type(tree[(a,v,False)]) == dict):
            return predict_example(x,tree[(a,v,False)])
        else:
            return tree[(a,v,False)]	

def update_predictions(x,p,initial_bias,learning_rate,trees):
    p_new = []
    for i in range(len(x)):
        v = 0
        log_odds = 0
        for tree in trees:
            v += learning_rate * predict_example(x[i],tree)
        log_odds = (initial_bias + v)
        prob = np.exp(log_odds)/(1+np.exp(log_odds))
        if prob == 0:
            prob = 1/(1+len(x))
        p_new.append(prob)
        
    return p_new

def update_residuals(y,p,r):
    r_new = []
    for i in range(len(y)):
        r_new.append(y[i] - p[i])

    return r_new

def final_prediction(x,initial_bias,learning_rate,trees):

    y_predicted = []
    for i in range(len(x)):
        v = 0
        log_odds = 0
        for tree in trees:
            v += learning_rate * predict_example(x[i],tree)
        log_odds = (initial_bias + v)
        prob = np.exp(log_odds)/(1+np.exp(log_odds))
        if(prob > 0.5):
            y_predicted.append(1)
        else:
            y_predicted.append(0)


    return y_predicted
	
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

def grad_boost(x,y,learning_rate,num_trees = 100,max_depth = 5):

    m = max_depth
    H_ens = []
    y_unique,y_count = np.unique(y,return_counts = True)
    initial_log_odds = np.log(y_count[1]/y_count[0])
    prob = np.exp(initial_log_odds)/(1+np.exp(initial_log_odds))  
    pred = np.full((len(x),1),prob)
    residual = np.empty((len(x),1))
    for i in range(len(x)):
        residual[i] = y[i] - pred[i]
    
    
    for i in range(num_trees):
        H = make_tree(x,y,pred,residual,max_depth = m)
        H_ens.append(H)
        pred = update_predictions(x,pred,initial_log_odds,learning_rate,H_ens)
        residual = update_residuals(y,pred,residual)

    return initial_log_odds,H_ens
	
	
if __name__ == '__main__':
    
    df = pd.read_csv('./Cleaned_data_withheading.csv')
    df = df.drop(labels = 'Age', axis = 1)
    X = np.array(df)[:,:-1]
    y = np.array(df)[:,-1]

    cv = ShuffleSplit(n_splits=10,test_size = 0.15)
    for num_trees in range(10,101,10):
        legends = []
        graph_title = "Gradient boosting \n"+"Number of trees " + str(num_trees)
        plt.figure(num_trees/10)
        plt.title(graph_title)
        accuracy = []
        precision = []
        recall = []
        for i, (train, test) in enumerate(cv.split(X, y)):
            learning_rate = 0.75
            log_odds,H = grad_boost(X[train],y[train],learning_rate,num_trees,max_depth = 3)
            y_pred = final_prediction(X[test],log_odds,learning_rate,H)
            #print(accuracy_score(y[test], y_pred)*100)
            accuracy.append(100*accuracy_score(y[test], y_pred))
            (tp,fp),(fn,tn) = confusion_matrix(y[test],y_pred)
            precision.append(100*tp/(tp+fp))
            recall.append(100*tp/(tp+fn))
            tpr,fpr = accuracy_metrics(y[test],y_pred)
            plt.plot(fpr,tpr)
            legends.append("Iteration " + str(i+1))
        plt.legend(legends)
        #plt.savefig(fname = "Gradient boosting "+str(num_trees))
        print("Number of trees %2d" %(num_trees))
        print("Average accuracy: %4.2f, Average precision: %4.2f, Average recall: %4.2f" %((sum(accuracy)/len(accuracy)),(sum(precision)/len(precision)),(sum(recall)/len(recall))))
        print("Maximum accuracy: %4.2f, Maximum precision: %4.2f, Maximum recall: %4.2f" %(max(accuracy),max(precision),max(recall)))
        print()