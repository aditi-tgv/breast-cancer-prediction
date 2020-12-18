import numpy as np
import pandas as pd
from pandas import DataFrame

from csv import reader
from random import seed
from random import randrange
import matplotlib.pyplot as plt
from math import sqrt
from math import exp
from math import pi


def crossvalidate(dataset, kfold):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / kfold)
	for _ in range(kfold):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


def findaccuracy(actual,predicted):
	accuracy=0
	p=float(len(actual))
	for i in range(len(actual)):
		if actual[i]==predicted[i]: #Correct Prediction
			accuracy=accuracy+1
	return accuracy/p*100

def predict(info,testrow):
	predictedlabel, maximum = None, 0  # Initialise to 0
	probabilities=classwiseprobability(info, testrow) #Get prob row belongs to each class
	for label,probability in probabilities.items():
		if predictedlabel is None or probability>maximum: #Prob it belongs to particular label is maximum
			maximum=probability
			predictedlabel=label
	return predictedlabel


def accuracy_metrics(ytrue, ypred):
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

		if (max(fp)==0):
			fpr = [0] * (len(ypred)+1)
		else:
			fpr = [i / max(fp) for i in fp]
		if (max(tp)==0):
			tpr = [0] * (len(ypred)+1)
		else:
			tpr = [i / max(tp) for i in tp]

	return tpr, fpr

def splitforCV(dataset,algorithm,kfold,*args):
	folds = crossvalidate(dataset, kfold)   #CROSS VALIDATION
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold) #Removing one-kth of train data for CV
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold: #removed portions becomes the test data
			row_copy=list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted=algorithm(train_set, test_set, *args) #Calling NB
		actual=[row[-1] for row in fold]
		accuracy=findaccuracy(actual, predicted)
		scores.append(accuracy) #Store accuracies of 10 models
		tpr,fpr= accuracy_metrics(actual,predicted)
		plt.plot(fpr,tpr)
	return scores


def splitclasswise(dataset):
	classes=dict() #maintain dictionary with key as each class value
	for i in range(len(dataset)):
		vector=dataset[i]
		classvalue=vector[-1] #last column
		if (classvalue not in classes):
			classes[classvalue]=list()
		classes[classvalue].append(vector)
	return classes

def pdinfo(dataset):
	info=[(mean(column),stddev(column),len(column)) for column in zip(*dataset)]
	del(info[-1])
	return info

def mean(numbers):
	return sum(numbers)/len(numbers)


def stddev(numbers):
	return np.std(numbers, dtype=np.float64)


def grouplabelwise(dataset):
	classes=splitclasswise(dataset)
	labels=dict()
	for i, rows in classes.items():
		labels[i]=pdinfo(rows) #for each class find PDF parameters
	return labels


def probability(x,mean,stddev):
	return (1/(sqrt(2*pi)*stddev))*(exp(-((x-mean)**2/(2*stddev**2)))) #gaussian prob dist


def classwiseprobability(info,row):
	total_rows=sum([info[label][0][2] for label in info])
	probabilities=dict() #For each class value maintain probability
	for label, class_summaries in info.items():
		probabilities[label] = info[label][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean,stddev,_=class_summaries[i]
			probabilities[label] *= probability(row[i],mean,stddev)
	return probabilities


def naivebayesclassifier(train,test):
	info=grouplabelwise(train) #Get PDF info such as mean and stddev
	output=list()
	for row in test:
		output.append(predict(info,row)) #Substitute new data in PDF
	return(output)

if __name__ == '__main__':
	df = pd.read_csv('./Cleaned_data.csv')
	dataset=df.values.tolist()
	for i in range(len(dataset[0])-1):
		for row in dataset:
			row[i]=float(row[i])
	kfold=20
	scores=splitforCV(dataset,naivebayesclassifier,kfold)
	print('Scores: %s' % scores)
	p=len(scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(p)))
