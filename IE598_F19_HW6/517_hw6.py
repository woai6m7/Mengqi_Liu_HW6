#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 12:35:38 2020

@author: liumengqi
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from time import process_time 

#Part 1: Random test train splits
df=pd.read_csv('/Users/liumengqi/Desktop/ccdefault.csv')

df.dropna(inplace=True)
df.describe()

X, y = df[df.columns[1:-1]].values, df['DEFAULT'].values

train_scores=[]
test_scores=[]
start= time.process_time()
for i in range(1,11):
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.1,stratify=y,random_state=i)
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    tree = DecisionTreeClassifier(random_state=1)
    tree.fit(X_train, y_train)
    y_train_pred=tree.predict(X_train)
    y_test_pred=tree.predict(X_test)
    train_scores.append(metrics.accuracy_score(y_train,y_train_pred))
    test_scores.append(metrics.accuracy_score(y_test,y_test_pred))
end= time.process_time()
print("train accuracy score:\n{}".format(train_scores))
print("\n")
print("mean of train accuracy score:\n{} ".format(np.mean(train_scores)))
print("\n")
print("standard deviation of train accuracy score:\n{}".format(np.std(train_scores)))
print("\n")
print("test accuracy score:\n{}".format(test_scores))
print("\n")
print("mean of test accuracy score:\n{} ".format(np.mean(test_scores)))
print("\n")
print("standard deviation of test accuracy score:\n{}".format(np.std(test_scores)))
print("\n")
print("running time: \n{}".format(end-start))
print("\n")
      
#Part 2: Cross validation
start1= time.process_time()
tree = DecisionTreeClassifier()
scores = cross_val_score(estimator=tree,
                             X=X_train,
                             y=y_train,
                             cv=10,
                            n_jobs=-1)
end1= time.process_time()
print('CV accuracy scores: %s' % scores)
print("\n")
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
print("\n")
print("running time: \n{}".format(end1-start1))
print("\n")
print("My name is {Mengqi Liu}")
print("My NetID is: {mengqi3}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
