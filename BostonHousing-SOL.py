# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:33:21 2018

@author: farzaad
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as rr
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.metrics import r2_score

df = pd.read_csv("BostonHousing-DS.csv")
df.drop(df.columns[0],axis=1,inplace=True)

plt.figure(figsize=(15, 16))
heat = sns.heatmap(df[list(df)].corr(),annot=True,)
plt.show()

#train = df.drop("medv",axis=1).values
# for ls in list(df):
X = df.drop("medv",axis=1)
y = df[["medv"]].values

X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3)
print(X)
print(y)
model = rr()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
    
    # poly = pf()
    # X_poly = poly.fit_transform(X_train) 
    # poly.fit(X_poly,y_train)
    # lin = lr()
    # lin.fit(X_poly,y_train)
    
    # #plot
    # print("COLUMN:",ls)
    # plt.scatter(X,y,c="b")
    # plt.plot(X_test,lin.predict(poly.fit_transform(X_test)),"r")
    # plt.show()
    
    # #Score
    # COD = r2_score(y_test,lin.predict(poly.fit_transform(X_test)))
    # print("R2 SCORE:",COD)
    
    # #Classifiers
    # R2_dict={}
    # mse_dict={}
    # algos = [SVR(), rr()]
        
    # for clf in algos:
    #     clf.fit(X_train,y_train.ravel())
    #     R2_dict[str(clf)[:3]] = clf.score(X_test,y_test)
    #     mse_dict[str(clf)[:3]] = mse(X_test,y_test) 
        
    #     for x,y in R2_dict.items():
    #         print(x,y)
    #     for x,y in mse_dict.items():
    #         print(x,y)
    

        