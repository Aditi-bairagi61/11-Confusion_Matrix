# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:49:54 2024

@author: HP
"""

#question: for large dataset why we are not using KNN algorithm
#Ans: bcz, of computational complexity etx.
import pandas as pd
import numpy as np
wbcd=pd.read_csv("D:/DS/11-Confusion_Matrix/wbcd.csv")
#there are 569 rows and 12 columns
wbcd.describe()
#in output column there is only B for Benien and M for Malignant
#let us first convert it as Begnien and Malignant
#Begien  =A noncancerous,Malignant =A cancerous
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='B','Beniegn',wbcd['diagnosis'])
#in wbcd there is column named 'diagnosis',where ever there is 'B' replace with 'Beneign'
#similarity where ever there is M in the same column replace with 'Malignant'
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='M','Malignant',wbcd['diagnosis'])
##########################################
#0th column is patient ID let us drop it
wbcd=wbcd.iloc[:,1:32]
########################
#normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#now let us apply his function to the dataframe
wbcd_n=norm_func(wbcd.iloc[:,1:32])
#because now 0th column is output or label it is not considered
################3
#let us now apply X as input and y as output
x=np.array(wbcd_n.iloc[:,:])
#since in wbcd_n we are already excluding output column,hence all rows 
y=np.array(wbcd['diagnosis'])
#Now let us split the data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred
#Now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test))
pd.crosstab(pred,y_test)
#let us check the apllicability of the model
#ie. miss classification, Actual patient is malignant
#i.e.  cancer patient but predicted is Benien is 1
#Actual patient  is Benign and predicted as cancer patient is 5
#Hence this model is not acceptable 
#let us try to select correct value of k

acc = []
#Running KNN algorithm for k=3 to 50 in the step of 2
#K value selected is odd value
for i in range(3,50,2):
    #Declare the model
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    train_acc = np.mean(neigh.predict(X_train)==y_train)
    test_acc = np.mean(neigh.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])
#If you will see the acc, it has got two accuracy, i[0]-train_acc
#i[1]= test_acc
#to plot the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
#there are 3,5,7 and 9 are possible values where accuracy is good let 
#us check for k=3
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
accuracy_score(pred,y_test)
pd.crosstab(pred,y_test)
#i.e. miss classification, Actual patient is malignant