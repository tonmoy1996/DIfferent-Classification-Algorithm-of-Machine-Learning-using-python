# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:41:05 2019

@author: sahat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

dataset= pd.read_csv('housing.csv')
print(dataset.head())
print(dataset.info())
print(dataset.columns)
sns.pairplot(dataset)
sns.distplot(dataset['Price'])
sns.heatmap(dataset.corr())
X = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y=dataset['Price']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=101)

linearreg=LinearRegression()

linearreg.fit(X_train,y_train)

# Model Evaluation finding coefficient


print(linearreg.intercept_)

coeffdataset=pd.DataFrame(linearreg.coef_,X.columns,columns=['Coefficient'])
print(coeffdataset)

#prediction 

predict= linearreg.predict(X_test)

plt.scatter(y_test,predict,color = 'green')

sns.distplot((y_test-predict),bins=50)

#Regression Evaluation Metrics

from sklearn import metrics

MAE=metrics.mean_absolute_error(y_test,predict)

print(MAE)
MSE=metrics.mean_squared_error(y_test,predict)
print(MSE)
RMSE=np.sqrt(metrics.mean_squared_error(y_test,predict))

print(RMSE)





