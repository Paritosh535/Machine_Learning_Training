# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:11:45 2019

@author: Abhishek Sharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#plot charts

dataset = pd.read_csv('Employment.csv')

X = np.array(dataset.iloc[:,:-1])
Y = np.array(dataset.iloc[:,-1])

#Spliting data set into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, 
                                                    random_state =0)

#Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting Test set results
y_pred = regressor.predict(x_test)

#Visualising traing sets results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training Sets)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualising test sets results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test Sets)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Calculating RMSE and R2 Score
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y[:5], y_pred)
#rmse = np.sqrt(mse)
r2_score = regressor.score(X, Y) # scores are btw 0 and 1 , larger score indicating  a better fit

print(np.sqrt(mse))
print(r2_score)