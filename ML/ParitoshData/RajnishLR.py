import numpy as np
import pandas as pd

data_frame = pd.read_csv('mlr04.csv')
ones = np.ones(len(data_frame["X1"].values))
X = np.array([ones, data_frame["X1"].values, data_frame["X2"].values, data_frame["X3"].values])

Y = np.array(data_frame["X4"].values)
step1 = np.matmul(X,X.T)
step2 = np.linalg.inv(step1)
step3 = np.matmul(step2,X)
step4 = np.matmul(step3,Y)
print(step4)
x1, x2, x3= 85.09999847,8.5,5.099999905


p = np.matmul(step4.T,X[:,:])
print(p)
print(np.sqrt(((p - Y) ** 2).mean()))