import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # 线性回归
data = pd.read_csv('data.csv', header=None)
#data.head()
#= data.iloc[:,1:]
# print(data.iloc[:,:1]) # 第一列
# print(data.iloc[:,1:])   # 除了第一列

def bulid_lr():
    X =data.iloc[:,2:]
    y =data.iloc[:,:1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=532)
    linreg = LinearRegression()
    model = linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    print(y_pred)
    y_pred=np.round(y_pred).astype("uint16")
    print(y_pred)
    y_test=np.array(y_test)
    print(y_test)
    print(y_pred==y_test)
    acc_num=list(y_pred==y_test).count(True)
    print(acc_num,len(y_pred))
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel("the number of feature")
    plt.ylabel('class')
    plt.show()
bulid_lr()