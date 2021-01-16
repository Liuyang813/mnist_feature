
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
# Import `train_test_split`
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
# Load in the `digits` data
from sklearn.preprocessing import scale
# iris = datasets.load_iris()
data = pd.read_csv('data.csv', header=None)
#data_0 = pd.read_csv('data_0.csv', header=None)
# split the data up - 3/4 for training, 1/4 for testing
X =np.array(data.iloc[:,1:])
Y =np.array(data.iloc[:,:1])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=532)

# Number of training features
# n_samples, n_features = data_train.shape

scaler = StandardScaler()
scaler.fit(x_train)
params_train_scaled = scaler.transform(x_train)
params_test_scaled = scaler.transform(x_test)

# 1 hidden layer, same size as the input layer
mlp = MLPClassifier(
    solver='lbfgs',
    hidden_layer_sizes=(X.shape[1], 100),
    random_state=0)
mlp.fit(params_train_scaled, y_train)

print(y_train)
print('Train score: %.3g' % mlp.score(params_train_scaled, y_train))
print('Test Score: %.3g' % mlp.score(params_test_scaled, y_test))
# print

# test_val =np.array(data_0.iloc[11:12,1:])


# print(mlp.predict(test_val))
