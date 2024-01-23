## Import necessary libraries
import numpy as np
import pandas as pd
import requests as rq
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## Downloading the <link>Davis.csv</link> file from the specified URL and saving it locally
url = "https://raw.githubusercontent.com/sdukshis/ml-intro/master/datareplaces/Davis.csv"
response = rq.get(url)
with open("Davis.csv", "wb") as file:
    file.write(response.content)

## Prepare dataset by changing the variable type of the gender field from string to int: replacing values 'F' with 1 and 'M' with 0. 
## Load height and weight measurement data from 'Davis.csv' and remove any missing values.
datareplace = pd.read_csv('Davis.csv')
datareplace.replace({'sex': {'F': 1, 'M': 0}}, inplace=True)
datareplace.to_csv('Davis.csv',index=False)
data = pd.read_csv('Davis.csv')
data = data.dropna()
print(data.head())


## 'X' will be the target variable containing height values, 'Y' will be the feature variable containing weight values. 
## Create a `Regressor` object and train it on the acquired data.  Compute the mean squared error and calculating the mean squared error for the test sample.
X = data[['height']].values
Y = data['weight'].values
Regressor = LinearRegression()
Regressor.fit(X, Y)
height = [[180]]
weight = Regressor.predict(height)
rounded_weight = round(weight[0])
print("Predicted weight:", rounded_weight)
Y_pred = Regressor.predict(X)
mse = mean_squared_error(Y, Y_pred)
rounded_mse = np.round(mse)
print('Ошибка:', rounded_mse)

## Build a linear regression line and illustrate the points of the training set
plt.xlabel('weight')
plt.ylabel('height')
plt.scatter(X, Y, color='blue', label='Обучающая выборка')
plt.plot(X, Y_pred, color='red', linewidth=1, label='Прямая регрессии')
plt.legend()
plt.show()

## Expand the number of variables by adding gender and repwt.
## 'X' will be the target variable and will contain height values, 'Y' - a variable containing features - weight value and added gender and repwt values.
## Create a 'Model' object and train it on the augmented data.
X = data['height'].values.reshape(-1, 1)
Y = data[['weight', 'sex', 'repwt']].values
Model = LinearRegression()
Model.fit(X, Y)
Y_pred = Model.predict(X)
mse = mean_squared_error(Y, Y_pred)
rounded_mse = np.round(mse)
print('Ошибка:', rounded_mse)
