import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.core.frame import DataFrame
from numpy import ndarray

column_names: list = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
boston: DataFrame = pd.read_csv("./data/housing.csv", header=None, delimiter=r"\s+", names=column_names)
column_selected: list = ["LSTAT", "INDUS", "NOX", "PTRATIO", "RM", "TAX", "DIS", "AGE", "RAD"]

top: int = 0.8*len(boston.index)
training: DataFrame = boston.loc[:top, :]
test: DataFrame = boston.loc[top:, :]

#independent: list = list(set(column_names).difference(set(["MEDV"])))
#x: ndarray =  training.loc[:, independent].to_numpy()
x: ndarray =  training.loc[:, column_selected].to_numpy()
y: ndarray = training.loc[:,"MEDV"].to_numpy()

mlr: LinearRegression = LinearRegression(n_jobs=-1)
mlr.fit(x,y)

# Results
r_sq = mlr.score(x, y)
print(f"Coefficient of determination: {r_sq}")
print(f"Intercept: {mlr.intercept_}")
print(f"Coefficients: {mlr.coef_}")

# Forecast
#y_pred = mlr.predict(test.loc[:, independent].to_numpy())
y_pred = mlr.predict(test.loc[:, column_selected].to_numpy())
y_actual = test.loc[:, "MEDV"].to_numpy()

print(f"Predicted response:\n{y_pred}")

# Comparison
print(f"Residual sum of squares is : {np.sum(np.square(y_pred - y_actual))}")
