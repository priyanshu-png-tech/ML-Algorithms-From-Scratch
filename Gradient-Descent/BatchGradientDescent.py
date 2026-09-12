# -*- coding: utf-8 -*-
"""Batch_GD.ipynb

#1. Custom Batch Gradient Descent Class
"""

from sklearn.base import BaseEstimator , RegressorMixin
import numpy as np

class BatchGDRegressor(BaseEstimator , RegressorMixin):
  def __init__(self , epochs = 100 , learning_rate = 0.01):
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.coef_ = None
    self.intercept_ = None

  def fit(self , x_train , y_train):
    x = np.asarray(x_train)
    y = np.asarray(y_train)

    self.intercept_ = 0
    self.coef_ = np.ones(x.shape[1])

    for i in range(self.epochs):
      y_hat = np.dot(x , self.coef_)+ self.intercept_
      intercept_derivative = -2 * np.mean(y - y_hat)
      self.intercept_ = self.intercept_ - (self.learning_rate * intercept_derivative)

      coef_der = -2 * np.dot((y - y_hat) , x)/x.shape[0]
      self.coef_ = self.coef_ - (self.learning_rate * coef_der)


    return self

  def predict(self , x_test):
    x_test = np.asarray(x_test)
    return np.dot(x_test , self.coef_) + self.intercept_

"""# 2. Sklearn LinearRegression (Benchmark)

"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

X , y = load_diabetes(return_X_y = True , as_frame = True)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 2)


lr = LinearRegression()
lr.fit(X_train , y_train)
y_pred = lr.predict(X_test)

print(lr.coef_)
print(lr.intercept_)
print(r2_score(y_test , y_pred))

"""# 3. Custom Batch GD Regressor


"""

obj = BatchGDRegressor(epochs=1000 , learning_rate=0.9)
obj.fit(X_train , y_train)

print(obj.coef_)
print(obj.intercept_)
print(r2_score(y_test , obj.predict(X_test)))