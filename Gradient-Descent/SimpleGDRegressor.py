# -*- coding: utf-8 -*-
"""
Custom Gradient Descent Regressor for Simple Linear Regression
"""

import numpy as np
from sklearn.base import BaseEstimator , RegressorMixin

class GDRegressor(BaseEstimator, RegressorMixin):
    def __init__(self , lr = 0.01 , iter = 1000):
      self.m_ = 100
      self.b_ = 0
      self.lr = lr
      self.iter = iter

    def fit(self , x_train , y_train):
      x = np.asarray(x_train).ravel()
      y = np.asarray(y_train).ravel()
      n = len(x)

      for i in range(self.iter):
        residual = y - self.m_ * x - self.b_

        loss_slope_m = (-2/n) * np.sum(x * residual)
        loss_slope_b = (-2/n) * np.sum(residual)

        self.b_ = self.b_ - (self.lr * loss_slope_b)
        self.m_ = self.m_ - (self.lr*loss_slope_m)

      return self

    def predict(self , x_test):
      x1 = np.asarray(x_test).ravel()
      return self.m_ * x1 + self.b_

"""# 1. Benchmark: Sklearn LinearRegression"""

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

X , y = make_regression(n_samples=1000 , n_features=1 , noise=20 , n_informative = 1 , n_targets = 1 , random_state=42)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

model = LinearRegression()
model.fit(X_train , y_train)
y_pred = model.predict(X_test)
print(model.coef_)
print(model.intercept_)
print(r2_score(y_test , y_pred))

"""# 2. Custom GDRegressor"""

obj = GDRegressor(lr = 0.01 ,iter = 1000)
obj.fit(X_train , y_train)
y_pred2 = obj.predict(X_test)

print(obj.b_)
print(obj.m_)
print(r2_score(y_test , y_pred2))

