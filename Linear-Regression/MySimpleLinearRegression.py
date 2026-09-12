# -*- coding: utf-8 -*-
"""SimpleLinearRegression.ipynb
    
"""

import numpy as np
from sklearn.base import BaseEstimator , RegressorMixin

class MySimpleLinearRegressor(BaseEstimator , RegressorMixin):
  def __init__(self):
    self.m = None
    self.b = None

  def fit(self , x_train , y_train):
    num = 0
    den = 0

    x_mean = x_train.mean()
    y_mean = y_train.mean()

    for i in range(X_train.shape[0]):
      num = num + ((x_train[i] - x_mean) * (y_train[i] - y_mean))
      den = den + ((x_train[i] - x_mean) * (x_train[i] - x_mean))

    self.m = num/den
    self.b = y_mean -(self.m * x_mean)

    return self


  def predict(self , x_test):
    return self.m * x_test + self.b



