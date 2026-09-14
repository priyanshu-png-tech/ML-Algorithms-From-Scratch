"""Ridge Regression.ipynb

# Comparing the custom implementation with sklearn Ridge

# Ridge Regression Class
"""

from sklearn.base import BaseEstimator , RegressorMixin

class MyRidge(BaseEstimator , RegressorMixin):
    def __init__(self , alpha = 1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self , x , y):
      x = np.asarray(x)
      y = np.asarray(y)

      x = np.insert(x , 0 , 1, axis = 1)
      I = np.identity(x.shape[1])

      weights = np.linalg.inv(np.dot(x.T , x) + self.alpha * I).dot(x.T).dot(y)

      self.coef_ = weights[1:]
      self.intercept_ = weights[0]
      return self

    def predict(self , x):
      x = np.asarray(x)
      return self.intercept_ + np.dot(x , self.coef_)

"""# Scikit-learn Ridge


"""

from sklearn.linear_model import Ridge , LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np

x,y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size = 0.2 , random_state = 4)

rge = Ridge(alpha = 0.1 , solver = 'cholesky').fit(x_train , y_train)
y_pred = rge.predict(x_test)
print(r2_score(y_test , y_pred) )
print("Coef_ : {}".format(rge.coef_))
print("intercept_ : {}".format(rge.intercept_))

"""# Custom Ridge

"""

myRidge = MyRidge(alpha = 0.1).fit(x_train , y_train)
y_pred = myRidge.predict(x_test)
r2_score(y_test , y_pred )
print("Coef_ : {}".format(myRidge.coef_))
print("intercept_ : {}".format(myRidge.intercept_))

