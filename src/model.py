from sklearn.linear_model import LinearRegression
import numpy as np


class MyModel(LinearRegression):
    def __init__(self):
        super().__init__()


class BaseModel:
    def __init__(self):
        self.mean = 0

    def fit(self, x, y):
        self.mean = np.mean(y)

    def predict(self, x):
        return [self.mean] * len(x)
