
import numpy as np
from sklearn.linear_model import LinearRegression

class Regressor:
    def __init__(self, params:dict):
        this.setup(params)

    def setup(self, params:dict):
        raise NotImplementedError('setup')

    def fit(self, x:np.ndarray, y:np.ndarray):
        raise NotImplementedError('fit')

    def predict(self, x:np.ndarray):
        raise NotImplementedError('predict')

    def print_params(self):
        raise NotImplementedError('print_params')

    def generate_plots(self, save_location='.'):
        raise NotImplementedError('generate_plots')

class LinearRegression(Regressor):
    None