
import numpy as np
from sklearn.linear_model import LinearRegression

class Regressor:
    def __init__(self, params:dict):
        self.setup(params)

    def setup(self, params:dict):
        raise NotImplementedError('setup')

    def fit(self, x:np.ndarray, y:np.ndarray):
        raise NotImplementedError('fit')

    def predict(self, x:np.ndarray):
        raise NotImplementedError('predict')

    def get_learned_features(self):
        raise NotImplementedError('get_learned_features')

    def generate_plots(self, save_location='.'):
        raise NotImplementedError('generate_plots')

class Model_LinearRegression(Regressor):
    def __init__(self, params:dict):
        super().__init__(params)
        self._model = None
        self._fitted_model = None

    def setup(self, params:dict):
        print('Setup')
        self._model = LinearRegression()
        print(self._model)

    def fit(self, x:np.ndarray, y:np.ndarray):
        self._fitted_model = self._model.fit(x, y)

    def predict(self, x:np.ndarray) -> np.ndarray:
        return self._fitted_model.predict(x)

    def get_learned_features(self) -> np.ndarray:
        if self._fitted_model:
            return np.array([self._fitted_model.coef_, 
                             self._fitted_model.intercept_])
    
    def generate_plots(self, save_location='.'):
        pass