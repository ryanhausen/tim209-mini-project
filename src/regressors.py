
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

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

    def name(self):
        raise NotImplementedError('name')

class Model_LinearRegression(Regressor):
    def __init__(self, params:dict):
        super().__init__(params)

    def setup(self, params:dict):
        self._model = LinearRegression()

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

    def get_name(self):
        return 'Linear Regression'


class Model_KnnRegression(Regressor):
    def __init__(self, params:dict):
        super().__init__(params)

    def setup(self, params:dict):
        self._model = KNeighborsRegressor()

    def fit(self, x:np.ndarray, y:np.ndarray):
        self._fitted_model = self._model.fit(x, y)

    def predict(self, x:np.ndarray) -> np.ndarray:
        return self._fitted_model.predict(x)

    def get_learned_features(self) -> np.ndarray:
        if self._fitted_model:
            return np.array([])

    def generate_plots(self, save_location='.'):
        pass

    def get_name(self):
        return 'K Nearest Neighbors'