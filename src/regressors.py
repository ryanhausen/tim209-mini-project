
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


class Regressor:
    def __init__(self, params:dict):
        self.params = params
        self.setup(params)

    def setup(self, params:dict):
        raise NotImplementedError('setup')

    def fit(self, x:np.ndarray, y:np.ndarray):
        raise NotImplementedError('fit')

    def predict(self, x:np.ndarray):
        raise NotImplementedError('predict')

    def get_learned_features(self):
        raise NotImplementedError('get_learned_features')

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

    def get_name(self):
        return 'Linear Regression'


class Model_KnnRegression(Regressor):
    def __init__(self, params:dict):
        super().__init__(params)

    def setup(self, params:dict):
        self._model = KNeighborsRegressor(n_neighbors=params['n'])

    def fit(self, x:np.ndarray, y:np.ndarray):
        self._fitted_model = self._model.fit(x, y)

    def predict(self, x:np.ndarray) -> np.ndarray:
        return self._fitted_model.predict(x)

    def get_learned_features(self) -> np.ndarray:
        if self._fitted_model:
            return np.array([])

    def get_name(self):
        return 'K Nearest Neighbors: k={}'.format(self.params['n'])


class Model_DecisionTree(Regressor):
    def __init__(self, params:dict):
        super().__init__(params)

    def setup(self, params:dict):
        self._model = DecisionTreeRegressor()

    def fit(self, x:np.ndarray, y:np.ndarray):
        self._fitted_model = self._model.fit(x, y)

    def predict(self, x:np.ndarray) -> np.ndarray:
        return self._fitted_model.predict(x)

    def get_learned_features(self) -> np.ndarray:
        if self._fitted_model:
            return np.array([])

    def get_name(self):
        return 'Decision Tree'

class Model_SupportVectorMachine(Regressor):
    def __init__(self, params:dict):
        super().__init__(params)

    def setup(self, params:dict):
        self._model = SVR()

    def fit(self, x:np.ndarray, y:np.ndarray):
        self._fitted_model = self._model.fit(x, y)

    def predict(self, x:np.ndarray) -> np.ndarray:
        return self._fitted_model.predict(x)

    def get_learned_features(self) -> np.ndarray:
        if self._fitted_model:
            return np.array([])

    def get_name(self):
        return 'Support Vector Machine'