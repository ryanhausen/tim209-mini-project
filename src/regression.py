import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import regressors as reg
import regression_data as reg_data

def main():
    for data in [reg_data.Abalone]:
        metrics = []
        x, y = data.get_preprocessed_XY()
        x, y = x.values, y.values
        kf = KFold(n_splits=10)

        for model in [reg.Model_LinearRegression, reg.Model_KnnRegression]:

            acc, r2 = [], []
            m = model(dict())
            for train_idx, test_idx in kf.split(x):
                x_train, y_train = x[train_idx], y[train_idx]
                x_test, y_test = x[test_idx], y[test_idx]

                m.fit(x_train, y_train)

                y_hat = m.predict(x_test)

                acc.append(mean_squared_error(y_test, y_hat))
                r2.append(r2_score(y_test, y_hat))

            print('Report for {}'.format(m.get_name()))
            print('MSE: {}'.format(np.mean(acc)))
            print('R2: {}'.format(np.mean(r2)))





if __name__=='__main__':
    main()

