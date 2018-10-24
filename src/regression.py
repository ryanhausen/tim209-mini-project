import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import regressors as reg
import regression_data as reg_data

def main():
    models = [reg.Model_LinearRegression(dict()),
              reg.Model_KnnRegression({'n':5}),
              reg.Model_DecisionTree(dict()),
              reg.Model_SupportVectorMachine(dict())]

    for data in [reg_data.Abalone, reg_data.ForestFires, reg_data.WineQuality]:
        x, y = data.get_preprocessed_XY()
        kf = KFold(n_splits=10)

        print('==============================')
        print(data.get_name())
        results = []
        for model in models:

            mse, r2 = [], []

            for train_idx, test_idx in kf.split(x):
                x_train, y_train = x[train_idx], y[train_idx]
                x_test, y_test = x[test_idx], y[test_idx]

                model.fit(x_train, y_train)

                y_hat = model.predict(x_test)

                mse.append(mean_squared_error(y_test, y_hat))
                r2.append(r2_score(y_test, y_hat, multioutput=None))

            name = model.get_name()
            mse = np.mean(mse)
            r2 = np.mean(r2)

            results.append((name, {'mse':mse,'r2':r2}))

            print('Report for {}'.format(name))
            print('MSE: {}'.format(mse))
            print('R2: {}'.format(r2))
        print('==============================')

        # https://matplotlib.org/gallery/statistics/barchart_demo.html
        bar_width = 0.35
        idx = np.arange(len(results))
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(idx,
               list(map(lambda a:a[1]['mse'], results)),
               bar_width,
               label='MSE')

        ax.bar(idx+bar_width,
               list(map(lambda a:a[1]['r2'], results)),
               bar_width,
               label='$R^2$')

        ax.set_xlabel('Classifier')
        ax.set_ylabel('Values')
        ax.set_title('MSE and $R^2$ for Classifiers on {}'.format(data.get_name()))
        ax.set_xticks(idx + bar_width / 2)
        ax.set_xticklabels(list(map(lambda a: a[0], results)))
        ax.legend()

        fig.tight_layout()
        plt.savefig('./{}_comparison.png'.format(data.get_name()))


if __name__=='__main__':
    main()

