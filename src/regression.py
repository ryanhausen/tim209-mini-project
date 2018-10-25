import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import regressors as reg
import regression_data as reg_data

# https://stackoverflow.com/a/10482477
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

def main():
    models = [reg.Model_LinearRegression(dict()),
              reg.Model_KnnRegression({'n':10}),
              reg.Model_DecisionTree(dict()),
              reg.Model_SupportVectorMachine(dict())]

    for data in [reg_data.Abalone, reg_data.ForestFires, reg_data.WineQuality]:
        x, y = data.get_preprocessed_XY()
        kf = KFold(n_splits=10)

        print('==============================')
        print(data.get_name())
        results = []
        
        
        f, a = plt.subplots(nrows=2, ncols=2, figsize=(20,20))
        a = np.array(a).flatten()
        plt.suptitle(data.get_name(), fontsize=24)
        
        for i, model in enumerate(models):

            mse, r2 = [], []

            for train_idx, test_idx in kf.split(x):
                x_train, y_train = x[train_idx], y[train_idx]
                x_test, y_test = x[test_idx], y[test_idx]

                model.fit(x_train, y_train)

                y_hat = model.predict(x_test)

                poly = np.poly1d(np.polyfit(y_hat, y_test, 1))
                
                

                mse.append(mean_squared_error(y_test, y_hat))
                r2.append(r2_score(y_test, y_hat, multioutput=None))
                a[i].scatter(y_hat, y_test, alpha=0.5, color='b')
                a[i].plot([y_hat.min(), y_hat.max()], 
                          poly([y_hat.min(), y_hat.max()]),
                          color='r',
                          alpha=0.5)
                a[i].set_xlabel('Prediction')
                a[i].set_ylabel('Label')
                a[i].set_xlim(y_test.min(), y_test.max())
                a[i].set_ylim(y_test.min(), y_test.max())
      
            
            name = model.get_name()
            mse = np.mean(mse)
            r2 = np.mean(r2)
            a[i].set_title('{}: MSE:{} $R^2$:{}'.format(model.get_name(), mse, r2))


            results.append((name, {'mse':mse,'r2':r2}))

            print('Report for {}'.format(name))
            print('MSE: {}'.format(mse))
            print('R2: {}'.format(r2))
            
            
        f.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig('{}.png'.format(data.get_name()))
        print('==============================')

        mses = list(map(lambda a:a[1]['mse'], results))
        r2s = list(map(lambda a:a[1]['r2'], results))
        # https://matplotlib.org/gallery/statistics/barchart_demo.html
        # https://matplotlib.org/examples/api/two_scales.html
        bar_width = 0.35
        idx = np.arange(len(results))
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(idx,
               mses,
               bar_width,
               color='b')
        ax.set_ylabel('MSE', color='b')
        ax.set_ylim((min(mses + r2s), max(mses+r2s)))
        ax.tick_params('y', colors='b')
        ax.set_xlabel('Classifier')
        ax.set_xticks(idx + bar_width / 2)
        ax.set_xticklabels(list(map(lambda a: a[0], results)))

        ax2 = ax.twinx()
        ax2.bar(idx+bar_width,
               r2s,
               bar_width,
               color='coral')
        ax2.set_ylabel('$R^2$', color='coral')
        ax2.tick_params('y', colors='coral')


        ax.set_title('MSE and $R^2$ for Classifiers on {}'.format(data.get_name()))

        align_yaxis(ax, 0, ax2, 0)
        fig.tight_layout()
        plt.savefig('./{}_comparison.png'.format(data.get_name()))


if __name__=='__main__':
    main()

