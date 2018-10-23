import regressors as reg
import regression_data as reg_data

def main():
    for data in [reg_data.Abalone]:
        x, y = data.get_preprocessed_XY()
        for model in [reg.Model_LinearRegression]:
            m = model(dict())
            m.fit(x, y)
    
if __name__=='__main__':
    main()

