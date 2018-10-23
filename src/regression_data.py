import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from quilt.data.uciml import abalone, forestfires, wine_quality

class Abalone:
    @staticmethod
    def get_raw_XY() -> (pd.DataFrame, pd.DataFrame):
        df = abalone.tables()
        return (df.iloc[:,:-1], df.ilocp[:,-1])

    @staticmethod 
    def get_preprocessed_XY() -> (pd.DataFrame, pd.DataFrame):
        data = abalone.tables()
        # Convert Sex from categorical to binary
        sexes = LabelBinarizer().fit_transform(data['Sex'])
        data.drop(columns=['Sex'], inplace=True)     
        for i, f in enumerate('FIM'):
            data['Sex_{}'.format(f)] = sexes[:, i]

        columns = ['Sex_F', 'Sex_I', 'Sex_M', 'Length', 'Diameter', 'Height', 
                   'Whole weight', 'Shucked weight', 'Viscera weight', 
                   'Shell weight']

        return (data.loc[:, columns], data.loc[:, 'Rings'])



