import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from quilt.data.uciml import abalone, forestfires, wine_quality

class Abalone:
    """UCI - Abalone Dataset
    Attribute Information:

    Given is the attribute name, attribute type, the measurement unit and a
    brief description. The number of rings is the value to predict: either as a
    continuous value or as a classification problem.

    Name / Data Type / Measurement Unit / Description
    -----------------------------
    Sex / nominal / -- / M, F, and I (infant)
    Length / continuous / mm / Longest shell measurement
    Diameter / continuous / mm / perpendicular to length
    Height / continuous / mm / with meat in shell
    Whole weight / continuous / grams / whole abalone
    Shucked weight / continuous / grams / weight of meat
    Viscera weight / continuous / grams / gut weight (after bleeding)
    Shell weight / continuous / grams / after being dried
    Rings / integer / -- / +1.5 gives the age in years

    """
    @staticmethod
    def get_raw_XY() -> (np.ndarray, np.ndarray):
        df = abalone.tables()
        return (df.iloc[:,:-1].values, df.iloc[:,-1].values)

    @staticmethod
    def get_preprocessed_XY() -> (np.ndarray, np.ndarray):
        data = abalone.tables()
        # Convert Sex from categorical to binary

        sexes = LabelBinarizer().fit_transform(data['Sex'])
        data.drop(columns=['Sex'], inplace=True)
        for i, f in enumerate('FIM'):
            data['Sex_{}'.format(f)] = sexes[:, i]

        columns = ['Sex_F', 'Sex_I', 'Sex_M', 'Length', 'Diameter', 'Height',
                   'Whole weight', 'Shucked weight', 'Viscera weight',
                   'Shell weight']

        return (data.loc[:, columns].values, data.loc[:, 'Rings'].values)

    @staticmethod
    def get_name() -> str:
        return 'Abalone'

class ForestFires:
    """UCI - Forest Fires Dataset
    Attribute Information:

    For more information, read [Cortez and Morais, 2007].
    1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
    2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
    3. month - month of the year: 'jan' to 'dec'
    4. day - day of the week: 'mon' to 'sun'
    5. FFMC - FFMC index from the FWI system: 18.7 to 96.20
    6. DMC - DMC index from the FWI system: 1.1 to 291.3
    7. DC - DC index from the FWI system: 7.9 to 860.6
    8. ISI - ISI index from the FWI system: 0.0 to 56.10
    9. temp - temperature in Celsius degrees: 2.2 to 33.30
    10. RH - relative humidity in %: 15.0 to 100
    11. wind - wind speed in km/h: 0.40 to 9.40
    12. rain - outside rain in mm/m2 : 0.0 to 6.4
    13. area - the burned area of the forest (in ha): 0.00 to 1090.84
    (this output variable is very skewed towards 0.0, thus it may make
    sense to model with the logarithm transform).
    """

    @staticmethod
    def get_raw_XY() -> (np.ndarray, np.ndarray):
        df = forestfires.tables()
        x = df.iloc[1:,:-1].values
        y = df.iloc[1:,-1].values
        return (x, y)

    @staticmethod
    def get_preprocessed_XY() -> (np.ndarray, np.ndarray):
        data = forestfires.tables()

        # month and day need to be turned into numeric values they are cyclical
        # so we need the sin/cos
        # http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
        encoded = LabelEncoder().fit_transform(data['month'].values)
        data['month_sin'] = np.sin(encoded * (2*np.pi/12))
        data['month_cos'] = np.cos(encoded * (2*np.pi/12))

        encoded = LabelEncoder().fit_transform(data['day'].values)
        data['day_sin'] = np.sin(encoded * (2*np.pi/12))
        data['day_cos'] = np.cos(encoded * (2*np.pi/12))

        # actual data column is 'MC' not 'DC'
        columns = ['X', 'Y', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                   'FFMC', 'DMC', 'MC', 'ISI', 'temp', 'RH', 'wind', 'rain']

        x = data.loc[:, columns].values[1:,:].astype(np.float64)

        # use the log of area as recommended in the dataset description
        y = data['area'].values[1:].astype(np.float64)
        y = np.log10(y, where=y>0)

        return (x, y)

    @staticmethod
    def get_name() -> str:
        return 'Forest Fires'

class WineQuality:
    """UCI - Wine Quality
    Attribute Information:

    For more information, read [Cortez et al., 2009].
    Input variables (based on physicochemical tests):
    1 - fixed acidity
    2 - volatile acidity
    3 - citric acid
    4 - residual sugar
    5 - chlorides
    6 - free sulfur dioxide
    7 - total sulfur dioxide
    8 - density
    9 - pH
    10 - sulphates
    11 - alcohol
    Output variable (based on sensory data):
    12 - quality (score between 0 and 10)
    """

    @staticmethod
    def get_raw_XY() -> (np.ndarray, np.ndarray):
        data = wine_quality.tables()

        x = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values

        return (x, y)

    @staticmethod
    def get_preprocessed_XY() -> (np.ndarray, np.ndarray):
        data = wine_quality.tables()

        x = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values

        return (x, y)

    @staticmethod
    def get_name() -> str:
        return 'Wine Qualtiy'
