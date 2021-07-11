import pandas as pd
from sklearn import preprocessing
from configs import *
from config import cols


def age_categories(x):
    """Takes an age as input and returns the age category. It will be applied on Age arrays"""
    if x == 0:
        cat = 0
    elif x <= 18:
        cat = 1
    elif x <= 60:
        cat = 2
    else:
        cat = 3
    return cat


class DataPreparation:
    def __init__(self, df):
        self.df = df
    def filling_data(self, imp):
        array = imp.fit_transform(self.df)
        self.df = pd.DataFrame(array, columns=self.df.columns)
    def feature_engineering(self):
        try:
            le = preprocessing.LabelEncoder()
            self.df['age_category'] = self.df.Age.apply(age_categories)
            self.df['Sex_code'] = le.fit_transform(self.df['Sex'])
            self.df['Pclass_code'] = le.fit_transform(self.df['Pclass'])
            self.df['Embarked_code'] = le.fit_transform(self.df['Embarked'])
            return self.df
        except Exception as e:
            print(f"error : {e}")
    def vectorize_train_set(self):
        X = self.df[cols].values
        y = self.df['Survived'].values.astype('int')
        return X, y
    def vectorize_test_set(self):
        X = self.df[cols].values
        return X

