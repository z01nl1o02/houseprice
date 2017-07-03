import os,sys,pdb
import numpy as np
import pandas as pd
from scipy.stats import skew
class REGDATA:
    def __init__(self,path):
        self._df = pd.read_csv(path)
    def remove_missing_data(self):
        total = self._df.isnull().sum().sort_values(ascending=False)
        percent = (self._df.isnull().sum() / self._df.isnull().count()).sort_values(ascending=False)
        md = pd.DataFrame({'total':total,'percent':percent})
        idx = (md[md['total']>1]).index
        self._df = self._df.drop(idx,1)
        return
    def one_hot_encoding(self):
        self._df = pd.get_dummies(self._df)
        return
    def remove_skewing(self):
        numeric_feats = self._df.dtypes[ self._df.dtypes != 'object'].index
        skewed_feats = self._df[numeric_feats].apply(lambda x:skew(x.dropna()))
        skewed_feats = skewed_feats[ skewed_feats > 0.75].index
        self._df[skewed_feats] = np.log1p(self._df[skewed_feats])
        return
    def add_buildlife(self):
        self._df['buildlife'] = self._df['YrSold'] - self._df['YearBuilt']
        return
    def get_data(self):
        return self._df
