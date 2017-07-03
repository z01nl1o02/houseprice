import os,sys,pdb
import numpy as np
import pandas as pd
from scipy.stats import skew
class REGDATA:
    def __init__(self,indir):
        dftrain = pd.read_csv(os.path.join(indir,'train.csv'))
        dftrain['fortrain'] = 1
        dftest = pd.read_csv(os.path.join(indir,'test.csv'))
        dftest['fortrain'] = 0
        dftest['SalePrice'] = 0
        self._df = pd.concat([dftrain,dftest])
    def remove_missing_data(self):
        self._df = self._df.fillna(self._df.mean())
        return
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
        skewed_feats = skewed_feats[ skewed_feats > 0.75].index.tolist()
        try:
            skewed_feats.remove('Id')
            skewed_feats.remove('fortrain')
        except Exception, e:
            pass
        self._df[skewed_feats] = np.log1p(self._df[skewed_feats])
        return
    def selection(self):
        goodfeats = 'Id,SalePrice,OverallQual,GrLivArea,TotalBsmtSF,BsmtFinSF1,GarageCars,YearBuilt,1stFlrSF,OverallCond,KitchenAbvGr,LotArea,YearRemodAdd,2ndFlrSF'.split(',')
        self._df = self._df[goodfeats]
    def add_higher_order(self):
        numericfeats = 'OverallQual,GrLivArea,TotalBsmtSF,BsmtFinSF1,GarageCars,1stFlrSF,KitchenAbvGr,LotArea,2ndFlrSF'.split(',')
        for feat in numericfeats:
            self._df[feat+'_SQ'] = self._df.apply( lambda X:X[feat] ** 2, axis = 1)
            #self._df[feat+'_THIRD'] = self._df.apply( lambda X:X[feat] ** 3, axis = 1)
            self._df[feat+'_SQRT'] = self._df.apply( lambda X: np.sqrt(X[feat]), axis = 1)
    def get_train(self):
        return self._df[ self._df['fortrain'] == 1 ].drop('fortrain',axis=1)
    def get_test(self):
        return self._df[ self._df['fortrain'] == 0 ].drop('fortrain',axis=1)

