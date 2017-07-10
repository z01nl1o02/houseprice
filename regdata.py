import os,sys,pdb
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

class REGDATA:
    def __init__(self,indir):
        dftrain = pd.read_csv(os.path.join(indir,'train.csv'))
        dftrain['fortrain'] = 1
        dftest = pd.read_csv(os.path.join(indir,'test.csv'))
        dftest['fortrain'] = 0
        dftest['SalePrice'] = 0
        # The test example with ID 666 has GarageArea, GarageCars, and GarageType 
        # but none of the other fields, so use the mode and median to fill them in.
        dftest.loc[666, 'GarageQual'] = 'TA'
        dftest.loc[666, 'GarageCond'] = 'TA'
        dftest.loc[666, 'GarageFinish'] = 'Unf'
        dftest.loc[666, 'GarageYrBlt'] = '1980'
        # The test example 1116 only has GarageType but no other information. We'll 
        # assume it does not have a garage.
        dftest.loc[1116, 'GarageType'] = np.nan
        #print len(dftrain), len(dftest), len(dftrain) + len(dftest)
        self._df = pd.concat([dftrain,dftest],ignore_index=True)
        #print len(dftrain), len(dftest), len(dftrain) + len(dftest)
        #print self._df.index
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
        ignore_feats = 'Id,fortrain,has3SsnPorch,hasScreenPorch'.split(',')
        for feat in ignore_feats:
            if feat in set(skewed_feats):
                skewed_feats.remove(feat)
       # print skewed_feats
        self._df[skewed_feats] = np.log1p(self._df[skewed_feats])
        return
    def standandlize(self):
        feats = 'GrLivArea,TotalBsmtSF,BsmtFinSF1,1stFlrSF,LotArea,2ndFlrSF'.split(',')
        scaler = StandardScaler()
        scaler.fit( self._df[feats] )
        scaled = scaler.transform(self._df[feats])
        for k,col in enumerate(feats):
            self._df[col] = scaled[:,k]
        return
    def remove_feature_out_of_test(self):
        targetFeats = []
        dftest = self._df[ self._df['fortrain'] == False]
        for feat in self._df.columns:
            if feat == 'fortrain' or feat == 'SalePrice':
                continue
            vtest = set(list(dftest[feat]))
            if len(vtest) < 2:
                targetFeats.append(feat)
        if len(targetFeats) > 0:
            print targetFeats
            print 'before column drop ', len(self._df.columns)
            for feat in targetFeats:
                self._df.drop(feat, inplace=True,axis=1)
            print 'after column drop ', len(self._df.columns)
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
    def add_new_feats(self):
        self._df['has3SsnPorch'] = self._df.apply(lambda X:X['3SsnPorch'] > 0,axis=1)
        thedict = {None:0,'Unf':1,'BLQ':1,'GLQ':1,"ALQ":1,"LwQ":1,'Rec':1}
        self._df['hasBsmtFinType2'] = self._df['BsmtFinType2'].map(thedict).astype(int)
       # thedict = {"NA":0,'MnPrv':1,'GdWo':1,'GdPrv':1}
       # print self._df['Fence'].head(10)
       # self._df['hasFance'] = self._df['Fence'].map(thedict).astype(int)
        self._df['hasFance'] = self._df.apply(lambda X:X['Fence'] is not None, axis=1)
        #thedict = {None:0,'Gd':1,'TA':1,'Ex':1,'Fa':1}
        #self._df['hasFire'] = self._df['FireplaceQu'].map(thedict).astype(int)
        self._df['hasFire'] = self._df.apply(lambda X:X['FireplaceQu'] is not None, axis=1)
        self._df['hasGradBathRoom'] = self._df.apply( lambda X:X['FullBath'] is not None, axis = 1)

        self._df['highSeason'] = self._df['MoSold'].replace(
                {1:0,2:0,3:0,4:0,5:1,6:1,7:1,8:1,9:0,10:0,11:0,12:0}
                )
                
        area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]
        self._df['TotalArea'] = self._df[area_cols].sum(axis=1)
        
        self._df.loc[self._df.Neighborhood == 'NridgHt', 'Neighborhood_Good'] = 1
        self._df.loc[self._df.Neighborhood == 'Crawfor', 'Neighborhood_Good'] = 1
        self._df.loc[self._df.Neighborhood == 'StoneBr', 'Neighborhood_Good'] = 1
        self._df.loc[self._df.Neighborhood == 'Somerst', 'Neighborhood_Good'] = 1
        self._df.loc[self._df.Neighborhood == 'NoRidge', 'Neighborhood_Good'] = 1
        self._df['Neighborhood_Good'].fillna(0, inplace=True)
        
        self._df['VeryNewHouse'] = (self._df['YearBuilt'] == self._df['YrSold']) * 1
        # House completed before sale or not
        self._df['BoughtOffPlan'] = self._df.SaleCondition.replace(
            {'Abnorml' : 0, 'Alloca' : 0, 'AdjLand' : 0, 'Family' : 0, 'Normal' : 0, 'Partial' : 1})
    
        self._df['hasScreenPorch'] = self._df.apply( lambda X: X['ScreenPorch'] > 0, axis = 1)
        self._df['hasRedmod'] = self._df.apply(lambda X: X['YearBuilt'] != X['YearRemodAdd'],axis=1)
        self._df['buildLife'] = self._df.apply(lambda X: X['YrSold'] - X['YearBuilt'],axis=1)
        self._df['buildLife2'] = self._df.apply(lambda X: X['YrSold'] - X['YearRemodAdd'],axis=1)
        
        self._df['SaleCondition_PriceDown'] = self._df.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 2, 'AdjLand': 3, 'Family': 4, 'Normal': 5, 'Partial': 0})
        
        self._df['Age'] = 2010 - self._df['YearBuilt'] 
        self._df['TimeSinceSold'] = 2010 - self._df['YrSold']
        
        return
    def delete_feats(self):
        self._df = self._df.drop('Alley') #99% NaN
        self._df = self._df.drop('PoolArea') # const var
        self._df = self._df.drop('AllPub') # const var
        return
    def delete_samples(self):
        #print 'before samples deletion: ', len(self._df)
        self._df['del'] = self._df.apply(lambda X: X['fortrain'] == 1 and X['GrLivArea'] > 4000,axis=1)
        idx = self._df[self._df['del'] == True].index
        #print self._df.iloc[idx]['del']
        self._df.drop(idx, inplace=True,axis=0)
        #print 'after samples deletion: ', len(self._df)
        return
    def get_train(self):
        return self._df[ self._df['fortrain'] == 1 ].drop('fortrain',axis=1)
    def get_test(self):
        return self._df[ self._df['fortrain'] == 0 ].drop('fortrain',axis=1)

