import os,sys,pdb
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from cvtcategory import CVT_CATEG
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
        self._df = pd.concat([dftrain,dftest],ignore_index=True)
        #GarageYrBlt should be numeric!
        self._df['GarageYrBlt'].replace("NaN","1980",inplace=True)
        idx = self._df[ self._df['GarageYrBlt'].isnull() ].index
        self._df['GarageYrBlt'].iloc[idx] = '1980'
        self._df['GarageYrBlt'] = np.int64( self._df['GarageYrBlt'])
        
        self._df.drop('Utilities',inplace=True,axis=1)
        self._df.drop('Street',inplace=True,axis=1)
    def save(self,path):
        self._df.to_csv(path,index=False)
        
    def categ2num_and_fillnan(self):
        feats = self._df.columns.tolist()
        categ = CVT_CATEG('categ')
        categ_feats = categ.get_categ_feats()
        for feat in feats:
            if feat in set(categ_feats):
                idx = self._df[ self._df[feat].isnull() ].index
                #self._df.loc[idx,feat] = "EMPYT"
                self._df[feat][idx] = "EMPTY"
            else:
                self._df[feat] = self._df[feat].fillna(self._df[feat].median()) #numeric featus 
        self._df.to_csv("numeric.csv",index=False)
        return
    def dummies(self):
        feats = self._df.columns.tolist()
        categ = CVT_CATEG('categ')
        categ_feats = categ.get_categ_feats()
        categ_feats.remove('Utilities')
        categ_feats.remove('Street')
        self._df = pd.get_dummies(self._df, columns = categ_feats, dummy_na=True)
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
        feats = 'OverallQual,GrLivArea,TotalBsmtSF,BsmtFinSF1,GarageCars,1stFlrSF,KitchenAbvGr,LotArea,2ndFlrSF'.split(',')
        newfeats = []
        for feat in feats:
            newfeats.append(feat + '_SQ')
            newfeats.append(feat+ '_THIRD')
            newfeats.append(feat + '_SQRT')
        feats.extend(newfeats)
        scaler = StandardScaler()
        dftrain = self._df[ self._df.fortrain == 1]
        scaler.fit(dftrain[feats] )
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
            self._df[feat+'_THIRD'] = self._df.apply( lambda X:X[feat] ** 3, axis = 1)
            self._df[feat+'_SQRT'] = self._df.apply( lambda X: np.sqrt( np.absolute(X[feat])), axis = 1)
    def add_new_feats(self):
        self._df['has3SsnPorch'] = self._df.apply(lambda X:X['3SsnPorch'] > 0,axis=1)
        thedict = {'EMPTY':0,'Unf':1,'BLQ':1,'GLQ':1,"ALQ":1,"LwQ":1,'Rec':1}
        self._df['hasBsmtFinType2'] = self._df['BsmtFinType2'].map(thedict).astype(int)
        thedict = {"EMPTY":0,'MnPrv':1,'GdWo':1,'GdPrv':1,'MnWw':1}
        self._df['hasFance'] = self._df['Fence'].map(thedict).astype(int)
        thedict = {'EMPTY':0,'Gd':1,'TA':1,'Ex':1,'Fa':1,'Po':1}
        self._df['hasFire'] = self._df['FireplaceQu'].map(thedict).astype(int)
        self._df['hasGradBathRoom'] = self._df.apply( lambda X:X['FullBath'] > 0, axis = 1)

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
        self._df['buildLife2BIN'] = self._df.apply(lambda X: X['buildLife2'] >= 5*30,axis=1)
        
        self._df['SaleCondition_PriceDown'] = self._df.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 2, 'AdjLand': 3, 'Family': 4, 'Normal': 5, 'Partial': 0})
        
        self._df['Age'] = 2010 - self._df['YearBuilt'] 
        self._df['AgeBin'] = self._df['Age'].apply(lambda X:np.int64(X/10))
        self._df['AgeBin'] = self._df.AgeBin.replace(
        {
        0:0,
        1:0,
        2:0,
        3:1,
        4:1,
        5:1,
        6:1,
        7:1,
        8:1,
        9:1,
        10:1,
        12:1,
        13:2,
        })
  
        
        self._df['TimeSinceSold'] = 2010 - self._df['YrSold']

        # IR2 and IR3 don't appear that often, so just make a distinction
        # between regular and irregular.
        self._df['IsRegularLotShape'] = (self._df['LotShape'] == 'Reg') * 1

        # Most properties are level; bin the other possibilities together
        # as 'not level'.
        self._df['IsLandLevel'] = (self._df['LandContour'] == 'Lvl') * 1

        # Most land slopes are gentle; treat the others as 'not gentle'.
        self._df['IsLandSlopeGentle'] = (self._df['LandSlope'] == 'Gtl') * 1

        # Most properties use standard circuit breakers.
        self._df['IsElectricalSBrkr'] = (self._df['Electrical'] == 'SBrkr') * 1

        # About 2/3rd have an attached garage.
        self._df['IsGarageDetached'] = (self._df['GarageType'] == 'Detchd') * 1

        # Most have a paved drive. Treat dirt/gravel and partial pavement
        # as 'not paved'.
        self._df['IsPavedDrive'] = (self._df['PavedDrive'] == 'Y') * 1

        # The only interesting 'misc. feature' is the presence of a shed.
        self._df['HasShed'] = (self._df['MiscFeature'] == 'Shed') * 1.  

        # If YearRemodAdd != YearBuilt, then a remodeling took place at some point.
        self._df['Remodeled'] = (self._df['YearRemodAdd'] != self._df['YearBuilt']) * 1
        
        # Did a remodeling happen in the year the house was sold?
        self._df['RecentRemodel'] = (self._df['YearRemodAdd'] == self._df['YrSold']) * 1
        
        # Was this house sold in the year it was built?
        self._df['VeryNewHouse'] = (self._df['YearBuilt'] == self._df['YrSold']) * 1

        self._df['Has2ndFloor'] = (self._df['2ndFlrSF'] == 0) * 1
        self._df['HasMasVnr'] = (self._df['MasVnrArea'] == 0) * 1
        self._df['HasWoodDeck'] = (self._df['WoodDeckSF'] == 0) * 1
        self._df['HasOpenPorch'] = (self._df['OpenPorchSF'] == 0) * 1
        self._df['HasEnclosedPorch'] = (self._df['EnclosedPorch'] == 0) * 1
        self._df['Has3SsnPorch'] = (self._df['3SsnPorch'] == 0) * 1
        self._df['HasScreenPorch'] = (self._df['ScreenPorch'] == 0) * 1    
        
        self._df['SaleCondition_PriceDown'] = self._df.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 2, 'AdjLand': 3, 'Family': 4, 'Normal': 5, 'Partial': 0})
        self._df['BadHeating'] = self._df.HeatingQC.replace(
                {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})
        self._df['TotalArea1st2nd'] = self._df['1stFlrSF'] + self._df['2ndFlrSF']
        self._df['NewerDwelling'] = self._df['MSSubClass'].replace(
            {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
             90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})    

                
        return
    def delete_feats(self):
        self._df = self._df.drop('Alley') #99% NaN
        self._df = self._df.drop('PoolArea') # const var
        self._df = self._df.drop('AllPub') # const var
        return
    def delete_samples(self):
        #print 'before samples deletion: ', len(self._df)
        self._df['del'] = self._df.apply(lambda X: X['fortrain'] == 1 and X['GrLivArea'] > 4000,axis=1)
        #self._df['del'] = self._df.apply(lambda X: X['fortrain'] == 1 and X['SalePrice'] > 550000,axis=1)
        idx = self._df[self._df['del'] == True].index
        #print self._df.iloc[idx]['del']
        self._df.drop(idx, inplace=True,axis=0)
        
        bad_idxs = [
               522,1291,1317,687,
               #521,1288,1313,685,
               #520,1285,1309,683,
               #519,1282,1305,681
               ]
        for idx in bad_idxs:
            for k in range(idx - 10, idx + 10):
                if k not in set( self._df.index.tolist()):
                    continue
                self._df.drop(k, inplace=True,axis=0)
        
        #print 'after samples deletion: ', len(self._df)
        return
    def get_train(self):
        return self._df[ self._df['fortrain'] == 1 ].drop('fortrain',axis=1)
    def get_test(self):
        return self._df[ self._df['fortrain'] == 0 ].drop('fortrain',axis=1)

