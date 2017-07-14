import os,sys,pdb
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from predictor_ridge import PREDICTOR_RIDGE
from predictor_ridgeboost import PREDICTOR_RIDGEBOOST
from predictor_GBoost import PREDICTOR_GBOOST
from predictor_svr import PREDICTOR_SVR
from predictor_elasticnet import PREDICTOR_ELASTICNET
from predictor_dtr import PREDICTOR_DTR
from predictor_rf import PREDICTOR_RF
from predictor_xgb import PREDICTOR_XGB
from predictor_kernelridge import PREDICTOR_KERNELRIDGE

class STACK_LINEAR:
    def __init__(self, outdir):
        self._name = "stacking.linear"
        self._outdir = outdir
        self._clfs = []
        self._clf2 = Ridge(alpha = 10, random_state=4000, normalize=False)
    def add_clf(self,clf):
        self._clfs.append(clf)
        return
    def name(self):
        return self._name
    def split(self,X,Y):
        idx = X.index.tolist()
        L = []
        for k in range(0,len(idx),2):
            L.append(idx[k])
        X0 = X.loc[L,:]
        Y0 = Y.loc[L]
        for k in range(1,len(idx),2):
            L.append(idx[k])
        X1 = X.loc[L,:]
        Y1 = Y.loc[L]
        return (X0,Y0,X1,Y1)
    def train_one_clf(self,clf, trainX, trainY):
        clf.train(trainX, trainY)
        #clf.write(self._outdir)
        return
    def train(self,trainX,trainY,savemodel=0):
        X0,Y0,X1,Y1 = self.split(trainX,trainY)
        trainC = np.empty( (len(self._clfs), len(Y1) ) )
        for k,clf in enumerate(self._clfs):
            self.train_one_clf(clf,X0, Y0)
            trainC[k,:] = np.asarray( clf.predict(X1) )
        trainC = np.transpose(trainC)
        self._clf2.fit(trainC, Y1)
        return
    def predict_one_clf(self,clf,testX):
        name = clf.name()
        testC = clf.predict(testX)
        testC = np.expm1(testC)
        return testC
    def predict(self, testX,readmodel=0):
        testC = np.empty( (len(self._clfs), testX.shape[0] ) )
        for k,clf in enumerate(self._clfs):
            testC[k,:] = np.asarray( self.predict_one_clf(clf,testX)  )
        testC = np.log1p( np.transpose(testC) ) 
        res = self._clf2.predict(testC)
        res = np.expm1(res)
        df = pd.DataFrame({'Id':testX['Id'],'SalePrice':res})
        df.to_csv(os.path.join(self._outdir, self._name + ".csv"), index=False,columns='Id,SalePrice'.split(',')) 
        return df['SalePrice']
    def get_clfs(self):
        return self._clfs

