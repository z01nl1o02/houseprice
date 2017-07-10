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
        self._clfs = [ PREDICTOR_XGB(), PREDICTOR_ELASTICNET(), PREDICTOR_RIDGE() ]
        self._clf2 = Ridge(alpha = 1.0)
    def name(self):
        return self._name
    def train_one_clf(self,clf, trainX, trainY):
        clf.train(trainX, trainY)
        clf.write(self._outdir)
        return
    def train(self,trainX,trainY):
        trainC = np.empty( (len(self._clfs), len(trainY) ) )
        for k,clf in enumerate(self._clfs):
            self.train_one_clf(clf,trainX, trainY)
            trainC[k,:] = np.asarray( clf.predict(trainX) )
        trainC = np.transpose(trainC)
        self._clf2.fit(trainC, trainY)
        return
    def predict_one_clf(self,clf,testX):
        name = clf.name()
        clf.read(self._outdir)
        testC = clf.predict(testX)
        testC = np.expm1(testC)
        pd.DataFrame({'Id':testX['Id'],'SalePrice':testC}).to_csv(os.path.join(self._outdir,name + '.csv'), 
                index=False,columns='Id,SalePrice'.split(','))
        return testC
    def predict(self, testX):
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

