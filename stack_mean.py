import os,sys,pdb
import argparse
import pandas as pd
import numpy as np
from regdata import REGDATA
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
import multiprocessing as mp

def train_one_clf_mt(param):
    k,clf,X,Y = param
    clf.train(X, Y)
    return (k,clf)
    
class STACK_MEAN:
    def __init__(self, outdir):
        self._name = "stacking.mean"
        self._outdir = outdir
        self._clfs = []
    def add_clf(self,clf):
        self._clfs.append(clf)
        return
    def name(self):
        return self._name
    def train_one_clf(self,clf, trainX, trainY):
        clf.train(trainX, trainY)
        return
    def train(self,trainX,trainY, cpu = 1):
        feats = trainX.columns.tolist()
        feats.remove('Id')
        results = []
        if cpu > 1:
            pool = mp.Pool(cpu)
            for k,clf in enumerate( self._clfs ):
                param = (k, clf, trainX[feats], trainY) 
                results.append( pool.apply_async( train_one_clf_mt, (param,)) )
                #self.train_one_clf(clf,trainX[feats], trainY)
            pool.close()
            pool.join()
            for res in results:
                k,clf = res.get()
                self._clfs[k] = clf
        else:
            for k,clf in enumerate( self._clfs ):
                #param = (k, clf, trainX[feats], trainY) 
                #results.append( pool.apply_async( train_one_clf_mt, (param,)) )
                self.train_one_clf(clf,trainX[feats], trainY)
        return
    def predict_one_clf(self,clf,testX):
        name = clf.name()
        testC = clf.predict(testX)
        testC = np.expm1(testC)
        return testC
    def predict(self, testX):
        feats = testX.columns.tolist()
        feats.remove('Id')
        testCs = []
        for clf in self._clfs:
            testCs.append( self.predict_one_clf(clf,testX[feats]) )
        res = reduce(lambda X,Y: X + Y,testCs)
        res = res / len(testCs)
        df = pd.DataFrame({'Id':testX['Id'],'SalePrice':res})
        df.to_csv(os.path.join(self._outdir, self._name + ".csv"), index=False,columns='Id,SalePrice'.split(',')) 
        return df['SalePrice']
    def get_clfs(self):
        return self._clfs

