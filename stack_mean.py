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
    def train_one_clf(self,clf, trainX, trainY,savemodel):
        clf.train(trainX, trainY)
        if savemodel != 0:
            clf.write(self._outdir)
        return
    def train(self,trainX,trainY,savemodel=1):
        for clf in self._clfs:
            self.train_one_clf(clf,trainX, trainY, savemodel)
        return
    def predict_one_clf(self,clf,testX,readmodel):
        name = clf.name()
        if readmodel != 0:
            clf.read(self._outdir)
        testC = clf.predict(testX)
        testC = np.expm1(testC)
        pd.DataFrame({'Id':testX['Id'],'SalePrice':testC}).to_csv(os.path.join(self._outdir,name + '.csv'), 
                index=False,columns='Id,SalePrice'.split(','))
        return testC
    def predict(self, testX,readmodel = 1):
        testCs = []
        for clf in self._clfs:
            testCs.append( self.predict_one_clf(clf,testX,readmodel) )
        res = reduce(lambda X,Y: X + Y,testCs)
        res = res / len(testCs)
        df = pd.DataFrame({'Id':testX['Id'],'SalePrice':res})
        df.to_csv(os.path.join(self._outdir, self._name + ".csv"), index=False,columns='Id,SalePrice'.split(',')) 
        return df['SalePrice']
    def get_clfs(self):
        return self._clfs

