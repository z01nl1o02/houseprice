import os,sys,pdb
import argparse
import pandas as pd
import numpy as np
from regdata import REGDATA
from sklearn.model_selection import KFold
from stack_mean import STACK_MEAN
from stack_linear import STACK_LINEAR
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


def RMSE(Y,C):
    Y = np.asarray(Y)
    C = np.asarray(C)
    E = np.sqrt( ((Y - C) ** 2).mean())
    return E

def mt_train(param):
    clf, trainX,trainY,testX, testY = param
    clf.train(trainX, trainY, savemodel = 0)
    C = clf.predict(testX, readmodel = 0)
    err = RMSE(testY, np.log1p(C))
    return err

class HOUSE_PRICE:
    def __init__(self,outdir):
        self._outdir = outdir
        self._trainX = None
        self._trainY = None
        self._verifyX = None
        self._verifyY = None
        self._testX = None
        try:
            os.makedirs(outdir)
        except Exception,e:
            pass
        return
    def load_and_convert(self,indir):
        rawdata = REGDATA(indir)
        rawdata.delete_samples()
        rawdata.add_new_feats()
        #rawdata.delete_feats()
        rawdata.remove_missing_data()
        rawdata.remove_skewing()
        rawdata.standandlize()
      #  rawdata.selection()
        rawdata.add_higher_order()
        rawdata.one_hot_encoding()
        #rawdata.remove_feature_out_of_test()
        data = rawdata.get_train()
        self._testX = rawdata.get_test()
        names = data.columns.tolist()
        names.remove('SalePrice')

        self._testX = self._testX[names]
        self._trainX = data[names]
        self._trainY = data['SalePrice']

        return

    def evaluate_one_clf(self,splitN=3):
        kf = KFold(n_splits = splitN, shuffle=False)
        params = []
        for itrain,itest in kf.split(self._trainX):
            clf = self.get_assemble_clf()
            params.append( (clf, self._trainX.iloc[itrain], self._trainY.iloc[itrain],  self._trainX.iloc[itest], self._trainY.iloc[itest] ) )
        results = [] 
        pool = mp.Pool(3)
        for param in params:
            #mt_train(param)
            results.append( pool.apply_async(mt_train, (param,)) )
        pool.close()
        pool.join()
        errs = []
        for res in results:
            errs.append( res.get())
        errs = np.asarray(errs)
        m = errs.mean()
        s = errs.std()
        return (m,s)
    def get_assemble_clf(self):
        clf = STACK_MEAN(self._outdir)
        clf.add_clf( PREDICTOR_ELASTICNET() )
        clf.add_clf( PREDICTOR_XGB() )
        clf.add_clf( PREDICTOR_RIDGE() )
        return clf
        clf.add_clf( PREDICTOR_GBOOST() )
        clf.add_clf( PREDICTOR_SVR() )
        clf.add_clf( PREDICTOR_RIDGEBOOST() )
        return clf
    def evaluate(self,indir):
        self.load_and_convert(indir)
        splitN = 3 
        err,std = self.evaluate_one_clf(splitN )
        print err,'+/-',std
    def run(self,indir):
        self.load_and_convert(indir)
        clf = self.get_assemble_clf()
        clf.train(self._trainX, self._trainY)
        clf.predict(self._testX)
        return

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('mode',help='work or evaluate')
    ap.add_argument('indir',help='input dir')
    ap.add_argument('outdir',help='output dir')
    args = ap.parse_args()
    if args.mode == 'work':
        HOUSE_PRICE(args.outdir).run(args.indir)
    elif args.mode == 'eva':
        HOUSE_PRICE(args.outdir).evaluate(args.indir)
    else:
        print 'unk option'



