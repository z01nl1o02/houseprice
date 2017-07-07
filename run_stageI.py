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

class PREDICTOR:
    def __init__(self, outdir):
        self._name = "stacking.mean"
        self._outdir = outdir
    def name(self):
        return self._name
    def train_one_clf(self,clf, trainX, trainY):
        clf.train(trainX, trainY)
        clf.write(self._outdir)
        return
    def train(self,trainX,trainY):
        self.train_one_clf(PREDICTOR_XGB(),trainX, trainY)
        self.train_one_clf(PREDICTOR_GBOOST(),trainX,trainY)
        #self.train_one_clf(PREDICTOR_RIDGEBOOST(),trainX, trainY)
        self.train_one_clf(PREDICTOR_RIDGE(),trainX, trainY)
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
        testCs = []
        testCs.append( self.predict_one_clf(PREDICTOR_GBOOST(),testX) )
        testCs.append( self.predict_one_clf(PREDICTOR_XGB(), testX) )
        #testCs.append( self.predict_one_clf(PREDICTOR_RIDGEBOOST(),testX) )
        testCs.append( self.predict_one_clf(PREDICTOR_RIDGE(), testX) )
        res = reduce(lambda X,Y: X + Y,testCs)
        res = res / len(testCs)
        df = pd.DataFrame({'Id':testX['Id'],'SalePrice':res})
        df.to_csv(os.path.join(self._outdir, self._name + ".csv"), index=False,columns='Id,SalePrice'.split(',')) 
        return df['SalePrice']
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
      #  rawdata.add_higher_order()
        rawdata.one_hot_encoding()
        data = rawdata.get_train()
        self._testX = rawdata.get_test()
        names = data.columns.tolist()
        names.remove('SalePrice')

        self._testX = self._testX[names]
        self._trainX = data[names]
        self._trainY = data['SalePrice']
        self._verifyX = data[names]
        self._verifyY = data['SalePrice']
        return
    def RMSE(self,Y,C):
        Y = np.asarray(Y)
        C = np.asarray(C)
        E = np.sqrt( ((Y - C) ** 2).mean())
        return E
    def evaluate_one_clf(self,clf, splitN=3):
        kf = KFold(n_splits = splitN, shuffle=False)
        errs = []
        for itrain,itest in kf.split(self._trainX):
            clf.train(self._trainX.iloc[itrain], self._trainY.iloc[itrain])
            C = clf.predict(self._trainX.iloc[itest])
            errs.append( self.RMSE(self._trainY.iloc[itest], np.log1p(C) ) )
        errs = np.asarray(errs)
        m = errs.mean()
        s = errs.std()
        return (m,s)
    def evaluate(self,indir):
        self.load_and_convert(indir)
        splitN = 3
        clf = PREDICTOR(self._outdir)
        err,std = self.evaluate_one_clf(clf, splitN )
        print clf.name(),',',err,'+/-',std

    def run(self,indir):
        self.load_and_convert(indir)
        prd = PREDICTOR(self._outdir)
        prd.train(self._trainX, self._trainY)
        prd.predict(self._testX)
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



