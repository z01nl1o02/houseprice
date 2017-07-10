import os,sys,pdb
import argparse
import pandas as pd
import numpy as np
from regdata import REGDATA
from sklearn.model_selection import KFold
from stack_mean import STACK_MEAN
from stack_linear import STACK_LINEAR

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
        rawdata.remove_feature_out_of_test()
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
        clf = STACK_MEAN(self._outdir)
        err,std = self.evaluate_one_clf(clf, splitN )
        print clf.name(),',',err,'+/-',std
    def run(self,indir):
        self.load_and_convert(indir)
        prd = STACK_MEAN(self._outdir)
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



