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
    def load_and_convert(self,indir, ratio,predictALL):
        rawdata = REGDATA(indir)
        rawdata.remove_missing_data()
        rawdata.remove_skewing()
       # rawdata.standandlize()
      #  rawdata.selection()
      #  rawdata.add_higher_order()
        rawdata.one_hot_encoding()
        data = rawdata.get_train()
        self._testX = rawdata.get_test()
        names = data.columns.tolist()
        names.remove('SalePrice')
        self._testX = self._testX[names]
        if 0 == predictALL and ratio < 1 and ratio > 0: #for stacking, this line should be skipped !
            print '!!!not all samples used in training'
            num = np.int64( len(data) * ratio)
            self._trainX = data[0:num][names]
            self._trainY = data[0:num]['SalePrice']
            self._verifyX = data[num:][names]
            self._verifyY = data[num:]['SalePrice']
        else:
            self._trainX = data[names]
            self._trainY = data['SalePrice']
            self._verifyX = data[names]
            self._verifyY = data['SalePrice']
        return
    def train_one_clf(self,clf):
        clf.train(self._trainX, self._trainY)
        clf.write(self._outdir)
        return
    def train(self):
        self.train_one_clf(PREDICTOR_GBOOST())
        self.train_one_clf(PREDICTOR_XGB())
        self.train_one_clf(PREDICTOR_RIDGEBOOST())
        self.train_one_clf(PREDICTOR_RIDGE())
        return
    def test_one_clf(self,clf):
        name = clf.name()
        clf.read(self._outdir)
        testC = clf.predict(self._verifyX)
        res = pd.DataFrame({'Id':self._verifyX['Id'],'Y':np.expm1(self._verifyY), 'C':np.expm1(testC)})
        res.to_csv( os.path.join(self._outdir,name + '.log'), index=False, columns = 'Id,Y,C'.split(','))

        testC = clf.predict(self._testX)
        pd.DataFrame({'Id':self._testX['Id'],'SalePrice':np.expm1(testC)}).to_csv(os.path.join(self._outdir,name + '.csv'), 
                index=False,columns='Id,SalePrice'.split(','))
        return np.expm1(testC)

    def test(self):
        testCs = []
        testCs.append( self.test_one_clf(PREDICTOR_GBOOST()) )
        testCs.append( self.test_one_clf(PREDICTOR_XGB()) )
        testCs.append( self.test_one_clf(PREDICTOR_RIDGEBOOST()) )
        testCs.append( self.test_one_clf(PREDICTOR_RIDGE()) )
        res = reduce(lambda X,Y: X + Y,testCs)
        res = res / len(testCs)
        pd.DataFrame({'Id':self._testX['Id'],'SalePrice':res}).to_csv(os.path.join(self._outdir, 'stacking.mean.csv'), 
                index=False,columns='Id,SalePrice'.split(',')) 
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
            errs.append( self.RMSE(self._trainY.iloc[itest], C ) )
        errs = np.asarray(errs)
        return errs.mean()
    def evaluate(self,indir,trainRatio,testALL):
        self.load_and_convert(indir,trainRatio, testALL)
        splitN = 3
        clf = PREDICTOR_XGB()
        err = self.evaluate_one_clf(clf, splitN )
        print clf.name(),',',err

        clf = PREDICTOR_RIDGE()
        err = self.evaluate_one_clf(clf, splitN )
        print clf.name(),',',err

        clf = PREDICTOR_RIDGEBOOST()
        err = self.evaluate_one_clf(clf, splitN )
        print clf.name(),',',err

        clf = PREDICTOR_GBOOST()
        err = self.evaluate_one_clf(clf, splitN )
        print clf.name(),',',err

    def run(self,indir, trainRatio, testALL):
        self.load_and_convert(indir,trainRatio, testALL)
        self.train()
        self.test()
        return

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='input dir')
    ap.add_argument('outdir',help='output dir')
    ap.add_argument('-split',help='split size for train(0,1)', type=np.float64, default=0.8)
    ap.add_argument('-testALL',help='predict all samples(for stacking)', type=np.int64, default=0)
    args = ap.parse_args()
    #HOUSE_PRICE(args.outdir).run(args.indir, args.split, args.testALL)
    HOUSE_PRICE(args.outdir).evaluate(args.indir, args.split, args.testALL)



