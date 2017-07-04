import os,sys,pdb
import argparse
import pandas as pd
import numpy as np
from predictor_ridge import PREDICTOR_RIDGE
class HOUSE_PRICE:
    def __init__(self,outdir):
        self._outdir = outdir
        self._trainX = None
        self._trainY = None
        self._testX = None
        try:
            os.makedirs(outdir)
        except Exception,e:
            pass
        return
    def load_and_convert(self,indir):
        dftrain = pd.read_csv(os.path.join(indir,'train.csv'))
        dftest = pd.read_csv(os.path.join(indir,'test.csv'))

        idx = dftest.columns
        self._trainX = dftrain[idx]
        self._trainY = dftrain['Y']
        self._testX = dftest

        
        idx = idx.tolist()
        idx.remove('Id')
        self._trainX[idx] = np.log1p(self._trainX[idx])
        self._testX[idx] = np.log1p(self._testX[idx])
        self._trainY = np.log1p(self._trainY)
        return
    def train_one_clf(self,clf):
        clf.train(self._trainX, self._trainY)
        clf.write(self._outdir)
        return
    def train(self):
        self.train_one_clf(PREDICTOR_RIDGE())
        return
    def test_one_clf(self,clf):
        name = clf.name()
        clf.read(self._outdir)
        testC = clf.predict(self._testX)
        pd.DataFrame({'Id':self._testX['Id'],'SalePrice':np.expm1(testC)}).to_csv(os.path.join(self._outdir,name + '.csv'), 
                index=False,columns='Id,SalePrice'.split(','))
        return

    def test(self):
        self.test_one_clf(PREDICTOR_RIDGE())
        return
    def run(self,indir):
        self.load_and_convert(indir)
        self.train()
        self.test()
        return

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='input dir')
    ap.add_argument('outdir',help='output dir')
    args = ap.parse_args()
    HOUSE_PRICE(args.outdir).run(args.indir)




