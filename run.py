import os,sys,pdb
import argparse
import pandas as pd
import numpy as np
from regdata import REGDATA
from predictor_ridge import PREDICTOR_RIDGE
from predictor_GBoost import PREDICTOR_GBOOST
from predictor_svr import PREDICTOR_SVR
from predictor_elasticnet import PREDICTOR_ELASTICNET
from predictor_dtr import PREDICTOR_DTR
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
    def load_and_convert(self,indir, ratio):
        rawdata = REGDATA(indir)
        rawdata.remove_missing_data()
        rawdata.remove_skewing()
      #  rawdata.selection()
      #  rawdata.add_higher_order()
        rawdata.one_hot_encoding()
        data = rawdata.get_train()
        self._testX = rawdata.get_test()
        names = data.columns.tolist()
        names.remove('SalePrice')
        self._testX = self._testX[names]
        if ratio < 1 and ratio > 0:
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
    def train(self):
        clf_ridge = PREDICTOR_RIDGE()
        clf_ridge.train(self._trainX, self._trainY)
        clf_ridge.write(self._outdir)

        clf = PREDICTOR_GBOOST()
        clf.train(self._trainX, self._trainY)
        clf.write(self._outdir)


        clf = PREDICTOR_SVR()
        clf.train(self._trainX, self._trainY)
        clf.write(self._outdir)

        clf = PREDICTOR_ELASTICNET()
        clf.train(self._trainX, self._trainY)
        clf.write(self._outdir)


        clf = PREDICTOR_DTR()
        clf.train(self._trainX, self._trainY)
        clf.write(self._outdir)

        return
    def test(self):
        clf_ridge = PREDICTOR_RIDGE()
        clf_ridge.read(self._outdir)
        testC = clf_ridge.predict(self._verifyX)
        res = pd.DataFrame({'Id':self._verifyX['Id'],'Y':np.expm1(self._verifyY), 'C':np.expm1(testC)})
        res.to_csv( os.path.join(self._outdir,'ridge.log'), index=False, columns = 'Id,Y,C'.split(','))

        self._testX.to_csv('test.convert.csv',index=False)
        testC = clf_ridge.predict(self._testX)
        pd.DataFrame({'Id':self._testX['Id'],'SalePrice':np.expm1(testC)}).to_csv( os.path.join(self._outdir,'ridge.csv'),
                index=False, columns='Id,SalePrice'.split(','))

        clf = PREDICTOR_GBOOST()
        clf.read(self._outdir)
        testC = clf.predict(self._verifyX)
        res = pd.DataFrame({'Id':self._verifyX['Id'],'Y':np.expm1(self._verifyY), 'C':np.expm1(testC)})
        res.to_csv( os.path.join(self._outdir,'GBoost.log'), index=False, columns = 'Id,Y,C'.split(','))

        testC = clf.predict(self._testX)
        pd.DataFrame({'Id':self._testX['Id'],'SalePrice':np.expm1(testC)}).to_csv(os.path.join(self._outdir,'GBoost.csv'), 
                index=False,columns='Id,SalePrice'.split(','))


        clf = PREDICTOR_SVR()
        clf.read(self._outdir)
        testC = clf.predict(self._verifyX)
        res = pd.DataFrame({'Id':self._verifyX['Id'],'Y':np.expm1(self._verifyY), 'C':np.expm1(testC)})
        res.to_csv( os.path.join(self._outdir,'svr.log'), index=False, columns = 'Id,Y,C'.split(','))

        testC = clf.predict(self._testX)
        pd.DataFrame({'Id':self._testX['Id'],'SalePrice':np.expm1(testC)}).to_csv(os.path.join(self._outdir,'svr.csv'), 
                index=False,columns='Id,SalePrice'.split(','))


        clf = PREDICTOR_ELASTICNET()
        clf.read(self._outdir)
        testC = clf.predict(self._verifyX)
        res = pd.DataFrame({'Id':self._verifyX['Id'],'Y':np.expm1(self._verifyY), 'C':np.expm1(testC)})
        res.to_csv( os.path.join(self._outdir,'elasticnet.log'), index=False, columns = 'Id,Y,C'.split(','))

        testC = clf.predict(self._testX)
        pd.DataFrame({'Id':self._testX['Id'],'SalePrice':np.expm1(testC)}).to_csv(os.path.join(self._outdir,'elasticnet.csv'), 
                index=False,columns='Id,SalePrice'.split(','))



        clf = PREDICTOR_DTR()
        clf.read(self._outdir)
        testC = clf.predict(self._verifyX)
        res = pd.DataFrame({'Id':self._verifyX['Id'],'Y':np.expm1(self._verifyY), 'C':np.expm1(testC)})
        res.to_csv( os.path.join(self._outdir,'DTR.log'), index=False, columns = 'Id,Y,C'.split(','))

        testC = clf.predict(self._testX)
        pd.DataFrame({'Id':self._testX['Id'],'SalePrice':np.expm1(testC)}).to_csv(os.path.join(self._outdir,'DTR.csv'), 
                index=False,columns='Id,SalePrice'.split(','))


        return
    def run(self,indir):
        self.load_and_convert(indir,0.6)
        self.train()
        self.test()
        return

if __name__=="__main__":
    HOUSE_PRICE('result').run('.')




