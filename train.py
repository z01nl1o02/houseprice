import os,sys,pdb
import argparse
import pandas as pd
import numpy as np
from regdata import REGDATA
from predictor_ridge import PREDICTOR_RIDGE

class HOUSE_PRICE:
    def __init__(self,outdir):
        self._outdir = outdir
        self._trainX = None
        self._trainY = None
        self._testdX = None
        self._testdY = None
        try:
            os.makedirs(outdir)
        except Exception,e:
            pass
        return
    def load_and_convert(self,filepath, ratio):
        rawdata = REGDATA(filepath)
        rawdata.remove_missing_data()
        rawdata.add_buildlife()
        rawdata.remove_skewing()
        rawdata.one_hot_encoding()
        data = rawdata.get_data()
        names = data.columns.tolist()
        nameX = []
        for n in names:
            if n == 'SalePrice':
                continue
            nameX.append(n)
        if ratio < 1 and ratio > 0:
            num = np.int64( len(data) * ratio)
            self._trainX = data[0:num][nameX]
            self._trainY = data[0:num]['SalePrice']
            self._testX = data[num:][nameX]
            self._testY = data[num:]['SalePrice']
        else:
            self._trainX = data
            self._trainY = data['SalePrice']
            self._testX = data
            self._testY = data['SalePrice']
        return
    def train(self):
        clf_ridge = PREDICTOR_RIDGE()
        clf_ridge.train(self._trainX, self._trainY)
        clf_ridge.write(self._outdir)
        return
    def test(self):
        clf_ridge = PREDICTOR_RIDGE()
        clf_ridge.read(self._outdir)
        testC = clf_ridge.predict(self._testX)
        res = pd.DataFrame({'ID':self._testX['Id'],'Y':self._testY, 'C':testC})
        res.to_csv( os.path.join(self._outdir,'rdige.log'), index=False, columns = 'ID,Y,C'.split(','))
        return
    def run(self,trainfile, testfile):
        self.load_and_convert(trainfile,0.8)
        self.train()
        self.test()
        return

if __name__=="__main__":
    HOUSE_PRICE('result').run('train.csv','test.csv')




