import os,sys,pdb
import numpy as np
import argparse
import pandas as pd

class MERGE_RES:
    def __init__(self,outdir):
        try:
            os.makedirs(outdir)
        except Exception,e:
            pass
        self._outdir = outdir
        self._train = None
        self._test = None
    def load(self,indir):
        num = 0
        for rdir,pdirs,names in os.walk(indir):
            for name in names:
                sname,ext = os.path.splitext(name)
                if ext == '.log':
                    continue
                trainfile = os.path.join(rdir,sname + '.log')
                testfile = os.path.join(rdir,sname + '.csv')
                df = pd.read_csv(trainfile)
                if self._train is None:
                    self._train = pd.DataFrame({'Id':df['Id'],'Y':df['Y'], 'C'+str(num):df['C']  })
                else:
                    self._train['C'+str(num)] = df['C']
                df = pd.read_csv(testfile)
                if self._test is None:
                    self._test = df
                    self._test.columns = ['Id','C'+str(num)]
                else:
                    self._test['C'+str(num)] = df['SalePrice']
                num += 1
        total = len(self._train.columns) -  2
        header = ['Id','Y']
        header.extend( ['C'+str(k) for k in range(total)] )
        self._train[header].to_csv(os.path.join(self._outdir,'train.csv'), index=False)
        header.remove('Y')
        self._test[header].to_csv(os.path.join(self._outdir, 'test.csv'), index=False)
        return


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='input dir')
    ap.add_argument('outdir',help='output dir')
    args = ap.parse_args()
    MERGE_RES(args.outdir).load(args.indir)





