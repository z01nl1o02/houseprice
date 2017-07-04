import os,sys,pdb
import pandas as pd
import numpy as np
import argparse

class ERROR:
    def RMSE_LOG(self,Y,C):
        Y = np.asarray(Y)
        C = np.asarray(C)
        Y = np.log(Y)
        C = np.log(C)
        return self.RMSE(Y,C)
    def RMSE(self,Y,C):
        Y = np.asarray(Y)
        C = np.asarray(C)
        E = np.sqrt( ((Y - C) ** 2).mean() )
        return E
    def run(self,filepath):
        df = pd.read_csv(filepath)
        E = self.RMSE_LOG(df['Y'],df['C'])
        return E
    def run_all(self,indir):
        res = []
        for root, pdirs, names in os.walk(indir):
            for name in names:
                sname,ext = os.path.splitext(name)
                if '.log' == ext:
                    E = self.run(os.path.join(root,name))
                    res.append((name,E))
        res = sorted( res, key = lambda X: X[1])
        for name, e in res:
            print '%9f,%s'%(e,name)
        return
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='input dir')
    args = ap.parse_args()
    ERROR().run_all(args.indir)

