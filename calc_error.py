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
        print 'rmse log: ',E
        E = self.RMSE(df['Y'],df['C'])
        print 'rmse: ',E
        return E
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('infile',help='input file')
    args = ap.parse_args()
    ERROR().run(args.infile)
