import os,sys,pdb,argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
categ_types = set( 'object,int64,int32'.split(',') )




def get_hist(df,col):
    counts = []
    names = []
    groups = df.groupby(col)
    for name, group in groups:
        names.append(name)
        counts.append(len(group))
    df_stat = pd.DataFrame({'name':names,'ratio':counts})
    df_stat['ratio'] = df_stat['ratio'] / len(df)
    df_stat = df_stat.sort_values('ratio',ascending=False)
    return df_stat

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('infile', help='input file')
    ap.add_argument('outdir', help='output folder')
    args = ap.parse_args()
    try:
        os.makedirs(args.outdir)
    except Exception,e:
        pass
    df = pd.read_csv(args.infile,nrows = 100)
    for col in df.columns.tolist():
        if str(df[col].dtype) not in categ_types:
            print 'skip ', col,df[col].dtype
            continue
        get_hist(df,col).to_csv( os.path.join(args.outdir,col + '.txt'), index=False)

