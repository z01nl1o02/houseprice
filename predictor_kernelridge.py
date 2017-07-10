import os,sys,pdb
import cPickle
from sklearn.kernel_ridge import KernelRidge
class PREDICTOR_KERNELRIDGE:
    def __init__(self):
        self._name = 'kernelridge'
        self._clf = KernelRidge(alpha=3500, kernel='polynomial', degree=2, coef0=0)
        return
    def name(self):
        return self._name
    def train(self,X,Y):
        self._clf = self._clf.fit(X,Y)
        return
    def write(self,outdir):
        with open(os.path.join(outdir,self._name + '.model'),'wb') as f:
            cPickle.dump( self._clf, f)
        return
    def read(self,indir):
        with open(os.path.join(indir,self._name + '.model'),'rb') as f:
            self._clf = cPickle.load(f)
        return
    def predict(self,X):
        return self._clf.predict(X)

