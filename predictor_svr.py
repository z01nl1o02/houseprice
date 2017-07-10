import os,sys,pdb
import cPickle
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
class PREDICTOR_SVR:
    def __init__(self):
        self._name = 'svr'
        param = {
        'kernel':['rbf'],
        'C':[10,20,30,50], 
        'epsilon':[0.0]}
        self._clf = GridSearchCV(SVR(),param, n_jobs = 3)
        self._param = None
        return
    def name(self):
        return self._name
    def pre_norm(self,X,Y):
        self._param = []
        m0,m1 = [np.min(X,0), np.max(X,0)]
        ran = m1 - m0
        ran[ np.abs(ran) < 0.0001 ] = 1.0
        self._param.extend([m0,ran])
    def do_norm(self,X,Y):
        m0 = self._param[0]
        ran = self._param[1]
        scaledX = (X - m0) / ran
        return (scaledX, Y)
    def train(self,X,Y):
        self.pre_norm(X,Y)
        normX,normY = self.do_norm(X,Y)
        self._clf = self._clf.fit(normX,normY)
        return
    def write(self,outdir):
        with open(os.path.join(outdir,self._name + '.model'),'wb') as f:
            cPickle.dump( (self._param,self._clf), f)
        return
    def read(self,indir):
        with open(os.path.join(indir,self._name + '.model'),'rb') as f:
            self._param,self._clf = cPickle.load(f)
        return
    def predict(self,X):
        normX, normY = self.do_norm(X,None)
        return self._clf.predict(normX)

