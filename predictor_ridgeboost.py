import os,sys,pdb
import cPickle
from sklearn.linear_model import RidgeCV
class PREDICTOR_RIDGEBOOST:
    def __init__(self):
        self._name = 'ridgeboost'
        self._weights = [0.5 for k in range(10)]
        self._clfs = [ RidgeCV(alphas=[0.05,0.1,0.3,1.3,5,10,15,30,50,75]) for k in range(len(self._weights))]
        return
    def name(self):
        return self._name
    def train(self,X,Y):
        nextY = Y
        for k in range( len(self._weights) ):
            self._clfs[k] = self._clfs[k].fit(X,nextY)
            nextY = nextY - self._weights[k] * self._clfs[k].predict(X)
            #print nextY.abs().min(), nextY.abs().max()
        return
    def write(self,outdir):
        with open(os.path.join(outdir,self._name + '.model'),'wb') as f:
            cPickle.dump( (self._weights, self._clfs), f)
        return
    def read(self,indir):
        with open(os.path.join(indir,self._name + '.model'),'rb') as f:
            self._weights, self._clfs = cPickle.load(f)
        return
    def predict(self,X):
        C = self._weights[0] * self._clfs[0].predict(X)
        for k in range(1, len(self._weights)):
            C = C + self._weights[k] * self._clfs[k].predict(X)
        return C
            

