import os,sys,pdb
import cPickle
from sklearn.tree import DecisionTreeRegressor
class PREDICTOR_DTR:
    def __init__(self):
        self._name = 'DTR.model'
        self._clf = DecisionTreeRegressor()
        return
    def train(self,X,Y):
        self._clf = self._clf.fit(X,Y)
        return
    def write(self,outdir):
        with open(os.path.join(outdir,self._name),'wb') as f:
            cPickle.dump( self._clf, f)
        return
    def read(self,indir):
        with open(os.path.join(indir,self._name),'rb') as f:
            self._clf = cPickle.load(f)
        return
    def predict(self,X):
        return self._clf.predict(X)

