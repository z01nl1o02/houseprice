import os,sys,pdb
import cPickle
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
class PREDICTOR_SVR:
    def __init__(self):
        self._name = 'svr'
        param = {'C':[0.00001,0.0001,0.001,0.01,0.1,1,10,20,30,50]}
        self._clf = GridSearchCV(LinearSVR(),param)
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

