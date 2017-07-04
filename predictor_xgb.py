import os,sys,pdb
import cPickle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

class PREDICTOR_XGB:
    def __init__(self):
        self._name = 'xgboost'
        params = {'learning_rate': [0.1,0.5], 'n_estimators':[1000]}
        self._clf = GridSearchCV(xgb.XGBRegressor(),params)
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
