import os,sys,pdb
import cPickle
from sklearn.ensemble import GradientBoostingRegressor
class PREDICTOR_GBOOST:
    def __init__(self):
        self._name = 'GBoost.model'
        params = {'n_estimators':500, 'max_depth':4,
                'min_samples_split':2,'learning_rate':0.01,
                'loss':'ls'}
        self._clf = GradientBoostingRegressor(**params)
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

