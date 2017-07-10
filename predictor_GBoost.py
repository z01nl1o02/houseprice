import os,sys,pdb
import cPickle
from sklearn.ensemble import GradientBoostingRegressor
class PREDICTOR_GBOOST:
    def __init__(self):
        self._name = 'gboost'
        params = {'n_estimators':1000, 'max_depth':19,
                #'min_samples_split':5,
                'min_samples_leaf':3,
                'learning_rate':0.01,
                'loss':'ls','random_state':100, 'subsample':0.1}
        self._clf = GradientBoostingRegressor(**params)
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

