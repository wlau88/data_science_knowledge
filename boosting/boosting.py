import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import sklearn.metrics as metrics 


class AdaBoostBinaryClassifier(object):
    '''
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50

    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    '''

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1):

        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

    def fit(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        Build the estimators for the AdaBoost estimator.
        '''

        sample_weight=np.ones(len(y))/float(len(y))
        for i in range(self.n_estimator):
            estimator, sample_weight, self.estimator_weight_[i] = self._boost(x,y,sample_weight)
            self.estimators_.append(estimator)



    def _boost(self, x, y, sample_weight):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array

        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)

        Go through one iteration of the AdaBoost algorithm. Build one estimator.
        '''

        estimator = clone(self.base_estimator)
        estimator.fit(x,y,sample_weight=sample_weight)
        y_pred = estimator.predict(x)
        estimator_error = np.inner(sample_weight, (y != y_pred))/np.sum(sample_weight)
        estimator_weight = np.log((1 - estimator_error)/estimator_error)
        sample_weight = sample_weight * np.exp((estimator_weight*(y != y_pred)))         

        return estimator, sample_weight, estimator_weight

    def predict(self, x):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix

        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        '''

        g_xsum = np.zeros(x.shape[0])
        
        for i in range(self.n_estimator):
            g_x = map((lambda x: 1 if x == 1.0 else -1.0), self.estimators_[i].predict(x))
            # pdb.set_trace()
            g_xsum += float(self.estimator_weight_[i])*np.array(g_x)
            

        return (g_xsum > 0)*1

    def score(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''

        return metrics.accuracy_score(y, self.predict(x))




























