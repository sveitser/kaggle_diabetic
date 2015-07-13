import numpy as np
from joblib import delayed, Parallel

from sklearn.base import clone
from sklearn.multiclass import OneVsRestClassifier

def _fit(estimator, X, y):
    return clone(estimator).fit(X, y)

class OrdinalClassifier(OneVsRestClassifier):
    def fit(self, X, y):
        columns = [(y <= i).astype(int) for i in range(len(np.unique(y)) - 1)]
        print(columns)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit)(
            self.estimator, X, column) for column in columns)
        return self

    def predict(self, X):
        probas = np.hstack([est.predict_proba(X)[:, 1][:, np.newaxis] 
                            for est in self.estimators_])
        print(probas)
        class_probas = np.zeros([probas.shape[0], probas.shape[1] + 1])
        class_probas[:, 0] = probas[:, 0]
        class_probas[:, 1:-1] = np.diff(probas, axis=1)
        class_probas[:, -1] = 1.0 - probas[:, -1]
        print(class_probas)
        return np.argmax(class_probas, axis=1)



