''' template matching classifier for neurofeedback time series
'''
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import numpy as np
from scipy import stats
import pandas as pd


class TemplateMatch(BaseEstimator, ClassifierMixin):
    '''
    '''
    def __init__(self, norm_pre=False, norm_post=False):
        '''
        '''
        self.norm_pre = norm_pre
        self.norm_post = norm_post

    def fit(self, X, y, ):
        # breakpoint()
        # cheack that number of subjects matches number of labels
        assert X.shape[0]==len(y), 'number of subjects and labels do not match'

        # normalize time series within subject before averaging?
        if self.norm_pre:
            X = stats.zscore(X, axis=1)

        # get index for each group
        #-------------------------------------------------------------
        self.labels_unique = np.unique(y)
        if len(self.labels_unique)==2:
            self.labels_numeric_map = {
                self.labels_unique[0]:-1,
                self.labels_unique[1]:1
            }
        y_numeric = np.array([self.labels_numeric_map[i] for i in y])
        self.labels_idx = [np.where(y==g)[0] for g in self.labels_unique]

        # create templates by averaging across subjects
        #-------------------------------------------------------------
        self.templates = {}
        # iterate over labels
        for label_i, label in enumerate(self.labels_unique):
            # get template for each label
            _template = X[self.labels_idx[label_i],:].mean(axis=0) # (rois/components, time)
            # rescale along time axis
            if self.norm_post:
                _template = stats.zscore(_template, axis=0)
            # store template
            self.templates[label] = _template # (rois/components, time)

        return self

    def predict(self, X):   
        # breakpoint()
        # apply matching classifier
        #-------------------------------------------------------------
        n_subj, n_times = X.shape
        predictions = []
        # iterate over subject-sessions
        for i in range(n_subj):
            # keep track of iscs for each label/template
            iscs=[]
            labels =[]
            # iterate over labels
            for label_i, label in enumerate(self.labels_unique):
                labels.append(label)

                # get template for current label
                X_train = self.templates[label] # (rois/components, time)
                # get data for current subject-session, with any label-specific transforms applied
                X_test = X[i,:] # (components, time)
                # rescale along time axis
                if self.norm_post:
                    X_test = stats.zscore(X_test, axis=0)
                # get mean ISC between held out subjects and top component for training data
                assert X_train.shape==X_test.shape, 'template and test data do not match'
                # get correlation between template and test data
                if len(X_train.flatten())==1:
                    _isc = X_train.flatten()[0]*X_test.flatten()[0]
                else:
                    _isc = stats.pearsonr(X_train.flatten(), X_test.flatten())[0]
                # keep track of isc for each template
                iscs.append(_isc)
            
            # get best match
            best_match = labels[np.argmax(iscs)]
            # store best match
            predictions.append(best_match)

        return predictions
    
    def nf_score(self, X, nan_method='ignore'):
        n_subj, n_times = X.shape
        scores = []
        # iterate over subject-sessions
        for i in range(n_subj):
            # keep track of iscs for each label/template
            iscs=[]
            labels =[]
            # iterate over labels
            try:
                for label_i, label in enumerate(self.labels_unique):
                    labels.append(label)

                    # get template for current label
                    X_train = self.templates[label] # (rois/components, time)
                    # get data for current subject-session, with any label-specific transforms applied
                    X_test = X[i,:] # (components, time)
                    nan_idx= None
                    if nan_method=='ignore':
                        # get nan indices in either X_train or X_test to ignore
                        nan_idx = np.isnan(X_train) | np.isnan(X_test)
                    elif nan_method=='interpolate':
                        # interpolate nan values
                        X_test  = pd.Series(X_test).interpolate(method='linear', limit_direction='both').to_numpy()
                    if nan_idx is not None:
                        # remove nan indices from both X_train and X_test
                        X_train = X_train[~nan_idx]
                        X_test = X_test[~nan_idx]
                    # rescale along time axis
                    if self.norm_post and X_test.std(axis=0)!=0:
                        X_test = stats.zscore(X_test, axis=0)
                    # get mean ISC between held out subjects and top component for training data
                    assert X_train.shape==X_test.shape, 'template and test data do not match'
                    # get correlation between template and test data
                    if len(X_train.flatten())==1:
                        _isc = X_train.flatten()[0]*X_test.flatten()[0]
                    else:
                        _isc = stats.pearsonr(X_train.flatten(), X_test.flatten())[0]
                    # keep track of isc for each template
                    iscs.append(_isc)
                corr1 = iscs[0]
                corr2 = iscs[1]
                score = (corr2-corr1)/(np.abs(corr2)+np.abs(corr1) + 1e-10)
            except:
                # if there is an error, set score to nan
                score = np.nan
            scores.append(score)
        return scores
    

 