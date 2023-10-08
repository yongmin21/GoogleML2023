import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


class Custom_Pipeline:
    """Custom Pipeline for Data Prerpocessing
    Parameters:
    X (pd.DataFrame) : Input Data to Pipeline (target feature included, it will be dropped in pipeline)
    y (pd.Series) : Input target feature to Pipeline
    
    Return:
    preprocessed X (pd.DataFrame) : Preprocessed Input Data (without target feature)
    """
    
    def __init__(self, X, y):
        assert X.columns.isin(["outcome"]).sum() == 1, "X must have target feature for this pipeline"
        
        self._train_bool = X.columns.isin(["outcome"]).sum() # if train: 1
        
        self._X = X
        self._y = y
        self._encoder_params = {}
        self.smoothing_memory = {}
        
        self._imputer = KNNImputer(n_neighbors=11).set_output(transform="pandas")
        
        self.ENCODE_MAP = {
            'died' : 0,
            'euthanized' : 1,
            'lived' : 2
        }
        
        self.DECODE_MAP = {
            0 : 'died',
            1 : 'euthanized',
            2 : 'lived'
        }

        
    def encode(self):
        """ Encode Target Features (str -> int)
        """
        self._X.outcome = self._X.outcome.map(self.ENCODE_MAP)
        self._y = self._y.map(self.ENCODE_MAP)
        return self
    
    def decode(self, y):
        """ Decode Target Features, reverse of encode (int -> str)
        """
        decoded_y = y.map(self.DECODE_MAP)
        return decoded_y
    
    def decode_lesion_1(self):
        pass
    
    def feature_transform(self):
        """ Add New Features
        """
        # Not works but drop score when removing
        # self._X['treated_more_than_once'] = self._X['hospital_number'].apply(
        #     lambda x: 1 if self._X['hospital_number'].value_counts()[x] > 1 else 0
        # )
        
        self._X['number_of_treatements'] = self._X['hospital_number'].apply(
            lambda x: self._X['hospital_number'].value_counts()[x]
        )
        
        self._X['deviation_from_normal_temp'] = self._X['rectal_temp'].apply(
            lambda x: abs(x - 37.8)
        )
        
        # feature importane 0 but drop score when removing
        # self._X['lesion_2'] = self._X['lesion_2'].apply(
        #     lambda x: 2 if '11' in str(x) else (0 if str(x) == '0' else 1)
        # )
        
        self._X['surgery'] = self._X['surgery'].apply(
            lambda x: 1 if x == 'yes' else 0
        )
        
        if self._train_bool:
            self._X['pain'] = self._X['pain'].replace({
                'slight' : 'mild_pain'
            })
        
        else:
            self._X['pain'] = self._X['pain'].replace({
                'moderate' : 'mild_pain'
            })
            
        self._X['capillary_refill_time'] = self._X['capillary_refill_time'].replace({
            '3' : 'more_3_sec'
        })
        
        self._X['peristalsis'] = self._X['peristalsis'].replace({
            'distend_small' : np.nan
        })
        
        self._X['nasogastric_reflux'] = self._X['nasogastric_reflux'].replace({
            'slight' : np.nan
        })
        
        self._X['rectal_exam_feces'] = self._X['rectal_exam_feces'].replace({
            'serosanguious' : np.nan
        })

        return self
    
    def drop_feature(self, dropcols=None):
        """ Drop useless feature
        """
        self._X = self._X.drop(dropcols, axis=1)
        
        return self
    
    
    def encoder(self, usecols, alpha=0.5):
        """ Encode String Variables to numeric.
        Parameters:
        usecols (list) : list of column name you want to encode
        alpha (float) : the smoothing parameter. smoother when alpha is high.
        
        Return:
        self
        """
        encoded_cols = []
        
        if self._train_bool: # for train
            
            self._n_rows = self._X.shape[0]
            self._global_mean = self._X['outcome'].mean()
            
            for col in usecols:
                # fit
                target_mean = self._X.groupby(col)['outcome'].mean()
                smoothing = (target_mean * self._n_rows + self._global_mean * alpha) / (self._n_rows + alpha)
                self.smoothing_memory[col] = smoothing
                
                # transform
                encoded_col = self._X[col].map(smoothing)
                encoded_cols.append(pd.DataFrame({col:encoded_col}))
    
            
        else: # for test
            for col in usecols:
                # transform
                smoothing = self.smoothing_memory[col]
                encoded_col = self._X[col].map(smoothing)
            
                encoded_cols.append(pd.DataFrame({col:encoded_col}))
        
        results = pd.concat(encoded_cols, axis=1)
        self._X = pd.concat([self._X.drop(usecols, axis=1), results], axis=1)
        
        return self
    
    def imputer(self):
        """Impute Missing Values.
        """

        if self._train_bool:
            self._X = self._X.drop(['outcome'], axis=1)
            self._X = self._imputer.fit_transform(self._X)
            
        else:
            self._X = self._imputer.transform(self._X)
            
        return self._X
        
    
    def fit_transform(self, usecols, alpha=None, dropcols=None):
        return self.encode().drop_feature(dropcols).feature_transform().encoder(usecols, alpha).imputer()
    
    def transform(self, X_test, usecols=None, alpha=None, dropcols=None):
        self._X = X_test
        self._train_bool = self._X.columns.isin(["outcome"]).sum()
        
        return self.drop_feature(dropcols).feature_transform().encoder(usecols, alpha).imputer()
