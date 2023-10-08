import pandas as pd
import numpy as np
import config
from functools import partial
from scipy.optimize import fmin

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from scipy import stats

def data_loader():
    """ read csv data from input folder and return four data.
    Parameters:
    None
    
    Returns:
    train, test, origin, submission (pd.DataFrame) : datas
    """
    train = pd.read_csv(config.TRAINING_FILE)
    test = pd.read_csv(config.TEST_FILE)
    origin = pd.read_csv(config.ORIGINAL_FILE)
    submission = pd.read_csv(config.SUBMISSION_FILE)
    
    return train, test, origin, submission

def preprocessing(df: pd.DataFrame, le_cols : list, ohe_cols :list):
    """Function to Encode Categorical Features by Label, Ordinal, OneHot.
    The Ordinal Features are fixed.
    Parameters:
    df (pd.DataFrame): Pandas DataFrame you want to Encode
    le_cols (list) : List that you want to put in LabelEncoder
    ohe_cols (list) : List that you want to put in OneHotEncoder
    
    Returns:
    df (pd.DataFrame) : Transformed input
    """
    # Label Encoding
    le = LabelEncoder()
    for col in le_cols:
        df[col] = le.fit_transform(df[col])
        
    # OneHot Encoding    
    df = pd.get_dummies(df, columns = ohe_cols)
    
    # Transform Values
    df["pain"] = df["pain"].replace('slight', 'moderate')
    df["peristalsis"] = df["peristalsis"].replace('distend_small', 'normal')
    df["rectal_exam_feces"] = df["rectal_exam_feces"].replace('serosanguious', 'absent')
    df["nasogastric_reflux"] = df["nasogastric_reflux"].replace('slight', 'none')
    
    # Ordinal Encoding
    df["temp_of_extremities"] = df["temp_of_extremities"].fillna("normal").map({'cold': 0, 'cool': 1, 'normal': 2, 'warm': 3})
    df["peripheral_pulse"] = df["peripheral_pulse"].fillna("normal").map({'absent': 0, 'reduced': 1, 'normal': 2, 'increased': 3})
    df["capillary_refill_time"] = df["capillary_refill_time"].fillna("3").map({'less_3_sec': 0, '3': 1, 'more_3_sec': 2})
    df["pain"] = df["pain"].fillna("depressed").map({'alert': 0, 'depressed': 1, 'moderate': 2, 'mild_pain': 3, 'severe_pain': 4, 'extreme_pain': 5})
    df["peristalsis"] = df["peristalsis"].fillna("hypomotile").map({'hypermotile': 0, 'normal': 1, 'hypomotile': 2, 'absent': 3})
    df["abdominal_distention"] = df["abdominal_distention"].fillna("none").map({'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3})
    df["nasogastric_tube"] = df["nasogastric_tube"].fillna("none").map({'none': 0, 'slight': 1, 'significant': 2})
    df["nasogastric_reflux"] = df["nasogastric_reflux"].fillna("none").map({'less_1_liter': 0, 'none': 1, 'more_1_liter': 2})
    df["rectal_exam_feces"] = df["rectal_exam_feces"].fillna("absent").map({'absent': 0, 'decreased': 1, 'normal': 2, 'increased': 3})
    df["abdomen"] = df["abdomen"].fillna("distend_small").map({'normal': 0, 'other': 1, 'firm': 2,'distend_small': 3, 'distend_large': 4})
    df["abdomo_appearance"] = df["abdomo_appearance"].fillna("serosanguious").map({'clear': 0, 'cloudy': 1, 'serosanguious': 2})
    
    return df

def chi_squared_test(df : pd.DataFrame, input_var : str, target_var : str, significance_level=0.05):
    """Function to test a significant relationship between categorical_variables and target variable by chi_squared.
    Parameters:
    df (pd.DataFrame) : Pandas DataFrame you want to check test result
    input_var (str) : input feature name
    target_var (str) : target feature name
    significance_level (float) : threshold for chi_squared test
    
    Returns:
    print test result
    """
    contingency_table = pd.crosstab(df[input_var], df[target_var])
    chi2, p, _, _ = stats.chi2_contingency(contingency_table)
    
    if p < significance_level:
        print(f'\033[32m{input_var} has a significant relationship with the target variable.\033[0m') 
    else:
        print(f'\033[31m{input_var} does not have a significant relationship with the target variable.\033[0m')  

class OptimizeF1:
    """Optimize F1 Ensemble
    """
    
    def __init__(self):
        self._coef = 0
        
    def _f1(self, coef, X, y):
        """calculate f1 score and return
        
        Parameters:
        coef (list) : list of weights
        X (pd.DataFrame) : train data
        y (pd.Series) : target train data
        
        Return:
        -1 * f1 score
        """
        
        x_coef = X * coef
        
        predictions = np.sum(x_coef, axis=1)
        
        f1_score = f1_score(y, predictions, average='micro')
        
        return -1 * f1_score
    
    def fit(self, X, y):
        loss_partial = partial(self._f1, X=X, y=y)
        
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), 
                                           size=1)
        
        self._coef = fmin(loss_partial, initial_coef, disp=True)
    
    def predict(self, X):
        X_coef = X * self._coef
        
        predictions = np.sum(X_coef, axis=1)
        
        return predictions