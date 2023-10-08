from pycaret.classification import *
import pandas as pd

def get_pycaret_model(data : pd.DataFrame, target : str):
    """Automated ML technique by PyCaret.
    Parameters:
    data (pd.DataFrame) : data for training. data must have target feature.
    target (str) : target feature name.
    
    Return:
    model compare results.
    """
    s = setup(data, target = target, fold = 5)
    best_model = compare_models()
    
    return best_model