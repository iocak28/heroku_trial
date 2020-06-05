import pytest
import sys
sys.path.insert(0, '..')
import pandas as pd
import src.prediction as pred
import os
from src.feature_extraction import *

def test_column_eliminator():
    '''
    Tests the column_eliminator() function in prediction.py
    '''
    #os.chdir('..')
    feature_extraction = FeatureExtractor()
    X_train, X_test, y_train, y_test= feature_extraction.get_train_test_split(test_size = 0.2, random_state = 1)
    raw_col = X_train.shape[1]
    
    X_train_new = pred.column_eliminator(X_train)
    col_elim = X_train_new.shape[1]
    
    assert raw_col>=col_elim
    assert type(X_train_new) == pd.DataFrame
    
def test_make_predictions():
    '''
    Tests the make_predictions() function in prediction.py
    '''
    os.chdir('..')
    assert type(pred.make_predictions()) == pd.DataFrame
    
    

