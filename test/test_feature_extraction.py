import pytest
import sys
sys.path.insert(0, '..')
import os
import pandas as pd

from src.feature_extraction import *
import src.feature_extraction as feature_ex

feature_extraction = FeatureExtractor()
@pytest.fixture(autouse=True)
def teardown():
    d = os.path.dirname(os.path.abspath('test_util.py'))
    d = d.split('/')
    if d[-1]!='test':
       os.chdir('test')

def test_load_preprocessed_data():
    '''
    Tests lead_preprocessed_data() function in feature_extraction.py.
    '''
    p = feature_extraction.load_preprocessed_data()
    assert type(p) == pd.DataFrame
    
def test_data_scaler():
    '''
    Tests data_scaler() function in feature_extraction.py.
    '''
    assert type(feature_extraction.data_scaler(feature_extraction.load_preprocessed_data())) == pd.DataFrame

def test_one_hot_encoder():
    '''
    Tests the one_hot_encoder() function in feature_extraction.py.
    '''
    encoding = feature_extraction.data_scaler(feature_extraction.data_scaler(feature_extraction.load_preprocessed_data()))
    assert type(encoding) == pd.DataFrame
    
def test_get_features():
    '''
    Tests the get_features() function in feature_extraction.py..
    '''
    features = feature_extraction.get_features()
    assert type(features) == pd.DataFrame

def test_get_train_test_split():
    '''
    Tests the get_train_test_split() function in feature_extraction.py..
    '''

    X_train, X_test, y_train, y_test = feature_extraction.get_train_test_split()

    assert type(X_train) == pd.DataFrame and  type(X_test) == pd.DataFrame
    assert type(y_train) == pd.DataFrame and  type(y_test) == pd.DataFrame
    assert X_train.shape == (32950, 73) and y_train.shape == (32950, 1)
    assert X_test.shape == (8238, 73) and y_test.shape == (8238, 1)


def test_get_feature_extractor():
    '''
    Tests the get_feature_extractor() function in feature_extraction.py.
    '''

    f = feature_ex.get_feature_extractor()
    assert isinstance(f,FeatureExtractor)



