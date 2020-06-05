"""
"""
import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, '..')
from src.pre_processing import *
import src.pre_processing as pre_processing

csv_path = '../data/bank-additional-full.csv'
my_pre_processing = pre_processing.load_data(csv_path)


@pytest.fixture(autouse=True)
def teardown():
    d = os.path.dirname(os.path.abspath('test_util.py'))
    d = d.split('/')
    if d[-1]!='test':
       os.chdir('test')

def test_load_data():
    '''
        Test for the load_data() method of pre_processing.py
    '''
    my_load_data_pre_processing = pre_processing.load_data(csv_path)
    assert isinstance(my_load_data_pre_processing,DfBankAdditional)

def test_process_all():
    '''
        Test for the process_all() method of DfBankAdditional class.
    '''
    my_pre_processing.process_all()
    assert isinstance(my_pre_processing.df,pd.DataFrame)

def test_re_map_column():
    '''
        Test for the re_map_column() method of DfBankAdditional class.
    '''
    try:
        for c in my_pre_processing.df.keys():
            if c in my_pre_processing.mappings.keys():
                my_pre_processing.re_map_column(c)
    except:
        print("Exception in re_map_column() method")

def test_validate_all():
    '''
        Test for the _validate_all() method of DfBankAdditional class.
    '''
    try:
        my_pre_processing._validate_all()
    except:
        print("Exception in _validate_all() method")

def test_validate():
    '''
        Test for the _validate() method of DfBankAdditional class.
    '''
    try:
        for c in my_pre_processing.df.keys():
            if c in my_pre_processing.mappings.keys():
                my_pre_processing._validate(c)
    except:
        print("Exception in _validate() method")
