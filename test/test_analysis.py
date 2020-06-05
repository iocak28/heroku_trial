'''
Test cases for analysis.py file where we perform preliminary analysis on the data.
'''
import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, '..')
import visualization.analysis as analysis
from visualization.analysis import *

csv_path = '../data/bank-additional-full.csv'
my_analysis = Analysis(csv_path)
marital_analysis = MaritalAnalysis(csv_path)
feature = FeatureAnalysis(csv_path)

@pytest.fixture(autouse=True)
def teardown():
    d = os.path.dirname(os.path.abspath('test_util.py'))
    d = d.split('/')
    if d[-1]!='test':
       os.chdir('test')

def test_get_column():
    '''
    Tests get_column() fucntion in analysis.py. The get_column() function is used to retrieve a column from the dataset.
    The get_column function accepts the column name as an argument and returns the column from the dataframe.
    '''
    marital_status = ['married', 'single', 'divorced', 'unknown']
    assert all(i in marital_status for i in my_analysis.get_column(column = 'marital').unique())  == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'age')) == True
    month = [ 5,  6,  7,  8, 10, 11, 12,  3,  4,  9]
    assert all(i in month for i in my_analysis.get_column(column ='month').unique())
    job_status = ['housemaid', 'services', 'admin', 'blue-collar', 'technician',
    'retired', 'management', 'unemployed', 'self-employed', 'unknown',
    'entrepreneur', 'student']
    assert all(i in job_status for i in my_analysis.get_column('job').unique()) == True
    contact_type = ['telephone', 'cellular']
    assert all(i in contact_type for i in my_analysis.get_column(column ='contact').unique()) == True
    credit_default = ['no', 'unknown', 'yes']
    assert all(i in credit_default for i in my_analysis.get_column(column ='default').unique()) == True
    day_of_week = [0, 1, 2, 3, 4]
    assert all(i in day_of_week for i in my_analysis.get_column(column ='day_of_week').unique())== True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'duration')) == True
    assert all(isinstance(i, (int)) for i in my_analysis.get_column(column = 'campaign')) == True
    assert all(isinstance(i, (int)) for i in my_analysis.get_column(column = 'pdays')) == True
    assert all(isinstance(i, (int)) for i in my_analysis.get_column(column = 'previous')) == True
    poutcome = ['nonexistent', 'failure', 'success']
    assert all(i  for i in my_analysis.get_column(column = 'poutcome')) == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'emp.var.rate')) == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'cons.price.idx')) == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'cons.conf.idx')) == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'euribor3m')) == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'nr.employed')) == True
     
     
def test_get_probabilities():
    '''
    Tests get_probabilities() fucntion in analysis.py. The get_probabilities() function is used to compute the probability of a customer subscribing to the term deposit plan.
    The get_probabilities function accepts the column name as an argument and returns the probability of a customer saying yes to the term deposit plan depending on the different categorical values that the column takes.
    '''
    p = my_analysis.get_probabilities('marital')
    assert all(1>=i>=0 for i in p['y']) == True
    assert all(isinstance(i,(int,float)) for i in p['y']) == True

def test_get_success_count():
    '''
    Tests get_success_count() fucntion in analysis.py. The get_success_count() function is used to compute the number of customers subscribing to the term deposit plan.
    The get_success_count() function accepts the column name as an argument and returns the number of customers subscribing to the term deposit plan depending on the different categorical values that the column takes.
    The test_get_success_count() checks if the value returned is expected by ensuring that the number returned is greater than 0 and the returned values are ints.
    '''
    p = my_analysis.get_success_count('marital')
    assert type(p) == pd.DataFrame
    assert all(i>=0 for i in p['y']) == True
    assert all(isinstance(i,int) for i in p['y']) == True
    
def test_get_count():
    '''
    Tests get_count() fucntion in analysis.py. The get_count() fucntion is used to compute the total number of customers contacted.
    The get_count() function accepts the column name as an argument and returns the total number of customers contacted depending on the different categorical values that the column takes.
    The test_get_count() checks if the value returned is expected by ensuring that the number returned is greater than 0 and the returned values are ints.
     
    '''
    p = my_analysis.get_count('marital')
    assert type(p) == pd.DataFrame
    assert all(i>=0 for i in p['y']) == True
    assert all(isinstance(i,int) for i in p['y']) == True


def test_percentage_of_population():
    '''
    Tests percentage_of_population() fucntion in analysis.py. This function is used to compute the percentage of customers subscribing to the term deposit plan.
    The percentage_of_population() function accepts the column name as an argument and returns the percentage of customers subscribing to the term deposit plan depending on the different categorical values that the column takes.
    The test_percentage_of_population() checks if the percentage value returned is expected by ensuring that the percentage is in between 0 and 100 and the returned values are either floats or ints.
    '''
    p = my_analysis.percentage_of_population('marital')
    assert all(100>=i>=0 for i in p) == True
    assert all(isinstance(i,(int, float)) for i in p) == True
    
def test_map_age():
    '''
    Tests map_age() function in analysis.py. The mapping of age to age group is done by mapping age to labels. This function checks if the labels and mapping have consistent data types.
    '''
    myList, labels = my_analysis.map_age()
    assert isinstance(myList,list) and isinstance(labels, list)
    
def test_get_age_prob_success():
    '''
    Tests get_age_prob_success() fucntion in analysis.py. The get_age_prob_success() function is used to compute the probability of a person subscribing to the term deposit based on the age group they fall into.
    The test_get_age_prob_success() checks if the percentage value returned is expected by ensuring that the percentage is in between 0 and 100 and the returned values are either floats or ints.
    '''
    myList, labels = my_analysis.map_age()
    p = my_analysis.get_age_prob_success(myList)
    assert isinstance(p, list)
    assert all(0<=i<=100 for i in p)
    assert all(isinstance(i, (int, float)) for i in p)

def test_filter_unknown_marital():
    '''
    Tests filter_unknown_marital() fucntion in analysis.py. The filter_unknown_marital() function is used to filter out columns that have marital status field set to unknown.
    The test_filter_unknown_marital() checks to ensure that the filtered values do not contain unknown marital status.
    '''
    
    k = marital_analysis.get_count('marital')['marital']
    status = ['married', 'divorced', 'single']
    assert all(i in status for i in k)==True

def test_get_feature_importance():
    '''
    Tests get_feature_importance() fucntion in analysis.py. The get_feature_importance() function is used get a dataframe of feature importance
    The test_get_feature_importance() checks to ensure that the returned values are a pandas DataFrame object.
    '''
   
    importance = feature.get_feature_importance()
    assert type(importance) == pd.DataFrame
    
def test_number_to_day_of_week():
    '''
    Tests number_to_day_of_week() fucntion in analysis.py. The number_to_day_of_week() function is used return the days of the week corresponding to the enumeration of days given in the dataframe.
    The test_number_to_day_of_week() checks to ensure that the returned values are days of the week and there are no stray values that are being returned.
    '''
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_of_week_probabilities = my_analysis.get_probabilities('day_of_week')
    num_to_day = analysis.number_to_day_of_week(day_of_week_probabilities['day_of_week'])
    assert all(i in days for i in num_to_day)==True

def test_number_to_month():
    '''
    Tests number_to_month() fucntion in analysis.py. The number_to_month() function is used return the months corresponding to the enumeration of months given in the dataframe.
    The test_number_to_month() checks to ensure that the returned values are months and there are no stray values that are being returned.
    '''
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_probabilities = my_analysis.get_probabilities('month')
    num_to_month = analysis.number_to_month(month_probabilities['month'])
    assert all(i in months for i in num_to_month)==True


