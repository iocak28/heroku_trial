import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from xgboost import XGBClassifier
from sklearn.externals import joblib

from src import feature_extraction as feat
from src import pre_processing as pp

# Initialize a FeatureExtractor object
feature_extractor = feat.get_feature_extractor()

# Get Preproccess Raw data (For Data Analysis)
preprocessed_raw = feature_extractor.load_preprocessed_data()
# Get Train and Test data
X_train_raw, X_test_raw, y_train, y_test = feature_extractor.get_train_test_split(test_size = 0.2, 
                                                                                  random_state = 1,
                                                                                  scale = False)

# Convert y_train and y_test to 1-d array
y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)


# Hand pick some features
best_feat = ['nr.employed', 'poutcome_success', 'emp.var.rate', 'pdays', 'cons.conf.idx', 'euribor3m', 'job_transformed_no_income']

X_train = X_train_raw[best_feat]
X_test = X_test_raw[best_feat]

# Logistic regression with these features
clf = LogisticRegression(random_state=0, max_iter = 1000)

clf.fit(X_train, y_train)
joblib.dump(clf, 'LR_prediction.joblib')
pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)
accuracy = accuracy_score(y_test, pred)
conf = confusion_matrix(y_test, pred)

# enter the params and get the predicted probability
def dynamic_predict(nr_employed=0, poutcome_success=0, emp_var_rate=0, pdays=0, cons_conf=0, euribor=0, no_income=0):
    '''
    Create predictions for new customers. Enter the inputs and get the probability.
    
    :param nr_employed: number of employees - quarterly indicator (numeric)
    :type nr_employed: numpy.ndarray or list or int or float
    :param poutcome_success: outcome of the previous marketing campaign (binary: 1 if success, 0 otherwise)
    :type poutcome_success: numpy.ndarray or list or int or float
    :param emp_var_rate: employment variation rate - quarterly indicator (numeric)
    :type emp_var_rate: numpy.ndarray or list or int or float
    :param pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    :type pdays: numpy.ndarray or list or int or float
    :param cons_conf: consumer confidence index - monthly indicator (numeric)
    :type cons_conf: numpy.ndarray or list or int or float
    :param euribor: euribor 3 month rate - daily indicator (numeric)
    :type euribor: numpy.ndarray or list or int or float
    :param no_income: binary income indicator, 1 if the customer job retired, student or unemployed
    :type no_income: numpy.ndarray or list or int or float
    :return: Array of predicted probabilities
    :rtype: numpy.ndarray
    '''
    #assert len(nr_employed) == len(poutcome_success) == len(emp_var_rate)
    assert isinstance(nr_employed, (list, int, float, np.ndarray))
    
    nr_employed = np.array(nr_employed)
    poutcome_success = np.array(poutcome_success)
    emp_var_rate = np.array(emp_var_rate)
    pdays = np.array(pdays)
    cons_conf = np.array(cons_conf)
    euribor = np.array(euribor)
    no_income = np.array(no_income)
    
    features = pd.DataFrame({'nr.employed' : nr_employed, 
                             'poutcome_success': poutcome_success, 
                             'emp.var.rate' : emp_var_rate, 
                             'pdays' : pdays,
                             'cons.conf.idx' : cons_conf, 
                             'euribor3m' : euribor, 
                             'job_transformed_no_income' : no_income}, 
    index = range(1))

    probs_new = clf.predict_proba(features)[:, 1]

    return probs_new


# some examples
    
# dynamic_predict(nr_employed = [5099], 
#                 poutcome_success = [0], 
#                 emp_var_rate = [-1.8], 
#                 pdays = [999], 
#                 cons_conf = [-46.2], 
#                 euribor = [1.244], 
#                 no_income = [0])

# dynamic_predict(nr_employed = [5099, 5195.8], 
#                 poutcome_success = [0, 1], 
#                 emp_var_rate = [-1.8, -0.1], 
#                 pdays = [999, 6], 
#                 cons_conf = [-46.2, -42], 
#                 euribor = [1.244, 4.076], 
#                 no_income = [0, 0])