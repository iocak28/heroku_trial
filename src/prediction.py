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
from xgboost import XGBClassifier


def column_eliminator(data):
    '''
    Eliminate columns from the feature data
    
    :param prep: A pandas dataframe of the feature data.
    :type prep: pandas.core.frame.DataFrame
    :return: A pandas dataframe with the final features to be used in prediction.
    :rtype: pandas.core.frame.DataFrame
    '''
    assert isinstance(data, pd.core.frame.DataFrame)
    
    # drop customer_id (no predictive info) and duration (unrealistic feature)
    data = data.copy()
    data.drop(['customer_id', 'duration'], axis = 1, inplace = True)
    
    return data

def make_predictions(model = 'xgb', tune = False, save = False, wd = ''):
    '''
    Train a desired model and make predictions on the test data
    
    :param model: Name of the model to be used. Possible values: 'lr', 'rf', 'xgb'
    :type model: str
    :param tune: False by default, make it True if you want to tune the hyper parameters
    :type tune: bool
    :param save: False by default, make it True if you want to save the result, the results will be saved to /data folder.
    :type save: bool
    :param wd: Path of the working directory, you need to provide the location of the git folder.
    :type wd: str
    :return: Predictions on the test data with the original test features.
    :rtype: pandas.core.frame.DataFrame
    '''
    
    # Import feature extraction code
    try:
        import src.feature_extraction as feat
    except:
        os.chdir(wd)
        import feature_extraction as feat
    
    # Initialize a FeatureExtractor object
    feature_extractor = feat.get_feature_extractor()
    # Get Preprocessed Raw data (For Data Analysis)
    preprocessed_raw = feature_extractor.load_preprocessed_data()

    # Get Train and Test data
    X_train_raw, X_test_raw, y_train, y_test = feature_extractor.get_train_test_split(test_size = 0.2, 
                                                                                      random_state = 1)
    
    # Convert y_train and y_test to 1-d array
    y_train = np.array(y_train).reshape(-1)
    y_test = np.array(y_test).reshape(-1)
    
    # Filter Feature Columns
    X_train = column_eliminator(X_train_raw)
    X_test = column_eliminator(X_test_raw)
    
    # Training and Prediction with Different Models
    if model == 'lr':
        clf = LogisticRegression(random_state=0, max_iter = 1000)
    if model == 'rf':
        if tune == True:
            param_grid_rf = {
                         'n_estimators': [50],
                         'max_depth': [5],
                         'min_samples_split': [2],
                         'min_samples_leaf': [1],
                         'n_jobs': [-1]
#                         'n_estimators': [100, 200],
#                         'max_depth': [5, 25, 50],
#                         'min_samples_split': [2, 5],
#                         'min_samples_leaf': [1, 10, 20],
#                         'n_jobs': [-1]
                     }
            
            clf_rf = RandomForestClassifier()
            grid_clf_rf = GridSearchCV(clf_rf, param_grid_rf, cv=5, n_jobs = -1)
            grid_clf_rf.fit(X_train, y_train)
            
            best_params_rf = grid_clf_rf.best_params_
        else:
            best_params_rf =  {'max_depth': 25,
                                'min_samples_leaf': 10,
                                'min_samples_split': 2,
                                'n_estimators': 100,
                                'n_jobs': -1}
        clf = RandomForestClassifier(**best_params_rf)
    if model == 'xgb':
        if tune == True:
            param_grid_xgb = {
                         'learning_rate': [0.3],
                         'max_depth': [5],
                         'min_child_weight': [1],
                         'gamma': [0],
                         'subsample': [0.7],
                         'colsample_bytree': [0.7],
                         'n_jobs': [-1],
                         'lambda': [1]
#                         'learning_rate': [0.001, 0.1, 0.3],
#                         'max_depth': [5, 25],
#                         'min_child_weight': [1, 3],
#                         'gamma': [0, 0.1],
#                         'subsample': [0.7],
#                         'colsample_bytree' : [0.7],
#                         'n_jobs': [-1],
#                         'lambda': [0, 1]
                     }
            
            clf_xgb = XGBClassifier()
            grid_clf_xgb = GridSearchCV(clf_xgb, param_grid_xgb, cv=5, n_jobs = -1)
            grid_clf_xgb.fit(X_train, y_train)
            
            best_params_xgb = grid_clf_xgb.best_params_
        else:
            best_params_xgb =  {'learning_rate': 0.1,
                                'max_depth': 5,
                                'min_child_weight': 1,
                                'gamma': 0.1,
                                'subsample': 0.7,
                                'colsample_bytree': 0.7,
                                'n_jobs': -1,
                                'reg_lambda': 1}
        clf = XGBClassifier(**best_params_xgb)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, pred)
    conf = confusion_matrix(y_test, pred)
    
    df_pred = X_test_raw[['customer_id']].copy()
    df_pred['pred'] = pred
    df_pred['prob_1'] = probs[:, 1]
    
    df_pred = pd.merge(preprocessed_raw, df_pred, how = 'inner', on = 'customer_id')
    df_pred.drop(columns = ['duration'], axis = 1, inplace = True)
    
    if model != 'lr':
        feat_imp = pd.DataFrame({'features' : X_train.columns,
                                    'importance' : clf.feature_importances_})
                
    if save == True:
        df_pred.to_csv('data/predictions.csv', index = False)
        
        if model != 'lr':
            feat_imp.to_csv('data/feature_importance.csv', index = False)
                
    return df_pred            
    
def main():
    # change wd if necessary
    # os.chdir('C:/Users/iocak/Desktop/git/ECE229-Project/')
    preds_with_features = make_predictions(model = 'xgb', tune = False, save = True, wd = '')
    
    
