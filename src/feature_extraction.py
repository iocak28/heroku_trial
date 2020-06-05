# Import Python Libraries
import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src import pre_processing as pp

class FeatureExtractor:
    '''
    Feature extractor class for the bank data set. Calls pre_processing code and pre-processes the data. Then conducts the feature extraction tasks.
    :param wd: The default working directory. It should be set to the location of the src folder. For example: 'C:/Users/iocak/Desktop/git/ECE229-Project/src'
    :type wd: str, optional
    :param file_path: Location of the csv data file. Pass the location of 'bank-additional-full.csv' if you would like to change the default data path.
    :type file_path: str, optional
    '''
    def __init__(self, wd = '../', filepath = 'data/bank-additional-full.csv'):
        """
        Constructor method
        """
        self.wd = wd
        self.filepath = os.path.join(self.wd,filepath)
        
    def load_preprocessed_data(self):
        '''
        Imports pre_processing library from src folder and loads the preprocessed data. If the change_wd option is True in the class constructor then imports pre_processing library from the new location else uses the default working directory. While reading the data uses the filepath string that was passed in the class constructor.
        
        :return: 'df' which is the preprocessed Pandas dataframe.
        :rtype: pandas.core.frame.DataFrame
        '''
        
        # read the data
        bank_data = pp.load_data(self.filepath)
        bank_data.process_all()
        df = bank_data.df
        
        # assign id
        df['customer_id'] = np.arange(0, len(df), 1)
        
        return df
    
    def data_scaler(self, prep, scale_time = False):
        '''
        Normalize the numerical variables. Subtract mean and divide by standard deviation.
        
        :param prep: A pandas dataframe of preprocessed data.
        :type prep: pandas.core.frame.DataFrame
        :param scale_time: If True scales time columns(month, day_of_week) as well, default is False.
        :type scale_time: bool, optional
        :return: A pandas dataframe with the scaled numerical features.
        :rtype: pandas.core.frame.DataFrame
        '''
        assert isinstance(prep, pd.core.frame.DataFrame)
        assert isinstance(scale_time, bool)
        
        cols = prep.dtypes
        cols = cols[(cols != 'object')]
        cols = cols[cols.index != 'y']
        cols = cols[cols.index != 'customer_id']
        
        if scale_time == False:
            cols = cols[~cols.index.isin(['month', 'day_of_week'])]
        
        # Initialize scaler
        scaler = StandardScaler()
        # Scale only desired columns
        scaler.fit(prep[cols.index])
        # Perform transformation
        prep[cols.index] = scaler.transform(prep[cols.index])
        
        return prep
            
    def one_hot_encoder(self, df, encode_time = True):
        '''
        Do one hot encoding on the provided dataframe
        
        :param df: A pandas dataframe of bank data to be one-hot-encoded.
        :type df: pandas.core.frame.DataFrame
        :param encode_time: Default is True. If True the columns month and day_of_week are one_hot_encoded too. Otherwise they are kept as integers.
        :type encode_time: bool, optional
        :return: A pandas dataframe with the scaled numerical features.
        :rtype: pandas.core.frame.DataFrame
        '''
        assert isinstance(df, pd.core.frame.DataFrame)
        assert isinstance(encode_time, bool)
        
        if encode_time == True:
            df['month'] = df['month'].astype(str)
            df['day_of_week'] = df['day_of_week'].astype(str)
        
        one_hot = pd.get_dummies(df)
        
        return one_hot
    
    def get_features(self, scale = True):
        '''
        Get final features. This method first preprocesses the data and then does the feature extraction. Finally it one-hot-encodes the data. These processes are done by using class methods.
        
        :param scale: True by default, scales the features.
        :type scale: bool
        :return: Final features to be used in machine learning tasks. (Pandas DataFrame)
        :rtype: pandas.core.frame.DataFrame
        '''
        
        # get the preprocessed data
        prep = self.load_preprocessed_data()
        
        # scaling for numeric variables
        if scale == True:
            features = self.data_scaler(prep)
        else:
            features = prep.copy()
        # one hot encoding
        features_final = self.one_hot_encoder(features)

        return features_final
    
    def get_train_test_split(self, test_size = 0.2, random_state = 1, scale = True):
        '''
        Train test splitting with desired parameters. Uses the result of get_features() method internally.
        
        :param test_size: Test data size as the fraction of whole data size.
        :type test_size: float, optional
        :param random_state: Random state to be used in splitting.
        :type random_state: float, optional
        :param scale: True by default, scales the features.
        :type scale: bool, optional
        :return: Train features, train labels, test features, test labels are returned in separate Pandas Dataframes in a tuple size of 4.
        :rtype: tuple
        '''
        assert isinstance(test_size, (float, int))
        assert isinstance(random_state, (int))
        
        features_final = self.get_features(scale = scale)
        
        X_train, X_test, y_train, y_test = train_test_split(features_final.loc[:, features_final.columns != 'y'], 
                                                            features_final.loc[:, features_final.columns == 'y'], 
                                                            test_size=test_size, 
                                                            random_state=random_state)
        
        return X_train, X_test, y_train, y_test

def get_feature_extractor(wd = '', filepath = 'data/bank-additional-full.csv'):
    '''
    Kickstart the FeatureExtractor class.
    
    :param wd: The default working directory. It should be set to the location of the src folder. For example: 'C:/Users/iocak/Desktop/git/ECE229-Project/src'
    :type wd: str, optional
    :param file_path: Location of the csv data file. Pass the location of 'bank-additional-full.csv' if you would like to change the default data path.
    :type file_path: str, optional
    :return: Returns a FeatureExtractor object.
    :rtype: class:`FeatureExtractor` 
    '''
    
    assert isinstance(wd, str)
    assert isinstance(filepath, str)
    
    return FeatureExtractor(wd, filepath)



