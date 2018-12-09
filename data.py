from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals.joblib import dump
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
import pandas as pd
import numpy as np

"""
Import data from a CSV file using pandas IO, split into test/train datasets using Sklearn test_train_split

Args:
    file: path to data file in csv format
    split: training fraction

Return:
    x_train: np.array of feature data
    x_test: np.array of test data
    y_train: np.array of training class labels
    y_test: np.array of testing class labels
"""
def parse_data(file, split=0.3):
    np.random.seed(0)

    df = pd.read_csv(file)

    # normalize the amount, all other features are already normalized by the dataset publisher
    df['NORM_AMOUNT'] = StandardScaler().fit_transform(df['AMOUNT'].values.reshape(-1, 1))

    # use all Vxx features (results of PCA, attribute name anonymized), and normalized transaction amount
    features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 
                'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'NORM_AMOUNT']
    classes = ['FRAUD']

    x = df[features].values
    y = df[classes].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split)
    
    return x_train, x_test, y_train, y_test

"""
Import data from a CSV file using pandas IO, split into test/train datasets using Sklearn.test_train_split
and oversample the training data using Imbalern.RandomOverSampler

Args:
    file: path to data file in csv format
    split: training fraction
    minority: fraction of minority class in training data

Return:
    x_resample: np.array of feature data with oversampling
    x_test: np.array of test data
    y_resample: np.array of training class labels with oversampling
    y_test: np.array of testing class labels
"""
def parse_data_random_oversample(file, split=0.3, minority=0.10):
    # use naive implementation to get initial test/train split
    x_train, x_test, y_train, y_test = parse_data(file)

    # resample training data using random oversampling
    x_resample, y_resample = RandomOverSampler(sampling_strategy=minority).fit_resample(x_train, y_train.ravel())

    return x_resample, x_test, y_resample, y_test

"""
Import data from a CSV file using pandas IO, split into test/train datasets using Sklearn.test_train_split
and oversample and undersample the training data using Imbalern.SMOTEENN. Saves the returned values to disk 
using Sklearn.joblib serialization

Args:
    file: path to data file in csv format
    split: training fraction
    minority: fraction of minority class in training data

Return:
    x_resample: np.array of feature data with oversampling
    x_test: np.array of test data
    y_resample: np.array of training class labels with oversampling
    y_test: np.array of testing class labels
"""
def parse_data_SMOTEENN(file, split=0.3, minority=0.10):
    # use naive implementation to get initial test/train split
    x_train, x_test, y_train, y_test = parse_data(file)

    # resample training data using SMOTE for oversampling and edited nearest neighbor distance (ENN) for undersampling
    x_resample, y_resample = SMOTEENN(sampling_strategy=minority).fit_resample(x_train, y_train.ravel())

    # SMOTEENN is slow, save the results to disk for later use
    dump(x_resample, 'x_resample_SMOTEENN.joblib')
    dump(x_test, 'x_test_SMOTEENN.joblib')
    dump(y_resample, 'y_resample_SMOTEENN.joblib')
    dump(y_test, 'y_test_SMOTEENN.joblib')

    return x_resample, x_test, y_resample, y_test
