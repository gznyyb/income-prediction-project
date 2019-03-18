# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import re

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders.ordinal import OrdinalEncoder


def read_data(data_path):    
    data = pd.read_csv(data_path)
    return data
    
def feature_target_split(dataset):
    y = dataset["Target"]
    X = dataset.drop("Target", axis=1)
    return X, y

def cat_num_feature_split(dataset):
    X_num = dataset[dataset.columns[(dataset.dtypes=="int64") | (dataset.dtypes=='float64')]]
    X_num = X_num.astype(dtype='float64')
    X_cat = dataset[dataset.columns[dataset.dtypes=="object"]]
    return X_num, X_cat
    
def drop_column(dataset, columns_to_drop):
    try:
        X_drop = dataset.drop(columns_to_drop, axis=1)
        return X_drop
    except Exception:
        print("please enter a valid list for the columns_to_drop argument to specify the columns to be dropped")

def filled_missing(dataset, fill_in='missing_val'):
    try:
        X_filled_in = dataset.fillna('missing_val')
        return X_filled_in 
    except Exception:
        print("please enter a valid value for the fill_in argument such that the missing values can be encoded")
        
def data_preprocess_first_way(data_path):
    data = read_data(data_path)
    X, y = feature_target_split(data)
    y = pd.Series(map(lambda x: re.sub('\.', '', x), y))
    X_num, X_cat = cat_num_feature_split(X)
    X_num_prepared = drop_column(X_num, ['Capital_Gain', 'Capital_Loss'])
    X_cat_prepared = filled_missing(X_cat, 'missing_val')
    X_prepared = pd.concat([X_cat_prepared, X_num_prepared], axis=1)
    y_prepared = pd.Series(map(int, y == ' >50K'))
    return X_prepared, y_prepared
        
def get_preprocess_pipeline(X, is_one_hot_encode=True):
    num_attribs = list(cat_num_feature_split(X)[0].columns)
    cat_attribs = list(cat_num_feature_split(X)[1].columns)
    
    pipeline_one_hot_encode = ColumnTransformer([
            ("num", StandardScaler(), num_attribs),
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse=False), cat_attribs),
        ])
    pipeline_label_encode = ColumnTransformer([
            ("num", StandardScaler(), num_attribs),
            ("cat", OrdinalEncoder(), cat_attribs),
        ])
    if is_one_hot_encode:
        return pipeline_one_hot_encode
    else:
        return pipeline_label_encode
        
    
    






