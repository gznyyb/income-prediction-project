# -*- coding: utf-8 -*-

# import necessary packages for dataframe and string manipulation
import pandas as pd
import re

# import necessary packages for preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders.ordinal import OrdinalEncoder


def read_data(data_path): 
    '''Function to read data from the .csv file

    Arguments
    ---------
    data_path: the path to the .csv file

    Returns
    -------
    data: the dataframe containing the censor data
    '''   
    data = pd.read_csv(data_path)
    return data
    
def feature_target_split(dataset):
	'''Function to separate the features and target

	Arguments
	---------
	dataset: the dataframe containing the censor data

	Returns
	-------
	X: the dataframe containing the features
	y: the dataframe containing the targets
	'''
	y = dataset["Target"]
	X = dataset.drop("Target", axis=1)
	return X, y

def cat_num_feature_split(dataset):
	'''Function to separate the categorical and numeric features

	Arguments
	---------
	dataset: the dataframe containing the features

	Returns
	-------
	X_num: the dataframe containing the numeric features
	X_cat: the dataframe containing the categorical features
	'''
	X_num = dataset[dataset.columns[(dataset.dtypes=="int64") | (dataset.dtypes=='float64')]]
	X_num = X_num.astype(dtype='float64')
	X_cat = dataset[dataset.columns[dataset.dtypes=="object"]]
	return X_num, X_cat

def drop_column(dataset, columns_to_drop):
	'''Function to drop column

	Arguments
	---------
	dataset: dataframe containing the data
	columns_to_drop: columns to drop from the dataframe

	Returns
	-------
	X_drop: dataframe with the columns dropped
	'''
	X_drop = dataset.drop(columns_to_drop, axis=1)
	return X_drop

def filled_missing(dataset, fill_in='missing_val'):
	'''Function to do missing values imputation

	Arguments
	---------
	dataset: the dataframe containing the data

	Returns
	-------
	X_filled_in: the dataframe with missing values substituted with other values
	'''
	X_filled_in = dataset.fillna('missing_val')
	return X_filled_in 

def data_preprocess_first_way(data_path):
	'''Function that suggests one way to preprocess the data

	Arguments
	---------
	data_path: the path to the .csv file

	Returns
	-------
	X_prepared: the preprocessed dataframe containing the features
	y_prepared: the preprocessed dataframe containing the targets
	'''
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
	'''Function to do label encoding and standardization

	Arguments
	---------
	X: the dataframe containing the data
	is_one_hot_encode: whether to do one hot encoding or label encoding of the categorical features

	Returns
	-------
	pipeline_one_hot_encode: the sklearn pipeline to do one hot encoding of the categorical features and standardization of the numeric features
	or
	pipeline_label_encode: the sklearn pipeline to do label encoding of the categorical features and standardization of the numeric features
	'''
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









