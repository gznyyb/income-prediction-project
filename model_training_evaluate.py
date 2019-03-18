# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:58:05 2019

@author: gznyy
"""

import warnings


import cross_val_helper_funcs as hf
import data_clean_preprocess as dcp

from sklearn.metrics import roc_auc_score, accuracy_score

def prepare_cross_validate_train_data(X_train, y_train, estimator_names):
	_, estimators, search_spaces = hf.get_ml_pipeline(X_train, ml_pipelines_list=estimator_names)

	print("nested cross validation started")
	result_estimator_dict, results = hf.nested_cross_validate(X_train, y_train, estimator_names, 
	                                                          estimators, search_spaces, verbose=1, n_points=4)
	print(_)
	return result_estimator_dict, results

def test_chosen_estimator(X_train, y_train, test_data_path, result_estimator_dict, chosen_estimator_name):
	print("fitting chosen estimator on the whole training data")
	chosen_estimator = result_estimator_dict[chosen_estimator_name]
	chosen_estimator.fit(X_train, y_train)

	X_test_prepared, y_test_prepared = dcp.data_preprocess_first_way(test_data_path)
	y_pred = chosen_estimator.predict(X_test_prepared)
	print("the final validation roc score is:", chosen_estimator.best_score_)
	print("the final test roc score is: ", roc_auc_score(y_test_prepared, y_pred))
	print("the final test accuracy score is:", accuracy_score(y_test_prepared, y_pred))

if __name__=="__main__":
	warnings.simplefilter("ignore")
	X_train_prepared, y_train_prepared = dcp.data_preprocess_first_way('adult_train.csv')
	estimator_names = ['svc pca', 'logistic regression', 'linear svc', 'svc', 'rfc', 'xgboost', 'light gbm', 'neural network'][7:8]
	result_estimator_dict, results = prepare_cross_validate_train_data(X_train_prepared, y_train_prepared, estimator_names)

	chosen_estimator_name = input("please choose one estimator according to the dictionary keys given above. Note that the input must match exactly with one of the dictionary keys: ")
	test_chosen_estimator(X_train_prepared, y_train_prepared, 'adult_test.csv', result_estimator_dict, chosen_estimator_name)














