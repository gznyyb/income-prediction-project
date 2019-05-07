# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 08:59:07 2019

@author: gznyy
"""

# import data_clean_preprocess.py for functions to do preprocessing
import data_clean_preprocess as dcp

# import necessary packages to do preprocessing
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# import necessary packages for modeling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier 

# import necessary packages for model selection
from skopt import BayesSearchCV
from sklearn.model_selection import cross_validate, StratifiedKFold


def nested_cross_validate(X, y, estimator_names, estimators, search_spaces, verbose=0, n_points=1, 
	inner_cv=2, outer_cv=2, scoring='roc_auc'):
	'''Function to do nested cross validation to choose the best classification algorithm with the best set of hyperparameters

	Arguments
	---------
	X: the features dataframe
	y: the targets dataframe
	estimator_names: the names assigned to different classification algorithm
	estimators: the sklearn estimator objects that represent the candidates from which we will choose the best classification algorithm
	search_spaces: the dictionary that specifies the range of values through which we will search for the best set of hyperparameters
	verbose: the verbosity level
	n_points: the number of parameter settings to sample in parallel
	inner_cv: the number of inner folds that the cross validation algorithm is going to divide the dataset into for hyperparameter selection
	outer_cv: the number of outer folds that the cross validation algorithm is going to divide the dataset into for algorithm selection
	scoring: the evaluation metric we use the monitor model performance

	Returns
	-------
	estimator_dict: a dictionary with algorithm names as keys and sklearn estimator objects as values
	results: a list containing the cross validation results
	'''
	estimator_dict = {}
	for name, method, search_space in zip(estimator_names, estimators, search_spaces):
		estimator = BayesSearchCV(method, search_spaces=search_space, 
			n_iter=50, scoring=scoring,
			n_jobs=-1, random_state=10, 
			cv=inner_cv, refit=True, n_points=n_points,  
			pre_dispatch='2 * n_jobs',
			verbose=verbose)
		estimator_dict[name] = estimator

		results = []
		result_estimator_list = []
		cv = StratifiedKFold(n_splits=outer_cv, random_state=20)
	for estimator_name, estimator in estimator_dict.items():
		outer_score = cross_validate(estimator, X, y,
			cv=cv, scoring=scoring, verbose=0,
			return_estimator=True)

		result_estimator_list.append(outer_score['estimator'])
		results.append([outer_score['test_score'].mean(), outer_score['test_score'].std()])
		print("hyperparameter tuning for", estimator_name, "done!")
		print("mean test score:", outer_score['test_score'].mean(), "std:", outer_score['test_score'].std())

	return estimator_dict, results 

def get_ml_pipeline(X, ml_pipelines_list):
	'''Function to do different preprocessing methods depending on the classification algorithm and specifies different search spaces for different classification algorithms

	Arguments
	---------
	X: the features dataframe
	ml_pipelines_list: specify a list of classification algorithms to use

	Returns
	-------
	estimator_dict.keys(): the names assigned to classification algorithms
	estimator_list: a list of sklearn estimator objects
	search_space_list: a list of search spaces
	'''
	lr_pipeline = Pipeline([
		('pipeline_one_hot_encode', dcp.get_preprocess_pipeline(X)), 
		('lr', LogisticRegression(random_state=0, class_weight='balanced'))
		])
	svc_lin_pipeline = Pipeline([
		('pipeline_one_hot_encode', dcp.get_preprocess_pipeline(X)),
		('svc_lin', LinearSVC(random_state=0, class_weight='balanced', dual=False))
		])
	svc_pipeline = Pipeline([
		('pipeline_one_hot_encode', dcp.get_preprocess_pipeline(X)),
		('svc', SVC(random_state=0, class_weight='balanced', gamma='scale'))
		])
	svc_pca_pipeline = Pipeline([
		('pipeline_one_hot_encode', dcp.get_preprocess_pipeline(X)),
		('pca', PCA(n_components=42)), 
		('svc', SVC(random_state=0, class_weight='balanced', gamma='scale'))
		])
	rfc_pipeline = Pipeline([
		('pipeline_label_encode', dcp.get_preprocess_pipeline(X, is_one_hot_encode=False)),
		('rfc', RandomForestClassifier(random_state=0, class_weight='balanced'))
		])
	xgb_pipeline = Pipeline([
		('pipeline_label_ecode', dcp.get_preprocess_pipeline(X, is_one_hot_encode=False)),
		('xgb', XGBClassifier(random_state=0, class_weight='balanced'))
		])
	gbm_pipeline = Pipeline([
		('pipeline_label_ecode', dcp.get_preprocess_pipeline(X, is_one_hot_encode=False)),
		('gbm', LGBMClassifier(random_state=0, is_unbalance =True))
		])
	mlp_pipeline = Pipeline([
		('pipeline_one_hot_encode', dcp.get_preprocess_pipeline(X)),
		('mlp', MLPClassifier(random_state=0))])

	logistic_search = {'lr__C': (0.01, 1000),
	'lr__penalty': ['l1', 'l2']}
	svc_lin_search = {'svc_lin__C': (0.01, 1000), 
	'svc_lin__penalty': ['l1', 'l2']}
	svc_search = {'svc__C': (0.01, 1000), 
	'svc__kernel': ['poly', 'rbf']}
	rfc_search = {'rfc__n_estimators': (1, 100),
	'rfc__max_depth': (1, 100)
	}  
	xgb_search = {
	'xgb__n_estimators': (50, 500), 
	'xgb__max_depth': (1, 100),
	'xgb__min_child_weight ': (0, 100),
	'xgb__gamma': (0, 100)
	}
	gbm_search = {
	'gbm__n_estimators': (50, 500),
	'gbm__num_leaves': (10, 100),
	'gbm__max_depth': (1, 100),
	'gbm__reg_alpha': (0.001, 200),
	'gbm__reg_lambda': (0.001, 200)
	}
	mlp_search = {
	'mlp__hidden_layer_sizes': [(5, 5), (10, 20), (50, 100), (10, 20, 50)],
	'mlp__alpha': (0.0001, 100),
	'mlp__learning_rate': ['constant', 'adaptive'],
	'mlp__beta_1': (0, 0.99),
	'mlp__beta_2': (0, 0.999),
	'mlp__max_iter': (100, 500)
	}

	estimator_dict = {"svc pca": (svc_pca_pipeline, svc_search), 
	"logistic regression": (lr_pipeline, logistic_search), 
	"linear svc": (svc_lin_pipeline, svc_lin_search), 
	"svc": (svc_pipeline, svc_search), 
	"rfc": (rfc_pipeline, rfc_search),
	"xgboost": (xgb_pipeline, xgb_search), 
	"light gbm": (gbm_pipeline, gbm_search),
	"neural network": (mlp_pipeline, mlp_search)
	}
	estimator_list = [estimator[1][0] for estimator in estimator_dict.items() if estimator[0] in ml_pipelines_list] 
	search_space_list = [estimator[1][1] for estimator in estimator_dict.items() if estimator[0] in ml_pipelines_list]
	return estimator_dict.keys(), estimator_list, search_space_list 

