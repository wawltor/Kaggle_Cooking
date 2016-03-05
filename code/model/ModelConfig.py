import numpy as np
from hyperopt import hp


############
## Config ##
############
cv_num = [0,1,2]
cv_test_size = 0.3
debug = True

## xgboost
xgb_random_seed = 2015
xgb_nthread = 8
xgb_dmatrix_silent = True

## sklearn
skl_random_seed = 2015
skl_n_jobs = 2

if debug:
    xgb_nthread = 4
    skl_n_jobs = 1
    xgb_min_num_round = 5
    xgb_max_num_round = 20
    xgb_num_round_step = 5
    skl_min_n_estimators = 5
    skl_max_n_estimators = 10
    skl_n_estimators_step = 5
    libfm_min_iter = 5
    libfm_max_iter = 10
    iter_step = 5
    hyperopt_param = {}
    hyperopt_param["xgb_max_evals"] = 1
    hyperopt_param["rf_max_evals"] = 1
    hyperopt_param["etr_max_evals"] = 1
    hyperopt_param["gbm_max_evals"] = 1
    hyperopt_param["lr_max_evals"] = 1
    hyperopt_param["ridge_max_evals"] = 1
    hyperopt_param["lasso_max_evals"] = 1
    hyperopt_param['svr_max_evals'] = 1
    hyperopt_param['dnn_max_evals'] = 1
    hyperopt_param['libfm_max_evals'] = 1
    hyperopt_param['rgf_max_evals'] = 1
else:
    xgb_min_num_round = 100 
    xgb_max_num_round = 400
    xgb_num_round_step = 15
    skl_min_n_estimators = 100
    skl_max_n_estimators = 500
    skl_n_estimators_step = 20
    libfm_min_iter = 10
    libfm_max_iter = 500
    iter_step = 10
    hyperopt_param = {}
    hyperopt_param["xgb_max_evals"] = 200
    hyperopt_param["rf_max_evals"] = 200
    hyperopt_param["etr_max_evals"] = 200
    hyperopt_param["gbm_max_evals"] = 200
    hyperopt_param["lr_max_evals"] = 200
    hyperopt_param["ridge_max_evals"] = 200
    hyperopt_param["lasso_max_evals"] = 200
    hyperopt_param['svr_max_evals'] = 200
    hyperopt_param['dnn_max_evals'] = 200
    hyperopt_param['libfm_max_evals'] = 200
    hyperopt_param['rgf_max_evals'] = 200




########################################
## Parameter Space for XGBoost models ##
########################################
## In the early stage of the competition, I mostly focus on
## raw tfidf features and linear booster.

## regression with linear booster
param_space_reg_xgb_linear = {
    'task': 'regression',
    'booster': 'gblinear',
    'objective': 'reg:linear',
    'eta' : hp.quniform('eta', 0.3, 2, 0.2),
    'lambda' : hp.quniform('lambda', 0, 5, 0.05),
    'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
    'num_round' : hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent' : 1,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"],
}
#
param_space_reg_xgb_tree_1 = {
    'task': 'regression',
    'booster': 'gblinear',
    'objective': 'multi:softmax',
    'max_delta_step':hp.qloguniform('max_delta_step',1,10,1),
    'num_class':20,
    'eta': hp.quniform('eta', 0.01, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.2, 4, 0.2),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10,1),
    'max_depth': hp.quniform('max_depth', 4, 8, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.01),
    'num_round': hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent': 1,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"],
}
param_space_reg_xgb_tree = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class':20,
    'max_delta_step':hp.qloguniform('max_delta_step',0,10,1),
    'eta': hp.quniform('eta', 0.3, 1, 0.1),
    'gamma': hp.quniform('gamma', 0.1, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'max_depth': hp.quniform('max_depth', 4, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.2, 1, 0.1),
    'num_round': hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent': 1,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"],
}


##sklearn 
#now we need to use sklearn regression to generate model
## gradient boosting regressor
param_space_reg_skl_gbm = {
    'task': 'reg_skl_gbm',
    'n_estimators': hp.quniform("n_estimators", skl_min_n_estimators, skl_max_n_estimators, skl_n_estimators_step),
    'learning_rate': hp.quniform("learning_rate", 0.01, 0.5, 0.01),
    'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
    'max_depth': hp.quniform('max_depth', 1, 15, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'random_state': skl_random_seed,
    "max_evals": hyperopt_param["gbm_max_evals"],
}
## ridge regression
param_space_reg_skl_ridge = {
    'task': 'reg_skl_ridge',
    'alpha': hp.loguniform("alpha", np.log(0.01), np.log(20)),
    'random_state': skl_random_seed,
    "max_evals": hyperopt_param["ridge_max_evals"],
}

param_space_reg_skl_lr = {
    'task': 'reg_skl_lr',
    'C': hp.quniform("C", 1,20,0.2),
    'random_state': skl_random_seed,
    "max_evals": hyperopt_param["ridge_max_evals"],
}


param_space_reg_skl_rf = {
    'task': 'reg_skl_rf',
    'n_estimators': hp.quniform("n_estimators", skl_min_n_estimators, skl_max_n_estimators, skl_n_estimators_step),
    'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
    'n_jobs': skl_n_jobs,
    'random_state': skl_random_seed,
    "max_evals": hyperopt_param["rf_max_evals"],
}

## extra trees regressor
param_space_reg_skl_etr = {
    'task': 'reg_skl_etr',
    'n_estimators': hp.quniform("n_estimators", skl_min_n_estimators, skl_max_n_estimators, skl_n_estimators_step),
    'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
    'n_jobs': skl_n_jobs,
    'random_state': skl_random_seed,
    "max_evals": hyperopt_param["etr_max_evals"],
}


param_space_reg_keras_dnn = {
    'task': 'reg_keras_dnn',
    'batch_norm': hp.choice("batch_norm", [True, False]),
    "hidden_units": hp.choice("hidden_units", [64, 128, 256, 512]),
    "hidden_layers": hp.choice("hidden_layers", [1, 2, 3, 4]),
    "input_dropout": hp.quniform("input_dropout", 0, 0.9, 0.1),
    "hidden_dropout": hp.quniform("hidden_dropout", 0, 0.9, 0.1),
    "hidden_activation": hp.choice("hidden_activation", ["relu", "prelu"]),
    "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
    "nb_epoch": hp.choice("nb_epoch", [10, 20, 30, 40]),
    "max_evals": hyperopt_param["dnn_max_evals"],
}


#set param for model
#main para:feat_folders,feat_spaces
para_spaces={}
feat_folders=[]
feat_names=[]
"""
#1.this is the param for xgb_linear and the number features is 10
feat_name = "raw_xgb_linear_feat_10"
feat_folder = "../data/feat/raw/feat_10"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_linear

#2.this is the param for xgb_linear and the number features is 15
feat_name = "raw_xgb_linear_feat_15"
feat_folder = "../data/feat/raw/feat_15"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_linear

#3.this is the param for xgb_linear and the number features is 20
feat_name = "raw_xgb_linear_feat_20"
feat_folder = "../data/feat/raw/feat_20"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_linear


#4.this is the param for xgb_linear and the number features is 10
feat_name = "raw_xgb_tree_feat_10"
feat_folder = "../data/feat/raw/feat_10"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_tree

#5.this is the param for xgb_linear and the number features is 15
feat_name = "raw_xgb_tree_feat_15"
feat_folder = "../data/feat/raw/feat_15"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_tree

#6.this is the param for xgb_linear and the number features is 20

feat_name = "raw_xgb_tree_feat_20"
feat_folder = "../data/feat/raw/feat_20"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_tree

feat_name = "raw_xgb_tree_feat_30"
feat_folder = "../data/feat/raw/feat_30"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_tree

feat_name = "raw_xgb_tree_feat_40"
feat_folder = "../data/feat/raw/feat_40"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_tree

feat_name = "raw_xbg_tree_feat_all_count"
feat_folder = "../data/feat/raw/feat_40"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_tree_1
"""
feat_name = "reg_dnn_ridge_feat_combine_6"
feat_folder = "../data/feat/combine/combine_feat_6"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_keras_dnn

"""
#4.this is the param for sklearn and the number features is 10
feat_name = "raw_sklearn_linear_feat_10"
feat_folder = "../data/feat/raw/feat_10"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_tree

#5.this is the param for xgb_linear and the number features is 15
feat_name = "raw_sklearn_linear_feat_15"
feat_folder = "../data/feat/raw/feat_15"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_xgb_tree
"""
#6.this is the param for xgb_linear and the number features is 20
"""
feat_name = "raw_sklearn_linear_feat_30"
feat_folder = "../data/feat/raw/feat_30"
feat_folders.append(feat_folder)
feat_names.append(feat_name)
para_spaces[feat_name]=param_space_reg_skl_gbm
"""



#integer features
int_feat = ["num_round", "n_estimators", "max_depth", "degree",
            "hidden_units", "hidden_layers", "batch_size", "nb_epoch",
            "dim", "iter",
            "max_leaf_forest", "num_iteration_opt", "num_tree_search", "min_pop", "opt_interval"]



