'''
Created on 2015.10.19

@author: FZY
'''
import xgboost as xgb
import pandas as pd
from trainModel import loadFeatNames
from sklearn.datasets import load_svmlight_file
if __name__ == "__main__":
    param_space_reg_xgb_tree = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class':20,
    'eta': 0.45,
    'gamma': 1.4,
    'min_child_weight': 2.0,
    'max_depth': 6.0,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'num_round': 150,
    'nthread': 4,
    'silent': 1,
    'seed': "20121",
    "max_evals": 1
}
    feat_folder = "../data/feat/raw/feat_all"
    X_train,Y_train = load_svmlight_file("%s/train.svm.txt"%(feat_folder))
    print Y_train
    """
    train_data = xgb.DMatrix("%s/train.svm.txt"%(feat_folder))
    valid_data = xgb.DMatrix("%s/valid.svm.txt"%(feat_folder))
    watchlist = [(train_data,'train'),(valid_data,'valid')]
    bst = xgb.train(param_space_reg_xgb_tree,train_data,int(param_space_reg_xgb_tree['num_round']),watchlist)
    pred = bst.predict(valid_data)
    test = pd.read_csv("../data/feat/raw/feat_all/test.csv")
    feat_names = loadFeatNames("%s/feat.name"%(feat_folder))
    feat_names.remove('cid')
    """
    
    