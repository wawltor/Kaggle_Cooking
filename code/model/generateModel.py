d = {0: 'brazilian', 1: 'british', 2: 'cajun_creole', 3: 'chinese', 4: 'filipino', 5: 'french', 6: 'greek', 7: 'indian', 8: 'irish', 9: 'italian', 10: 'jamaican', 11: 'japanese', 12: 'korean', 13: 'mexican', 14: 'moroccan', 15: 'russian', 16: 'southern_us', 17: 'spanish', 18: 'thai', 19: 'vietnamese'}
{0: ['brazilian'], 1: ['british'], 2: ['cajun_creole'], 3: ['chinese'], 4: ['filipino'], 5: ['french'], 6: ['greek'], 7: ['indian'], 8: ['irish'], 9: ['italian'], 10: ['jamaican'], 11: ['japanese'], 12: ['korean'], 13: ['mexican'], 14: ['moroccan'], 15: ['russian'], 16: ['southern_us'], 17: ['spanish'], 18: ['thai'], 19: ['vietnamese']}
import xgboost as xgb
import pandas as pd
import matplotlib
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
param = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class':20,
    'eta': 0.1,
    'gamma': 0.5,
    'min_child_weight': 1,
    'max_depth': 16,
    'subsample': 1.0,
    'colsample_bytree': 0.2,
    'num_round': 440,
    'nthread': 4,
    'silent': 1,
    'seed': 2015,
    "max_evals": 200,
}

def getDict(train):
    d = dict()
    for i in range(0,20):
        if not d.has_key(i):
            d[i] = list(train[train['cid']==i].iloc[:1]['cuisine'])
    return d
if __name__ == "__main__":
    #train model 
    #train data  
    """
    train_data = "../data/feat/combine/combine_feat_6/train.raw_and_tfidf.svm.txt"
    test_data = "../data/feat/combine/combine_feat_6/test.raw_and_tfidf.svm.txt"
    train_data = xgb.DMatrix(train_data)
    test_data,test_label = load_svmlight_file(test_data)
    watchlist = [(train_data,'train')]
    bst = xgb.train(param,train_data,int(param['num_round']),watchlist)
    pred_data = xgb.DMatrix(test_data)
    test_data = test_data.toarray()
    pred = bst.predict(pred_data)
    test = pd.read_json("../data/test.json")
    print test['id']
    test['cid'] = pred
    test['cuisine'] = list(test.apply(lambda x : d[x['cid']],axis=1))
    test.to_csv("../data/result_10291.csv")
    
    
    train_data = "../data/feat/combine/combine_feat_6/run1/train.svm.txt"
    test_data = "../data/feat/combine/combine_feat_6/test.raw_and_tfidf.svm.txt"
    train_data = xgb.DMatrix(train_data)
    test_data,test_label = load_svmlight_file(test_data)
    watchlist = [(train_data,'train')]
    bst = xgb.train(param,train_data,int(param['num_round']),watchlist)
    pred_data = xgb.DMatrix(test_data)
    test_data = test_data.toarray()
    pred = bst.predict(pred_data)
    test = pd.read_json("../data/test.json")
    print test['id']
    test['cid'] = pred
    test['cuisine'] = list(test.apply(lambda x : d[x['cid']],axis=1))
    test.to_csv("../data/result_10292.csv")
    
    train_data = "../data/feat/combine/combine_feat_6/run2/train.svm.txt"
    test_data = "../data/feat/combine/combine_feat_6/test.raw_and_tfidf.svm.txt"
    train_data = xgb.DMatrix(train_data)
    test_data,test_label = load_svmlight_file(test_data)
    watchlist = [(train_data,'train')]
    bst = xgb.train(param,train_data,int(param['num_round']),watchlist)
    pred_data = xgb.DMatrix(test_data)
    test_data = test_data.toarray()
    pred = bst.predict(pred_data)
    test = pd.read_json("../data/test.json")
    print test['id']
    test['cid'] = pred
    test['cuisine'] = list(test.apply(lambda x : d[x['cid']],axis=1))
    test.to_csv("../data/result_10293.csv")
    
    
    
    
    """
    """
    train_data,train_label = load_svmlight_file("../data/feat/combine/combine_feat_6/train.raw_and_tfidf.svm.txt")
    test_data,test_label = load_svmlight_file("../data/feat/combine/combine_feat_6/test.raw_and_tfidf.svm.txt")
    train_data  = train_data.tocsr()
    test_data = test_data.tocsr()
    clf = LogisticRegression(C=8.96)
    clf.fit(train_data,train_label)
    pred = clf.predict(test_data)
    test = pd.read_json("../data/test.json")
    print test['id']
    test['cid'] = pred
    test['cuisine'] = list(test.apply(lambda x : d[x['cid']],axis=1))
    test.to_csv("../data/result.csv")
    """
    train_data,train_label = load_svmlight_file("../data/feat/combine/combine_feat_1/run1/train.svm.txt")
    test_data,test_label = load_svmlight_file("../data/feat/combine/combine_feat_1/run1/test.svm.txt")
    train_data  = train_data.tocsr()
    test_data = test_data.tocsr()
    clf = LogisticRegression(C=1.8)
    clf.fit(train_data,train_label)
    pred = clf.predict(test_data)
    pred = pd.Series(pred)
    pred.to_csv("lr_c1_1.csv")


    
    train_data,train_label = load_svmlight_file("../data/feat/combine/combine_feat_1/run2/train.svm.txt")
    test_data,test_label = load_svmlight_file("../data/feat/combine/combine_feat_1/run2/test.svm.txt")
    train_data  = train_data.tocsr()
    test_data = test_data.tocsr()
    clf = LogisticRegression(C=1.8)
    clf.fit(train_data,train_label)
    pred = clf.predict(test_data)
    pred = pd.Series(pred)
    pred.to_csv("lr_c1_2.csv")


    train_data,train_label = load_svmlight_file("../data/feat/combine/combine_feat_2/run1/train.svm.txt")
    test_data,test_label = load_svmlight_file("../data/feat/combine/combine_feat_2/run1/test.svm.txt")
    train_data  = train_data.tocsr()
    test_data = test_data.tocsr()
    clf = LogisticRegression(C=15.9)
    clf.fit(train_data,train_label)
    pred = clf.predict(test_data)
    pred = pd.Series(pred)
    pred.to_csv("lr_c2_1.csv")



    train_data,train_label = load_svmlight_file("../data/feat/combine/combine_feat_2/run2/train.svm.txt")
    test_data,test_label = load_svmlight_file("../data/feat/combine/combine_feat_2/run2/test.svm.txt")
    train_data  = train_data.tocsr()
    test_data = test_data.tocsr()
    clf = LogisticRegression(C=15.9)
    clf.fit(train_data,train_label)
    pred = clf.predict(test_data)
    pred = pd.Series(pred）
    pred.to_csv("lr_c2_2.csv")
    
    
    
    
    
    
    
