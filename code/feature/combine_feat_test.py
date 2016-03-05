import pandas as pd
import cPickle as cp
from scipy.sparse import hstack
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.base import BaseEstimator
from decimal import Decimal
from sklearn.preprocessing import Imputer
feat_names =['basic_data','count','tfidf','tfidf_svd','bow','bow_svd']
feat_folders = ['../data/feat/raw/feat_0.09','../data/feat/count','../data/feat/tfidf','../data/feat/tfidf','../data/feat/tfidf','../data/feat/tfidf']
feat_names_1 =['basic_data','count','tfidf','tfidf_svd']
feat_folders_1 = ['../data/feat/raw/feat_0.09','../data/feat/count','../data/feat/tfidf','../data/feat/tfidf']
feat_names_2 =['count','tfidf','tfidf_svd']
feat_folders_2 = ['../data/feat/count','../data/feat/tfidf','../data/feat/tfidf']
feat_names_3 =['count','basic_data','tfidf_svd']
feat_folders_3 = ['../data/feat/count','../data/feat/raw/feat_0.09','../data/feat/tfidf']
feat_names_4 =['count','basic_data','tfidf']
feat_folders_4 = ['../data/feat/count','../data/feat/raw/feat_0.09','../data/feat/tfidf']
feat_names_5 =['count','basic_data','tfidf']
feat_folders_5 = ['../data/feat/count','../data/feat/raw/feat_0.05','../data/feat/tfidf']

feat_names_6 = ["count",'basic_data','tfidf_svd']
feat_folders_6 = ["../data/feat/count","../data/feat/raw/feat_0.05","../data/feat/tfidf"]
feat_names_7 = ["count",'basic_data','tfidf','tfidf_svd']
feat_folders_7 = ["../data/feat/count","../data/feat/raw/feat_0.05","../data/feat/tfidf","../data/feat/tfidf"]
combine_svd = True
def identity(x):
    return x
def cutDot(x):
    return Decimal.from_float(x).quantize(Decimal('0.0000000'))
    
    
class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=cutDot):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return self.transformer(X)
def loadFeatNames(file_path):  
    f = open(file_path,"rb")
    for line in f :
        words = line.split(",")
        words = list(words)
    return words
def combine_feat(feat_names,feat_folders,combine_svd,save_name,save_path):
    print 'combine feature start'
    #now we generate the model 
    ind = 1
    for feat_name,path in zip(feat_names,feat_folders):
        print 'in'
        if feat_name == "basic_data" :
            train = pd.read_csv("%s/train.csv"%(path))
            test = pd.read_csv("%s/test.csv"%(path))
            feats = loadFeatNames("%s/feat.name"%(path))
            feats.remove('cid')
            feats.remove('id')
            x_train = train.loc[:,feats]
            x_test = test.loc[:,feats]
            x_train.fillna(0)
            x_test.fillna(0)
            
        elif feat_name == "count":
            print 'in count'
            train_count = pd.read_csv("../data/feat/count/train_count.csv")
            test_count = pd.read_csv("../data/feat/count/test_count.csv")
            x_train = train_count.loc[:,'ingred_count']
            x_test = test_count.loc[:,'ingred_count']
            
        elif feat_name == "tfidf":
            with open("%s/train.tfidf.feat.pkl"%(path),"rb") as f:
                x_train = cp.load(f)
            with open("%s/test.tfidf.feat.pkl"%(path),"rb") as f:
                x_test = cp.load(f)
        elif feat_name == "tfidf_svd":
            with open("%s/train.tfidf_svd.feat.pkl"%(path),"rb") as f:
                x_train = cp.load(f)
            with open("%s/test.tfidf_svd.feat.pkl"%(path),"rb") as f:
                x_test = cp.load(f)
        elif feat_name == "bow":
            with open("%s/train.bow.feat.pkl"%(path),"rb") as f:
                x_train = cp.load(f)
            with open("%s/test.bow.feat.pkl"%(path),"rb") as f:
                x_test = cp.load(f)
        elif feat_name == "bow_svd":
            with open("%s/train.bow_svd.feat.pkl"%(path),"rb") as f:
                x_train = cp.load(f)
            with open("%s/test.bow_svd.feat.pkl"%(path),"rb") as f:
                x_test = cp.load(f)
        #now we need to deal some data only have one dinmonson
            
        if len(x_train.shape) == 1:    
            x_train = np.array(x_train)
            x_train = x_train.reshape(len(x_train),1)
            print x_train.shape
        if len(x_test.shape) == 1:
            x_test = np.array(x_test)
            x_test = x_test.reshape(len(x_test),1)
        #now we must deal with the dimension differ between the train and test
        dim_differ = abs(x_train.shape[1]-x_test.shape[1])
        
        if x_train.shape[1]>x_test.shape[1]:
            x_test = hstack([x_test,np.zeros(x_test.shape[0],dim_differ)]).tocsr()
        elif x_train.shape[1]<x_test.shape[1]:
            x_train = hstack([x_train,np.zeros(x_train.shape[0],dim_differ)]).tocsr()
        
        if ind == 1:
            X_train,X_test = x_train,x_test
        else:
            try:
                X_train,X_test = hstack([X_train,x_train]),hstack([X_test,x_test])
            except:
                X_train, X_test = np.concatenate((X_train, x_train), 1),np.concatenate((X_test, x_test), 1)
        ind = ind + 1
        print "combine the feature of %s success"%(feat_name)
    print 'save the data'
    #print X_train
    #X_train = Imputer().fit_transform(X_train)
    #X_test = Imputer().fit_transform(X_test)
    dump_svmlight_file(X_train, Y_train, "../data/feat/combine/%s/train.%s.svm.txt"%(save_path,save_name))
    dump_svmlight_file(X_test, Y_test, "../data/feat/combine/%s/test.%s.svm.txt"%(save_path,save_name))
    print 'All Done.'
#def combine_feat1(train,test,train_count,test_count,train_tfidf,test_tfidf,train_svd_tfidf,test_svd_tfidf,train_bow,test_bow,train_svd_bow,test_svd_bow): 
if __name__ == "__main__":
    print 'read data'
    path = "../data"
    train = pd.read_csv("%s/train.process.csv"%(path))
    test = pd.read_csv("%s/test.process.csv"%(path))
    Y_train = train['cid']
    test['cid'] = 20
    Y_test = test['cid']
    combine_feat(feat_names, feat_folders, combine_svd, save_name="raw_data_for_0.09_and_bow_and_svd",save_path="combine_feat_1")
    #combine_feat(feat_names_1, feat_folders_1, combine_svd, save_name="raw_data_for_0.09_and_tfidf_and_svd",save_path="combine_feat_2")
    combine_feat(feat_names_2, feat_folders_2, combine_svd, save_name="tfidf_and_svd",save_path="combine_feat_3")
    combine_feat(feat_names_3,feat_folders_3, combine_svd, save_name="raw_and_svd",save_path="combine_feat_4")
    combine_feat(feat_names_4,feat_folders_4, combine_svd, save_name="raw_and_tfidf",save_path="combine_feat_5")
    combine_feat(feat_names_5,feat_folders_5, combine_svd, save_name="raw_and_tfidf",save_path="combine_feat_6")
    combine_feat(feat_names_6,feat_folders_6, combine_svd, save_name="raw5_and_svd",save_path="combine_feat_7")
    combine_feat(feat_names_7,feat_folders_7, combine_svd, save_name="raw5_tfidf_and_svd",save_path="combine_feat_8")
