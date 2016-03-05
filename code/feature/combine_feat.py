'''
Created on 2015.10.16

@author: FZY
'''
import pandas as pd
from sklearn.base import BaseEstimator
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
def identity(x):
    return x
class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return self.transformer(X)
def combine_feat(df1,df2,feature,opPath):
    for feat in feature:
        df1[feat] = df2[feat]
        df1.to_csv("%s/train.csv"%(opPath))
        #add the feature to 
        f = open("%s/feat.name"%(opPath),"rb")
        for line in f:
            line = line +feat+","
        f.close()
        f = open("%s/feat.name"%(opPath),"wb")
        f.write(line)
        f.close()
        print 'Done it'
        
    
if __name__ == "__main__":
    #
    raw_train = pd.read_csv("../data/feat/raw/feat_50/train.csv")
    print 'combine the ingredients to train,test'
    count_feat = pd.read_csv("../data/feat/count/train_count.csv")
    opPath = "../data/feat/raw/feat_50"
    combine_feat(raw_train, count_feat, ['ingred_count'], opPath)
    print 'combine the tfidf to file'
    tfidf_feat = pd.read_csv("../data/feat/tfidf/train_tfidf.csv")
    opPath = "../data/feat/raw/feat_50"
    feature = ['bow_feat','tfidf_feat']
    combine_feat(raw_train, tfidf_feat, feature, opPath)
    raw_train.to_csv("../data/feat/raw/feat_50/train.csv")
    print 'All Done.'
    
    

    
    
    