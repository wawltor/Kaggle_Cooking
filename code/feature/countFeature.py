"""
this file will extract the count in every receipe
this is very simple
"""

import pandas as pd
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def extraFeature(df):
    df['ingred_count'] = list(df.apply(lambda x : len(set(x['ingred'])),axis=1))

if __name__== "__main__":
    print 'read data from preprocessed data'
    train = pd.read_csv("../data/train.process.csv")
    test = pd.read_csv("../data/test.process.csv")
    print 'extract counting feature for train'
    extraFeature(train)
    print 'extract counting feature for test'
    extraFeature(test)
    print 'now we need to save the features'
    print train.columns
    print train['id']
    clos = ['cid','id','ingred_count']
    train.to_csv("../data/feat/count/train_count.csv")
    test.to_csv("../data/feat/count/test_count.csv")  
    
    
   
