'''
Created on 2015.10.15

@author: FZY
'''
import pandas as pd
import numpy as np
from ParaConfig import config
from extraRawFeature import dumpFeatNames
import sys
from extraRawFeature import extraFeature
reload(sys)
sys.setdefaultencoding("utf-8")
def readAllFeatures():
    print 'generate the feature from n_feature'
    a=set()
    i=set()
    index = 1
    for c in cuisines:
        fig = pd.read_csv("../data/ingred/ingred_%s.csv"%(c))
        #print num of features
        fig = fig[fig['count']>=config.low_num_features]
        fig = set(fig['ingredients'])
        a = a.union(fig)
        if index == 1:
            i = fig
        else:
            i = i.intersection(fig)
        index = index + 1
    
    print len(a-i)
    return a-i

if __name__ == "__main__":
    train = pd.read_csv("../data/train.process.csv")
    test = pd.read_csv("../data/test.process.csv")
    cuisines = config.cuisines
    ingreds = readAllFeatures()
    ingreds = list(ingreds)
    i = 1 
    for ingred in ingreds:
        print i 
        train[ingred] = extraFeature(train,ingred)
        test[ingred] =extraFeature(test,ingred) 
        i = i + 1
    print 'load file'
    ingreds.append('cid')
    ingreds.append('id')
    dumpFeatNames(ingreds)
    train.loc[:,ingreds].to_csv("../data/feat/raw/feat_all/train.csv")
    ingreds.remove('cid')
    test.loc[:,ingreds].to_csv("../data/feat/raw/feat_all/test.csv")
    
    
    
    
    
    