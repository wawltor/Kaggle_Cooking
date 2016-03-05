'''
Created on 2015/10/21

@author: FZY
'''
    
import pandas as pd
from pandas import Series, DataFrame
from sklearn.datasets import load_svmlight_file
import numpy as np
from numpy import dtype
from createCVFile import loadCVIndex,dumpCVIndex
from sklearn.cross_validation import train_test_split
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
if __name__ == "__main__":
    train = pd.read_json("../data/train.process")
    train['idx'] = range(0,train.shape[0])
    print train['idx']
    X_train,X_Test,Y_Train,Y_Test = train_test_split(train[['id','idx']],train['cid'],test_size=0.3,random_state=2008)
    print X_train.columns
    train_index = X_train['idx']
    print train_index.shape
    test_index = X_Test['idx']
    print test_index.shape
    dumpCVIndex("train.run1.txt", train_index)
    dumpCVIndex("test.run1.txt", test_index)    