#for reducing the time to generate the file 
#we use this model to generate the file 
from sklearn.datasets import load_svmlight_file,dump_svmlight_file
import random
import pandas as pd
from ModelConfig import cv_test_size
from numpy.doc import creation
import numpy as np
#we use this function to create the CV index 
def dumpCVIndex(path,list):
    ind = 1
    f = open(path,"wb")
    for i in list:
        if ind == 1:
            f.write(str(i))
        else:
            f.write(",%d"%(i))
        ind = ind + 1
    f.close()

def loadCVIndex(path):
    f = open(path,"rb")
    for l in f :
        line = l 
    indexlist = list(line.split(","))
    integerlist = []
    for ind in indexlist:
        integerlist.append(int(ind))
    return integerlist
    
def create_cv(length,raito):
    data_index = set(range(0,length))
    num = int(length*raito)
    test_index = set(random.sample(data_index,num))
    train_index = data_index - test_index
    return list(train_index),list(test_index)

#we use the this function to create the list of CV index 
def createCV(nums,length,cv_test_size,path):
    trainlist= []
    testlist = []
    for i in range(0,nums):
        print "this is %d run cv"%(i)
        trainindx,testindx = create_cv(length, cv_test_size)
        dumpCVIndex("%s/train.run%d.txt"%(path,i+1),trainindx)
        dumpCVIndex("%s/test.run%d.txt"%(path,i+1),testindx)
        trainlist.append(trainindx)
        testlist.append(testindx)
    return trainlist,testlist


def creatFile(path,f):
    print 'load raw file to split'
    train_list , test_list = createCV(nums=3,length=39774, cv_test_size=cv_test_size,path=path)
    X_train,Y_train = load_svmlight_file("%s/%s"%(path,f))
    Y_train = pd.Series(Y_train)
    print 'create the train_index,test_index'
    
    #now we create the cv 
    ind = 1
    for train_index,test_index in zip(train_list,test_list):
        print 'generate the file for run %d'%(ind)
        print "len:%d"%(len(train_index))
        print "len:%d"%(len(test_index))
        X_train_cv = X_train[train_index,:]
        Y_train_cv = Y_train.iloc[train_index]
        X_test_cv = X_train[test_index,:]
        Y_test_cv = Y_train.iloc[test_index]
        dump_svmlight_file(X_train_cv, Y_train_cv,"%s/run%d/train.svm.txt"%(path,ind))
        dump_svmlight_file(X_test_cv, Y_test_cv,"%s/run%d/test.svm.txt"%(path,ind))
        ind = ind + 1

def creatFile_1(path,f):
    print 'load raw file to split'
    X_train,Y_train = load_svmlight_file("%s/%s"%(path,f))
    Y_train = pd.Series(Y_train)
    print 'create the train_index,test_index'
    
    #now we create the cv 
    ind = 1
    for i in range(0,3):
        print 'generate the file for run %d'%(ind)
        train_index = loadCVIndex("../data/feat/combine/train.run%d.txt"%(i+1))
        test_index = loadCVIndex("../data/feat/combine/test.run%d.txt"%(i+1))
        print "len:%d"%(len(train_index))
        print "len:%d"%(len(test_index))
        X_train_cv = X_train[train_index,:]
        Y_train_cv = Y_train.iloc[train_index]
        X_test_cv = X_train[test_index,:]
        Y_test_cv = Y_train.iloc[test_index]
        dump_svmlight_file(X_train_cv, Y_train_cv,"%s/run%d/train.svm.txt"%(path,ind))
        dump_svmlight_file(X_test_cv, Y_test_cv,"%s/run%d/test.svm.txt"%(path,ind))
        ind = ind + 1
    
if __name__ == "__main__":
    
    
    creatFile_1("../data/feat/combine/combine_feat_6","train.raw_and_tfidf.svm.txt")
    """
    creatFile_1("../data/feat/combine/combine_feat_7","train.raw5_and_svd.svm.txt")
    creatFile_1("../data/feat/combine/combine_feat_8","train.raw5_tfidf_and_svd.svm.txt")
    creatFile("../data/feat/combine/combine_feat_3", "train.tfidf_and_svd.svm.txt")
    creatFile("../data/feat/combine/combine_feat_1","train.raw_data_for_0.09_and_bow_and_svd.svm.txt")
    creatFile("../data/feat/combine/combine_feat_2","train.raw_data_for_0.09_and_tfidf_and_svd.svm.txt")
    creatFile("../data/feat/combine/combine_feat_4","train.raw_and_svd.svm.txt")
    creatFile("../data/feat/combine/combine_feat_5","train.raw_and_tfidf.svm.txt")
    """
    