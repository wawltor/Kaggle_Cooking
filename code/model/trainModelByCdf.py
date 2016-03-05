'''
Created on 2015.10.12

@author: FZY
''' 
import xgboost as xgb
import sys
import numpy as np
import pandas as pd
from ModelConfig import feat_folders,feat_names,para_spaces,int_feat
from ModelConfig import cv_num,cv_test_size
from sklearn.datasets import load_svmlight_file,dump_svmlight_file
import random
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model.ridge import Ridge
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from ml_metrics import quadratic_weighted_kappa,accuracy_model
from createCVFile import loadCVIndex
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import time 
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
def getScore(pred, cdf, valid=False):
    num = pred.shape[0]
    output = np.asarray([19]*num, dtype=int)
    rank = pred.argsort()
    output[rank[:int(num*cdf[0]-1)]] = 0
    output[rank[int(num*cdf[0]):int(num*cdf[1]-1)]] = 1
    output[rank[int(num*cdf[1]):int(num*cdf[2]-1)]] = 2
    output[rank[int(num*cdf[2]):int(num*cdf[3]-1)]] = 3
    output[rank[int(num*cdf[3]):int(num*cdf[4]-1)]] = 4
    output[rank[int(num*cdf[4]):int(num*cdf[5]-1)]] = 5
    output[rank[int(num*cdf[5]):int(num*cdf[6]-1)]] = 6
    output[rank[int(num*cdf[6]):int(num*cdf[7]-1)]] = 7
    output[rank[int(num*cdf[7]):int(num*cdf[8]-1)]] = 8
    output[rank[int(num*cdf[8]):int(num*cdf[9]-1)]] = 9
    output[rank[int(num*cdf[9]):int(num*cdf[10]-1)]] = 10
    output[rank[int(num*cdf[10]):int(num*cdf[11]-1)]] = 11
    output[rank[int(num*cdf[11]):int(num*cdf[12]-1)]] = 12
    output[rank[int(num*cdf[12]):int(num*cdf[13]-1)]] = 13
    output[rank[int(num*cdf[13]):int(num*cdf[14]-1)]] = 14
    output[rank[int(num*cdf[14]):int(num*cdf[15]-1)]] = 15
    output[rank[int(num*cdf[15]):int(num*cdf[16]-1)]] = 16
    output[rank[int(num*cdf[16]):int(num*cdf[17]-1)]] = 17
    output[rank[int(num*cdf[17]):int(num*cdf[18]-1)]] = 18
    return output
def dumpModelMessage(best_params,best_acc_mean,best_acc_std,logPath,f_name,start_time,end_time):
    f = open("%s/%s_bastParamodel_log.txt"%(logPath,f_name),"wb") 
    f.write('Mean:%.6f \nStd:%.6f \n'%(best_acc_mean,best_acc_std))
    for(key,value) in best_params.items():
        f.write("%s:%s\n"%(key,str(value))) 
    f.write("startTime:%s\n"%(start_time))
    f.write("endTime:%s\n"%(end_time))
    f.close()
     

def loadFeatNames(file_path):  
    f = open(file_path,"rb")
    
    for line in f :
        words = line.split(",")
        words = list(words)
    return words


def creatCDF(train,index):
    Y = train['cid']
    #count num of each class 
    hist = np.bincount(Y)
    print hist
    cdf = np.cumsum(hist)/float(sum(hist))
    return cdf
        

def trainModel(param,feat_folder,feat_name):
    #read data from folder
    print 'now we read data from folder:%s'%(feat_folder)
   
    #start cv
    print 'now we need to generate cross_validation'
    accuracy_cv = []
  
    for i in range(0,2):
        print 'this is the run:%d cross-validation'%(i+1)
        testIndex = loadCVIndex("%s/test.run%d.txt"%("../data/feat/combine",(i+1)))
        #if we use xgboost to train model ,we need to use svmlib format
        if param['task'] in ['regression']:
            #with xgb we will dump the file with CV,and we will read data 
            train_data = xgb.DMatrix("%s/run%d/train.svm.txt"%(feat_folder,(i+1)))
            valid_data = xgb.DMatrix("%s/run%d/test.svm.txt"%(feat_folder,(i+1)))
            watchlist = [(train_data,'train'),(valid_data,'valid')]
            bst = xgb.train(param,train_data,int(param['num_round']),watchlist)
            pred = bst.predict(valid_data)
        
        elif param['task'] in ['clf_skl_lr']:
            train_data,train_label = load_svmlight_file("%s/run%d/train.svm.txt"%(feat_folder,(i+1)))
            test_data,test_label = load_svmlight_file("%s/run%d/test.svm.txt"%(feat_folder,(i+1)))
            train_data  = train_data.tocsr()
            test_data = test_data.tocsr()
            clf = LogisticRegression()
            clf.fit(train_data,train_label)
            pred = clf.predict(test_data)
        
        elif param['task'] == "reg_skl_rf":
                    ## regression with sklearn random forest regressor
                    train_data,train_label = load_svmlight_file("%s/run%d/train.svm.txt"%(feat_folder,(i+1)))
                    test_data,test_label = load_svmlight_file("%s/run%d/test.svm.txt"%(feat_folder,(i+1)))
                    rf = RandomForestRegressor(n_estimators=param['n_estimators'],
                                               max_features=param['max_features'],
                                               n_jobs=param['n_jobs'],
                                               random_state=param['random_state'])
                    rf.fit(train_data, test_label)
                    pred = rf.predict(test_data)
        
        elif param['task'] == "reg_skl_etr":
                    ## regression with sklearn extra trees regressor
                    train_data,train_label = load_svmlight_file("%s/run%d/train.svm.txt"%(feat_folder,(i+1)))
                    test_data,test_label = load_svmlight_file("%s/run%d/test.svm.txt"%(feat_folder,(i+1)))
                    etr = ExtraTreesRegressor(n_estimators=param['n_estimators'],
                                              max_features=param['max_features'],
                                              n_jobs=param['n_jobs'],
                                              random_state=param['random_state'])
                    etr.fit(train_data,test_label)
                    pred = etr.predict(test_data)
                    
        elif param['task'] in ['reg_skl_gbm'] :
            train_data,train_label = load_svmlight_file("%s/run%d/train.svm.txt"%(feat_folder,(i+1)))
            test_data,test_label = load_svmlight_file("%s/run%d/test.svm.txt"%(feat_folder,(i+1)))
            gbm = GradientBoostingClassifier(n_estimators=int(param['n_estimators']),
                                             learning_rate=param['learning_rate'],
                                             max_features=param['max_features'],
                                             max_depth=param['max_depth'],
                                             subsample=param['subsample'],
                                             random_state=param['random_state'])
            feat_names.remove('cid')
            gbm.fit(train_data,train_label)
            pred = gbm.predict(test_data) 
        
        elif param['task'] in ['reg_skl_ridge']:
            train_data,train_label = load_svmlight_file("%s/run%d/train.svm.txt"%(feat_folder,(i+1)))
            test_data,test_label = load_svmlight_file("%s/run%d/test.svm.txt"%(feat_folder,(i+1)))
            train_data  = train_data.tocsr()
            test_data = test_data.tocsr()
            ridge = Ridge(alpha=param["alpha"], normalize=True)
            ridge.fit(train_data,train_label)
            
            predraw = ridge.predict(test_data)
            print predraw
            predrank = predraw.argsort().argsort()
            trainIndex = loadCVIndex("%s/train.run%d.txt"%("../data/feat/combine",(i+1)))
            cdf = creatCDF(train, trainIndex)
            pred = getScore(predrank,cdf)
            print pred
            
        """
        elif param['task'] in ['regression']:
            
            
        
        elif param['task'] in ['reg_skl_gbm'] :
            gbm = GradientBoostingClassifier(n_estimators=int(param['n_estimators']),
                                             learning_rate=param['learning_rate'],
                                             max_features=param['max_features'],
                                             max_depth=param['max_depth'],
                                             subsample=param['subsample'],
                                             random_state=param['random_state'])
            feat_names.remove('cid')
            gbm.fit(train_data[feat_names],train_data['cid'])
            pred = gbm.predict(valid_data[feat_names])
        elif param['task'] in ['reg_skl_ridge']:
            feat_names.remove('cid')
            ridge = Ridge(alpha=param["alpha"], normalize=True)
            ridge.fit(train_data[feat_names],train_data['cid'])
            pred = ridge.predict(valid_data[feat_names])
        """
        #now we use the the accuracy to limit our model
        acc = accuracy_model(pred,train.iloc[testIndex]['cid'])
        print "the model accurary:%s"%(acc)
        accuracy_cv.append(acc)

    #here we will count the 
    accuracy_cv_mean = np.mean(accuracy_cv)
    accuracy_cv_std = np.std(accuracy_cv)
    print 'the accuracy for %.6f'%(accuracy_cv_mean)
    return {'loss':-accuracy_cv_mean,'attachments':{'std':accuracy_cv_std},'status': STATUS_OK}
        
if __name__ == "__main__":
    print 'now we start train our model'
    print 'we use two tools to train our model:xgboost and sklearn'
   
    #feat_names1 = "raw_xgb_tree_feat_20"
    #feat_folders1 = "../data/feat/raw/feat_20"
    #we create the same cv for model
    #so we can judge the model at the same condition
    train = pd.read_csv("../data/train.process.csv")
    for feat_name,feat_fold in zip(feat_names,feat_folders):
        #at first we need to read to for our model 
        #this is for reduce time to read data
        print 'read data for trainning'
        print 'generate model in condition in %s'%(feat_name)
        print "Search for the best models"
        print "fea_name %s"%(feat_name)
        #for reduce the time for read data
        #the train.shape[0]=39774
        ISOTIMEFORMAT='%Y-%m-%d %X'
        start_time = time.strftime( ISOTIMEFORMAT, time.localtime() )
        param_space = para_spaces[feat_name]
        trials = Trials()
        objective = lambda p : trainModel(p, feat_fold, feat_name)
        best_params = fmin(objective,param_space,algo=tpe.suggest,
                          trials=trials, max_evals=param_space["max_evals"])
        print type(best_params)
        print best_params
        for f in int_feat:
            if best_params.has_key(f):
                best_params[f] = int(best_params[f])
        trial_acc = -np.asanyarray(trials.losses(), dtype=float )
        best_acc_mean = max(trial_acc)
        ind = np.where(trial_acc==best_acc_mean)[0][0]
        best_acc_std = trials.trial_attachments(trials.trials[ind])['std']
        end_time = time.strftime( ISOTIMEFORMAT, time.localtime() )
        dumpModelMessage(best_params, best_acc_mean, best_acc_std, feat_fold,feat_name,start_time,end_time)
        print ("Best stats")
        print ('Mean:%.6f \nStd:%.6f \n'%(best_acc_mean,best_acc_std))
        