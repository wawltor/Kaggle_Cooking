'''
Created on 2015.10.18

@author: FZY
'''
import pandas as pd
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#convert the cid to cuisine
d = {0: 'brazilian', 1: 'british', 2: 'cajun_creole', 3: 'chinese', 4: 'filipino', 5: 'french', 6: 'greek', 7: 'indian', 8: 'irish', 9: 'italian', 10: 'jamaican', 11: 'japanese', 12: 'korean', 13: 'mexican', 14: 'moroccan', 15: 'russian', 16: 'southern_us', 17: 'spanish', 18: 'thai', 19: 'vietnamese'}
#now we use majorrity to select model 



## now we just use two model to create the majorrity
#the baseline is my best model
def majority(train):
    allPreds = np.bincount(train)

    prmax = np.max(allPreds) >= colslen
    if prmax : 
        return allPreds.argmax()
    else:
        return train['c5']
    

if __name__ == "__main__":
    #now we read data from the the csv file  
    
    data = pd.read_csv("../data/allPreds.csv")
    colslen = (data.shape[1]/2+1)
    data['final'] = list(data.apply(lambda x : majority(x),axis=1))
    data['cuisine'] = list(data.apply(lambda x : d[x['final']],axis=1))
    data.to_csv("../data/result/lr_1xgb_5_6.csv")
    
    
    
    
    
    