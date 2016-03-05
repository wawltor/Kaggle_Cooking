import pandas as pd 
from ParaConfig import config
from sklearn.datasets import dump_svmlight_file
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
def readAllFeaturesByNumber(n_feature):
    print 'generate the feature from n_feature%0.2f'%(n_feature)
    a=set()
    i=set()
    index = 1
    for c in cuisines:
        fig = pd.read_csv("../data/ingred/ingred_%s.csv"%(c))
        fig = set(fig.loc[:int(int(numsByCuisines.get(c))*n_feature),'ingredients'])
        a = a.union(fig)
        if index == 1:
            i = fig
        else:
            i = i.intersection(fig)
        index = index + 1
    return a-i
#judge the ingredients  whether in the x 
def judge(ingred,x):
    if ingred in x :
        return 1 
    else :
        return 0

def extraFeature(df,ingred):
    return list(df.apply(lambda x : judge(ingred, x['ingred']),axis=1))
    
def dumpFeatNames(ingreds,file_path):
    f = open(file_path,"wb")
    for word in ingreds :
        f.write(word+",")
    f.close()

def loadFeatNames(file_path):  
    f = open(file_path,"rb")
    for line in f :
        words = line.split(",")
        words = set(words)
    print words
          
if __name__ == "__main__":
    n_features = config.n_features
    cuisines = config.cuisines
    numsByCuisines = config.numsBycusiens
    
    #read data from process
    train = pd.read_csv("../data/train.process.csv")
    test = pd.read_csv("../data/test.process.csv")
    print 'generate the raw features '
    
    
    for n_feature in n_features:
        ingreds = readAllFeaturesByNumber(n_feature)
        print "all:%d"%(len(ingreds))
        ind = 0
        for ingred in ingreds:
            train[ingred] = extraFeature(train, ingred)
            test[ingred] =extraFeature(test,ingred)
            ind = ind + 1
            print ind
        
        #at first we need to dump all data to json
        #dump into svm file by sklearn 
        #this is prepare for the xgboost
        ingreds = list(ingreds)
        ingreds.append('cid')
        ingreds.append('id')
        #dump feature names into file
        dumpFeatNames(ingreds, "../data/feat/raw/feat_%0.2f/feat.name"%(n_feature))   
        train.loc[:,ingreds].to_csv("../data/feat/raw/feat_%0.2f/train.csv"%(n_feature))
        ingreds.remove('cid')
        test.loc[:,ingreds].to_csv("../data/feat/raw/feat_%0.2f/test.csv"%(n_feature))
        