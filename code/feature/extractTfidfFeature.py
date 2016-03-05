'''
Created on 2015.10.16

@author: FZY
this file will extract the tfidf 
'''
import pandas as pd
from ParaConfig import config
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.stem import WordNetLemmatizer
import sys
import cPickle as cp
import re

#define first type of Vectorizer
""""
tfidf_min_df = config.tfidf_min_df
tfidf_max_df = config.bow_max_df
ngram_range = config.ngram_range
token_pattern = r"(?u)\b\w\w+\b"
norm = "l2"


stemmer = WordNetLemmatizer()
english_stemmer = nltk.stem.PorterStemmer()
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,self).build_analyzer()
        return lambda doc : (english_stemmer.stem(w) for w in analyzer(doc))
        
def getTFV(token_pattern = token_pattern,
           norm = norm,
           max_df = tfidf_max_df,
           min_df = tfidf_min_df,
           ngram_range = (1, 1),
           vocabulary = None,
           stop_words = 'english'):
    tfv = StemmedTfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None, 
                                 strip_accents='unicode', analyzer='char', token_pattern=token_pattern,
                                 ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                 stop_words = stop_words, norm=norm, vocabulary=vocabulary)
    return tfv

token_pattern = r"(?u)\b\w\w+\b"
bow__max_df = 0.75
bow__min_df = 3

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
def getBOW(token_pattern = token_pattern,
           max_df = bow__max_df,
           min_df = bow__min_df,
           ngram_range = (1, 1),
           vocabulary = None,
           stop_words = 'english'):
    bow = StemmedCountVectorizer(min_df=min_df, max_df=max_df, max_features=None, 
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range,
                                 stop_words = stop_words, vocabulary=vocabulary)
    return bow 

def cat_text(list_words):
    line = ""
    ind = 1 
    for word in list_words:
        if ind == 1:
            line = word
        else:
            line = line+" "+word
        ind = ind + 1
    return line
        
     
def formatIngredforVectorizer(df):
    print 'format ingredients'
    df['format_ingred'] = list(df.apply(lambda x: cat_text(list(x['ingred'])),axis=1))   
    

def extraFeatures(train,test,vec_type):
    #genrate the basic bow and tfidf features
    print 'now we generate the basic feature of ingreds'
    if vec_type == 'tfidf':
        vec = getTFV()
    else:
        vec = getBOW()
    
    train["%s_feat"%(vec_type)] = vec.fit_transform(train['ingred'])
    print train["%s_feat"%(vec_type)]
    test["%s_feat"%(vec_type)] = vec.transform(test['ingred'])
    
    #generate the svd tfidf features
    print 'generate the pca features by svd'
    svd = TruncatedSVD(random_state=2015,n_components=200,n_iter=15)
    train["%s_svd_feat"%(vec)] = svd.fit_transform(train["ingred"])
    test["%s_svd_feat"%(vec)] = svd.transform(test["ingred"])
    
    print 'save the data'
    print 'ALl Done.'
    
    #we use the svd to reduce our 
if __name__ == "__main__":
    print "generate the feature of tfidf"
    train = pd.read_csv("../data/1.csv")
    test = pd.read_csv("../data/2.csv")
    print 'format the ingredients'
    
    #formatIngredforVectorizer(train)
    #formatIngredforVectorizer(test)
    #we know there have two types of CountVectorizer
    vec_types = ['tfidf','bow']
    for t in vec_types:
        extraFeatures(train, test, t)
    #feat_list = ['bow_feat','tfidf_feat','bow_svd_feat','tfidf_bow_feat','id']
    feat_list = ["bow_svd_feat","tfidf_svd_feat","id"]
    train.loc[:,feat_list].to_csv("../data/feat/tfidf/train_tfidf_svd.csv")
    test.loc[:,feat_list].to_csv("../data/feat/tfidf/test_tfidf_svd.csv")   
"""
english_stemmer = nltk.stem.SnowballStemmer('english') 
token_pattern = r"(?u)\b\w\w+\b"
tfidf__norm = "l2"
tfidf__max_df = 0.65
tfidf__min_df = 3
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,self).build_analyzer()
        return lambda doc : (english_stemmer.stem(w) for w in analyzer(doc))
        
def getTFV(token_pattern = token_pattern,
           norm = tfidf__norm,
           max_df = tfidf__max_df,
           min_df = tfidf__min_df,
           ngram_range = (1, 1),
           vocabulary = None,
           stop_words = 'english'):
    tfv = StemmedTfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None, 
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                 stop_words = stop_words, norm=norm, vocabulary=vocabulary)
    return tfv

token_pattern = r"(?u)\b\w\w+\b"
bow__max_df = 0.65
bow__min_df = 3

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
def getBOW(token_pattern = token_pattern,
           max_df = bow__max_df,
           min_df = bow__min_df,
           ngram_range = (1, 1),
           vocabulary = None,
           stop_words = 'english'):
    bow = StemmedCountVectorizer(min_df=min_df, max_df=max_df, max_features=None, 
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range,
                                 stop_words = stop_words, vocabulary=vocabulary)
    return bow

def cat_text(list_words):
    line = ""
    ind = 1 
    for word in list_words:
        if ind == 1:
            line = word
        else:
            line = line+" "+word
        ind = ind + 1
    return line
        
     
def formatIngredforVectorizer(df):
    df['ingredients_clean_string'] = [' , '.join(z).strip() for z in df['ingredients']]  
    df['format_ingred'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in df['ingredients']]       

def extraFeatures(train,test,vec_type):
    #genrate the basic bow and tfidf features
    print 'now we generate the basic feature of ingreds'
    if vec_type == 'tfidf':
        vec = getTFV()
    else:
        vec = getBOW()
    print train['format_ingred']
    
    X_train = vec.fit_transform(train['format_ingred'])

    print X_train.shape
    X_test = vec.transform(test['format_ingred'])
    with open("../data/feat/tfidf/train.%s.feat.pkl"%(vec_type),"wb") as f:
        cp.dump(X_train, f, -1)
    with open("../data/feat/tfidf/test.%s.feat.pkl"%(vec_type),"wb") as f:
        cp.dump(X_test, f, -1)   
    
    
    #generate the svd tfidf features
    print 'generate the pca features by svd'
    svd = TruncatedSVD(random_state=2015,n_components=150,n_iter=15)
    X_svd_train = svd.fit_transform(X_train)
    print X_svd_train.shape
    X_svd_test = svd.transform(X_test)
    with open("../data/feat/tfidf/train.%s_svd.feat.pkl"%(vec_type),"wb") as f:
        cp.dump(X_svd_train, f, -1)
    with open("../data/feat/tfidf/test.%s_svd.feat.pkl"%(vec_type),"wb") as f:
        cp.dump(X_svd_test, f, -1)
    print 'ALl Done.'
    

if __name__ == "__main__":
    print "generate the feature of tfidf"
    traindf = pd.read_json("../data/train.json")
    testdf = pd.read_json("../data/test.json") 
    formatIngredforVectorizer(traindf)
    traindf.to_csv("tmp.csv")
    """
    formatIngredforVectorizer(testdf)
    vec_types = ['tfidf','bow']
    for t in vec_types:
        extraFeatures(traindf, testdf, t)
    """











  
    
    