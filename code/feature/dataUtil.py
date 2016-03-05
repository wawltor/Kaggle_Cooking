import pandas as pd
from sklearn import preprocessing
import nltk
from collections import Counter
from nltk.stem import WordNetLemmatizer
import sys
stemmer = WordNetLemmatizer()
english_stemmer = nltk.stem.PorterStemmer()

reload(sys)
sys.setdefaultencoding("utf-8")

def clean_recipe(recipe):
    # To lowercase
    recipe = [ i.lower() for i in recipe ]

    # Remove some special characters
    # Individuals replace have a very good performance
    # http://stackoverflow.com/a/27086669/670873
    def replacing(i):
        i = i.replace('&', '').replace('(', '').replace(')','')
        i = i.replace('\'', '').replace('\\', '').replace(',','')
        i = i.replace('.', '').replace('%', '').replace('/','')
        i = i.replace('"', '')      
        return i
    
    recipe = [ replacing(i) for i in recipe ]
    recipe = [ i for i in recipe if not i.isdigit() ]
    recipe = [ stemmer.lemmatize(i) for i in recipe ]
    
    return recipe
def stem_words(line,stemer):
    words = []
    for word in line:
        words.append(stemer.stem(word))
    return words
    
def splitData(line):
    line = [word.lower() for word  in line]
    def replacing(i):
        i = i.replace('&', '').replace('(', '').replace(')','')
        i = i.replace('\'', '').replace('\\', '').replace(',','')
        i = i.replace('.', '').replace('%', '').replace('/','')
        i = i.replace('"', '')
        return i

    line = [replacing(word) for word in line]
    line = [word for word in line if not word.isdigit()]
    line = stem_words(line, english_stemmer)
    return line
def countIngreients():
    train = pd.read_json("../data/train.process")
    bags_of_words = [ Counter(recipe) for recipe in train.ingred ]
    print 'generate the sum of bags'
    sumbags = sum(bags_of_words, Counter())
    print sumbags
    fig = pd.DataFrame(sumbags,index=[0]).transpose()
    fig.to_csv("../data/ingred.count.csv")
    fig = pd.read_csv("../data/ingred.count.csv")
    fig.columns = ['id','ingredient','count']
    fig  = fig.sort(ascending=False, inplace=False,columns=['count'])
    fig.to_csv("../data/ingred.count.csv")
def countIngreientsByCuisine():
    train = pd.read_json("../data/train.process")
    for cuisine in train.cuisine.unique():
        bags_of_words=[Counter(recipe) for recipe in train[train['cuisine']==cuisine].ingred]
        sumbags = sum(bags_of_words,Counter())
        fig = pd.DataFrame(sumbags,index=[0]).transpose()
        fig.to_csv("../data/ingred/ingred_%s.csv"%(cuisine))
        fig = pd.read_csv("../data/ingred/ingred_%s.csv"%(cuisine))
        fig.columns=['ingredients','count']
        fig  = fig.sort(ascending=False, inplace=False,columns=['count'])
        print fig.shape[0]
        fig.to_csv("../data/ingred/ingred_%s.csv"%(cuisine))

     
if __name__ == "__main__":
    #1.read data from our json
    
    train = pd.read_json("../data/train.json")
    test = pd.read_json("../data/test.json")
    #preprocessing the data
    encoders = dict()
    encoders['cuisine'] = preprocessing.LabelEncoder()
    train['cid'] = encoders['cuisine'].fit_transform(train['cuisine']) 
    train['ingred'] =  list(train.apply(lambda x : clean_recipe(x['ingredients']),axis=1))
    test['ingred'] = list(test.apply(lambda x : clean_recipe(x['ingredients']),axis=1)) 
    #count the ingreients,aim to get most ingredients.top 10 ingredients each cuisine
    train.to_csv("../data/train.process.csv", force_ascii=True)
    test.to_csv("../data/test.process.csv",force_ascii=True)
  
  
    

    
    
    


    
   
    
    

    
    
    
    
    
    
    



