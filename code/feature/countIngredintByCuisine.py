'''
Created on 2015/10/19

@author: FZY
'''
import pandas as pd
import numpy as np
from ParaConfig import config
import sys
reload(sys)


def countIngredinet():
    nums = dict()
    for cu in cuisines:
        data = pd.read_csv("../data/ingred/ingred_%s.csv"%(cu))
        nums[cu] = data.shape[0]
    print nums
    return nums
 
   

if __name__ == "__main__":
    cuisines = config.cuisines
    countIngredinet()


    
    
