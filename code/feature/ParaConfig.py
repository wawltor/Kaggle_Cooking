'''
Created on 2015.10.12

@author: FZY
'''


class ParamConfig:
    def __init__(self):
        self.n_features = [0.05,0.06,0.07,0.08,0.09]
        self.cuisines = ['greek','southern_us','filipino','indian','jamaican','spanish','italian','mexican','chinese','british','thai','vietnamese','cajun_creole','brazilian','french','japanese','irish','korean','moroccan' ,'russian']
        self.cv_num = 3
        self.cv_raito = 0.3
        self.low_num_features = 10
        self.bow_min_df = 3
        self.bow_max_df = 0.75
        self.tfidf_max_df = 0.75
        self.tfidf_min_df = 3
        self.ngram_range=(1,1)
        self.numsBycusiens = {'irish': 999, 'mexican': 2681, 'chinese': 1791, 'filipino': 947, 'vietnamese': 1108, 'moroccan': 974, 'brazilian': 853, 'japanese': 1439, 'british': 1165, 'greek': 1198, 'indian': 1663, 'jamaican': 877, 'french': 2100, 'spanish': 1262, 'russian': 872, 'cajun_creole': 1575, 'thai': 1376, 'southern_us': 2458, 'korean': 898, 'italian': 2928}
config = ParamConfig()

