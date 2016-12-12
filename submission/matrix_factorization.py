import numpy as np
import pandas as pd
import graphlab
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import sys

BASE  = './bytecup2016data'
IINFO = BASE + '/invited_info_train.txt'
QINFO = BASE + '/question_info.txt'
VAL   = BASE + '/validate_nolabel.txt'
TEST   = BASE + '/test_nolabel.txt'

invdata = pd.read_csv(IINFO, delim_whitespace=True, header=None, names=["qid", "uid", "label"])
qdata   = pd.read_csv(QINFO, delim_whitespace=True, header=None, names=["qid", "qtag", "wseq", "cseq", "nvotes", "nans", "ntqans"])
valdata = pd.read_csv(VAL)

qdata = qdata.drop(["wseq","cseq","qtag"], axis = 1)
if len(sys.argv) > 1:
    param1 = sys.argv[1]
    if param1 == '1':
        print "Running for Test set"
        valdata = pd.read_csv(TEST)


def algorithm_predict(new_invdata, new_valdata, qdata):
    sf = graphlab.SFrame(new_invdata, format='dataframe')
    #side_item = graphlab.SFrame(qdata, format='dataframe')
    #m = graphlab.recommender.create(sf, target='rating')
    #m = graphlab.item_similarity_recommender.create(sf, user_id='uid', item_id='qid', target='label', similarity_type='pearson')
    #m = graphlab.factorization_recommender.create(sf, target='label', regularization=0.0001, num_factors = 32,
                                                  #max_iterations = 25, linear_regularization = 1e-05, user_id='uid', item_id='qid')
    
    m = graphlab.recommender.ranking_factorization_recommender.create(sf, target='label', regularization=1e-08, num_factors = 32,
                                                  max_iterations = 25, ranking_regularization = 0.1, user_id='uid', item_id='qid',
                                                                     num_sampled_negative_examples = 8)
    #m = graphlab.popularity_recommender.create(sf, target='rating')  
    
    sf = graphlab.SFrame(new_valdata, format='dataframe')
    return m.predict(sf)


valdata['label'] = algorithm_predict(invdata, valdata, qdata)
valdata.to_csv('matrix_factorization-recommender.csv', separator=",")



