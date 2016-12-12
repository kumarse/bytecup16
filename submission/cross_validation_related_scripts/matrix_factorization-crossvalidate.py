
# coding: utf-8

# In[21]:

import numpy as np
import pandas as pd
import graphlab
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

BASE  = '../bytecup2016data'
IINFO = BASE + '/invited_info_train.txt'
QINFO = BASE + '/question_info.txt'
VAL   = BASE + '/validate_nolabel.txt'
TEST   = BASE + '/test_nolabel.txt'

invdata = pd.read_csv(IINFO, delim_whitespace=True, header=None, names=["qid", "uid", "label"])
qdata   = pd.read_csv(QINFO, delim_whitespace=True, header=None, names=["qid", "qtag", "wseq", "cseq", "nvotes", "nans", "ntqans"])
valdata = pd.read_csv(TEST)

qdata = qdata.drop(["wseq","cseq","qtag"], axis = 1)


# In[22]:

def algorithm_predict(new_invdata, new_valdata, qdata):
    sf = graphlab.SFrame(new_invdata, format='dataframe')
    training, validation = sf.random_split(0.75)
    params = {'target':'label', 'user_id':'uid', 'item_id':'qid'}
    job = graphlab.model_parameter_search.create((training, validation), graphlab.ranking_factorization_recommender.create, params)
    print job.get_results()
#     folds = graphlab.cross_validation.KFold(sf, 5)
#     params = dict([('target', 'label'), ('num_factors', 8), ('user_id' , 'uid'), ('item_id' , 'qid')])
#     job = graphlab.cross_validation.cross_val_score(folds,
#                                               graphlab.recommender.factorization_recommender.create,
#                                               params)
#     print job.get_results()


# In[23]:

algorithm_predict(invdata, valdata, qdata)


# In[48]:

print "",str(sum([0.4151434821370934, 0.415209384158512, 0.41351925583323684, 0.40903968519099976, 0.4161257849170225]) / 5)
print "",str(sum([0.4164529878223203, 0.41688970440625606, 0.4148440425795822, 0.41031352673596194, 0.4176783775931639]) / 5)
print "",str(sum([0.41635797722212375, 0.4154277450022756, 0.4135618690187132, 0.4083892215575476, 0.4165047271638262]) / 5)


# In[ ]:



