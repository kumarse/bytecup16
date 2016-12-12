BASE = './bytecup2016data'
IINFO = BASE + '/invited_info_train.txt'
QINFO = BASE + '/question_info.txt'
UINFO = BASE + '/user_info.txt'
VAL = BASE + '/validate_nolabel.txt'
TEST = BASE + '/test_nolabel.txt'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')

invdata = pd.read_csv(IINFO, delim_whitespace=True, header=None, names=["qid", "uid", "answered"])
qdata   = pd.read_csv(QINFO, delim_whitespace=True, header=None, names=["qid", "qtag", "wseq", "cseq", "nvotes", "nans", "ntqans"])
udata   = pd.read_csv(UINFO, delim_whitespace=True, header=None, names=["uid", "exptag", "wseq", "cseq"])
valdata = pd.read_csv(VAL)

udata = pd.read_csv(UINFO, delim_whitespace=True, header=None, names=["uid", "exptag", "wseq", "cseq"])


# In[3]:

inv_user = np.unique(invdata.uid)
val_user = np.unique(valdata.uid)

inv_user_set = set(inv_user.tolist())
val_user_set = set(val_user.tolist())

strange_usr =  val_user_set - inv_user_set

valdata_uid = valdata.uid.tolist()


# In[5]:

LOGISTIC_FULL_RESULT = 'logistic_model_full_data.csv'
SVM_RESULT = 'per_user_svm.csv'
logistic_full   = pd.read_csv(LOGISTIC_FULL_RESULT, delimiter=",", skiprows=1, names=["qid", "uid", "label"])
svm   = pd.read_csv(SVM_RESULT, delimiter=",", skiprows=1, names=["qid", "uid", "label"])

for uid in strange_usr:
    logistic_full_entries = logistic_full[logistic_full.uid == uid]    
    svm.ix[svm.uid == uid, 'label'] = logistic_full_entries.label
    
svm.to_csv("logistic_svm_hybrid.csv")


# In[ ]:



