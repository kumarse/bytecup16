
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import sklearn as sk



# In[2]:

BASE  = './bytecup2016data'
IINFO = BASE + '/invited_info_train.txt'
QINFO = BASE + '/question_info.txt'
UINFO = BASE + '/user_info.txt'
VAL   = BASE + '/validate_nolabel.txt'

invdata = pd.read_csv(IINFO, delim_whitespace=True, header=None, names=["qid", "uid", "answered"])
qdata   = pd.read_csv(QINFO, delim_whitespace=True, header=None, names=["qid", "qtag", "wseq", "cseq", "nvotes", "nans", "ntqans"])
udata   = pd.read_csv(UINFO, delim_whitespace=True, header=None, names=["uid", "exptag", "wseq", "cseq"])
valdata = pd.read_csv(VAL)


# In[3]:

from sklearn.feature_extraction.text import CountVectorizer


# In[4]:

# Process the qdata

def tokenize(text):
    return text.split("/")

# Convert the character sequence column into a bag of words kind of vector
# Refer: http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
cseq_vec = CountVectorizer(tokenizer=tokenize)
cseq_matrix = cseq_vec.fit_transform(qdata.cseq).toarray()

# Do 1-of-K encoding for tags
qtags = qdata["qtag"].apply(str)
qtag_vec = CountVectorizer(tokenizer=tokenize)
qtag_matrix = qtag_vec.fit_transform(qtags).toarray()

# Convert the numpy arrays to dataframes
cseq_pd = pd.DataFrame(cseq_matrix)
qtag_pd = pd.DataFrame(qtag_matrix)

# Merge
proc_qdata = pd.concat([qdata.qid, cseq_pd, qtag_pd, qdata.nvotes, qdata.nans, qdata.ntqans], axis = 1)


# In[5]:

# Process the udata

# Convert the character sequence column into a bag of words kind of vector
# Refer: http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
cseq_matrix = cseq_vec.fit_transform(udata.cseq).toarray()
exptag_matrix = qtag_vec.fit_transform(udata.exptag).toarray()

# Convert the numpy arrays to dataframes
cseq_pd = pd.DataFrame(cseq_matrix)
exptag_pd = pd.DataFrame(exptag_matrix)

# Merge
proc_udata = pd.concat([udata.uid, cseq_pd, exptag_pd], axis = 1)


# In[6]:

# Insert a column in valdata to store the predicted label probabilities
valdata.insert(2, "label", value = 0.0)


# In[7]:

udata_sim = proc_udata.drop(["uid"], axis = 1)
user_features = udata_sim.values


# In[8]:

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from scipy import sparse

user_sparse = sparse.csr_matrix(user_features)
similarities = cosine_similarity(user_sparse, dense_output=False)
n_users = user_features.shape[0]


# In[10]:
def process_users(i):
    cosine_similarities = linear_kernel(similarities[i:i+1], similarities).flatten()
    related_users_indices = cosine_similarities.argsort()[:-6:-1]
    return related_users_indices

inputs = range(n_users-1)
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(process_users)(i) for i in inputs)
print results


# In[ ]:

# def prepare_testing_data():
#     # Merge with processed udata to get the testing data for the user
#     user_valdata = valdata.merge(proc_udata, on="uid", how="inner").drop(["wseq"], axis = 1)
#     user_ques_valdata = user_valdata.merge(proc_qdata, on="qid", how="inner").drop(["uid","qid","wseq"], axis = 1)
#     test_data = user_ques_valdata.drop(["label"], axis = 1)
#     return test_data


# # In[ ]:

# test_data = prepare_testing_data()

