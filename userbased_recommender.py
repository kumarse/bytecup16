
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation as cv


# In[2]:

BASE  = './bytecup2016data'
IINFO = BASE + '/invited_info_train.txt'
QINFO = BASE + '/question_info.txt'
UINFO = BASE + '/user_info.txt'
VAL   = BASE + '/validate_nolabel.txt'

invdata = pd.read_csv(IINFO, delim_whitespace=True, header=None, names=["qid", "uid", "answered"])
train_invdata, test_invdata = cv.train_test_split(invdata, test_size=0.25)
qdata   = pd.read_csv(QINFO, delim_whitespace=True, header=None, names=["qid", "qtag", "wseq", "cseq", "nvotes", "nans", "ntqans"])
udata   = pd.read_csv(UINFO, delim_whitespace=True, header=None, names=["uid", "exptag", "wseq", "cseq"])
valdata = pd.read_csv(VAL)

merged_data = qdata.merge(train_invdata,on="qid", how="left").merge(udata, on="uid", how="right") 
ratings_mtx_df = merged_data.pivot_table(values='answered',
                                             index='uid',
                                             columns='qid')


# In[3]:

ratings_mtx_df = ratings_mtx_df.reindex(udata.uid)
ratings_mtx_df = pd.concat([ratings_mtx_df,pd.DataFrame(columns=qdata.qid)])


# In[4]:

data = ratings_mtx_df.fillna(0)


# In[5]:

user_preferences = data.as_matrix()
user_similarity = pairwise_distances(user_preferences, metric='correlation')


# In[6]:

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred


# In[7]:

user_prediction = predict(user_preferences, user_similarity, type='user')


# In[8]:

user_predictions_df = pd.DataFrame(data=user_prediction,index=data.index,columns=data.columns)


# In[14]:

test_invdata=test_invdata.reset_index()
test_invdata["label"] = ""


# In[15]:

n_validation_users = len(test_invdata.index)
for i in range(n_validation_users):
    qid_val = test_invdata.iloc[i]['qid']
    uid_val = test_invdata.iloc[i]['uid']
    test_invdata.iloc[i, test_invdata.columns.get_loc('label')] = user_predictions_df.loc[uid_val,qid_val]


# In[17]:

test_invdata.to_csv('CF-validation-correlation.csv', separator=",")


# # In[18]:

# # Using SVD
# from scipy.sparse.linalg import svds

# #get SVD components from train matrix. Choose k.
# u, s, vt = svds(user_preferences, k = 20)
# s_diag_matrix=np.diag(s)
# pred = np.dot(np.dot(u, s_diag_matrix), vt)


# # In[19]:

# svd_predictions_df = pd.DataFrame(data=pred,index=data.index,columns=data.columns)


# # In[20]:

# test_invdata["label"] = ""


# # In[21]:

# n_validation_users = len(test_invdata.index)
# for i in range(n_validation_users):
#     qid_val = test_invdata.iloc[i]['qid']
#     uid_val = test_invdata.iloc[i]['uid']
#     test_invdata.iloc[i, test_invdata.columns.get_loc('label')] = svd_predictions_df.loc[uid_val,qid_val]


# # In[22]:

# test_invdata.to_csv('svd-testresults.csv', separator=",")


# # In[24]:

# from sklearn.feature_extraction.text import CountVectorizer


# # In[25]:

# # Process the qdata

# def tokenize(text):
#     return text.split("/")

# # Convert the character sequence column into a bag of words kind of vector
# # Refer: http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
# cseq_vec = CountVectorizer(tokenizer=tokenize)
# cseq_matrix = cseq_vec.fit_transform(qdata.cseq).toarray()

# # Do 1-of-K encoding for tags
# qtags = qdata["qtag"].apply(str)
# qtag_vec = CountVectorizer(tokenizer=tokenize)
# qtag_matrix = qtag_vec.fit_transform(qtags).toarray()

# # Convert the numpy arrays to dataframes
# cseq_pd = pd.DataFrame(cseq_matrix)
# qtag_pd = pd.DataFrame(qtag_matrix)

# # Merge
# proc_qdata = pd.concat([qdata.qid, cseq_pd, qtag_pd, qdata.nvotes, qdata.nans, qdata.ntqans], axis = 1)


# # In[26]:

# test_invdata["label"] = 0.0


# # In[27]:

# def prepare_training_data_for_user(uid):
#     # Get entries for the user from invited data
#     user_invdata = train_invdata[train_invdata.uid == uid]
#     # Merge with processed qdata to get the training data for the user
#     user_data = user_invdata.merge(proc_qdata, on="qid", how="inner").drop(["qid", "uid", "wseq"], axis = 1)
#     user_train_labels = user_data.answered
#     user_train_data = user_data.drop(["answered"], axis = 1)
#     return user_train_data, user_train_labels


# # In[28]:

# def get_val_data_for_user(uid):
#     user_valdata = test_invdata[test_invdata.uid == uid]
#     user_valdata = user_valdata.merge(proc_qdata, on="qid", how="inner").drop(["wseq", "label"], axis = 1)
#     return user_valdata


# # In[29]:

# from sklearn import linear_model


# # In[ ]:

# for uid in np.unique(test_invdata.uid):
#     user_unique_labels = np.unique(train_invdata[train_invdata.uid == uid].answered)

#     if len(user_unique_labels) != 1:
#         user_train_data, user_train_labels = prepare_training_data_for_user(uid)
#         if user_train_data.shape[0] > 0:
#             regr = linear_model.LogisticRegression()
#             regr.fit(user_train_data, user_train_labels)
    
#     user_val_data = get_val_data_for_user(uid)
#     user_val_trimmed_data = user_val_data.drop(["qid", "uid","index","answered"], axis = 1)
    
#     if len(user_unique_labels) != 1 and user_train_data.shape[0] > 0:
#         predicted_proba = regr.predict_proba(user_val_trimmed_data)
#     else:
#         if len(user_unique_labels) == 0:
#             user_unique_labels = [0]
#         predicted_proba = np.array([[0.0, 1.0] if user_unique_labels[0] == 1 else [1.0, 0.0] for i in range(user_val_data.shape[0])])
    
#     test_invdata.ix[test_invdata.uid == uid, 'label'] = predicted_proba[:, 1]


# # In[ ]:

# # Write output as CSV
# test_invdata.to_csv("content-based-testresults.csv")

