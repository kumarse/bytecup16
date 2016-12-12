import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation as cv


BASE  = './bytecup2016data'
IINFO = BASE + '/invited_info_train.txt'
QINFO = BASE + '/question_info.txt'
UINFO = BASE + '/user_info.txt'
VAL   = BASE + '/validate_nolabel.txt'
TEST = BASE + '/test_nolabel.txt'

invdata = pd.read_csv(IINFO, delim_whitespace=True, header=None, names=["qid", "uid", "answered"])
qdata   = pd.read_csv(QINFO, delim_whitespace=True, header=None, names=["qid", "qtag", "wseq", "cseq", "nvotes", "nans", "ntqans"])
udata   = pd.read_csv(UINFO, delim_whitespace=True, header=None, names=["uid", "exptag", "wseq", "cseq"])
valdata = pd.read_csv(VAL)

if len(sys.argv) > 1:
    param1 = sys.argv[1]
    if param1 == '1':
        print "Running for Test set"
        valdata = pd.read_csv(TEST)

merged_data = qdata.merge(invdata,on="qid", how="left").merge(udata, on="uid", how="right") 
ratings_mtx_df = merged_data.pivot_table(values='answered',
                                             index='uid',
                                             columns='qid')


ratings_mtx_df = ratings_mtx_df.reindex(udata.uid)
ratings_mtx_df = pd.concat([ratings_mtx_df,pd.DataFrame(columns=qdata.qid)])

data = ratings_mtx_df.fillna(0)


user_preferences = data.as_matrix()
user_similarity = pairwise_distances(user_preferences, metric='cosine')


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred


user_prediction = predict(user_preferences, user_similarity, type='user')

user_predictions_df = pd.DataFrame(data=user_prediction,index=data.index,columns=data.columns)

test_invdata=valdata.reset_index()
test_invdata["label"] = ""


n_validation_users = len(test_invdata.index)
for i in range(n_validation_users):
    qid_val = test_invdata.iloc[i]['qid']
    uid_val = test_invdata.iloc[i]['uid']
    test_invdata.iloc[i, test_invdata.columns.get_loc('label')] = user_predictions_df.loc[uid_val,qid_val]



test_invdata.to_csv('userbased_recommender.csv', separator=",")

