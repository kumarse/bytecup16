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
TEST   = BASE + '/test_nolabel.txt'

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


from scipy.sparse import coo_matrix
train = coo_matrix(user_preferences)


merged_data_test = qdata.merge(valdata,on="qid", how="left").merge(udata, on="uid", how="right") 
ratings_mtx_test = merged_data_test.pivot_table(values='answered',
                                             index='uid',
                                             columns='qid')


ratings_mtx_test = ratings_mtx_test.reindex(udata.uid)
ratings_mtx_test = pd.concat([ratings_mtx_test,pd.DataFrame(columns=qdata.qid)])
test_data = ratings_mtx_test.fillna(0)


user_preferences_test = test_data.as_matrix()
test = coo_matrix(user_preferences_test)


from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

model = LightFM(learning_rate=0.05, loss='warp')

model.fit_partial(train, epochs=10)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))


test_qids = valdata.qid.tolist()
test_uids = valdata.uid.tolist()
all_uids = udata['uid'].tolist()
all_qids = qdata['qid'].tolist()


n_test = len(test_qids)
test_qids_int = []
test_uids_int = []
for i in range(n_test):
    test_qids_int.append(all_qids.index(test_qids[i]))
    test_uids_int.append(all_uids.index(test_uids[i]))


test_uids_int = np.array(test_uids_int, dtype=np.int32)
test_qids_int = np.array(test_qids_int, dtype=np.int32)


valdata['label']=model.predict(test_uids_int,test_qids_int)

valdata.to_csv('bpr.csv')

