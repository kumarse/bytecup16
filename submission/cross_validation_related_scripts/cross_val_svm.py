import numpy as np
import pandas as pd
from sklearn import svm
import ndcg
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

BASE  = './bytecup2016data'
IINFO = BASE + '/invited_info_train.txt'
QINFO = BASE + '/question_info.txt'
UINFO = BASE + '/user_info.txt'
VAL   = BASE + '/validate_nolabel.txt'

invdata = pd.read_csv(IINFO, delim_whitespace=True, header=None, names=["qid", "uid", "label"])
qdata   = pd.read_csv(QINFO, delim_whitespace=True, header=None, names=["qid", "qtag", "wseq", "cseq", "nvotes", "nans", "ntqans"])
udata   = pd.read_csv(UINFO, delim_whitespace=True, header=None, names=["uid", "exptag", "wseq", "cseq"])
valdata = pd.read_csv(VAL)

from sklearn.feature_extraction.text import CountVectorizer

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

# Insert a column in valdata to store the predicted label probabilities
valdata.insert(2, "label", value = 0.0)

def prepare_training_data_for_user(new_invdata, uid):
    # Get entries for the user from invited data
    user_invdata = new_invdata[new_invdata.uid == uid]
    # Merge with processed qdata to get the training data for the user
    user_data = user_invdata.merge(proc_qdata, on="qid", how="inner").drop(["qid", "uid", "wseq"], axis = 1)
    user_train_labels = user_data.label
    user_train_data = user_data.drop(["label"], axis = 1)
    return user_train_data, user_train_labels

def get_val_data_for_user(new_valdata, uid):
    user_valdata = new_valdata[new_valdata.uid == uid]
    user_valdata = user_valdata.merge(proc_qdata, on="qid", how="inner").drop(["wseq", "label"], axis = 1)
    return user_valdata

from sklearn import linear_model
import multiprocessing as mp
import time

tasks = mp.Queue()
results = mp.Queue()
numproc = mp.cpu_count()

# Map function
def handle_user(users_queue, new_invdata, new_valdata, param, results_queue):
    while True:
        uid = users_queue.get()
        if uid is None:
            break
        user_unique_labels = np.unique(new_invdata[new_invdata.uid == uid].label)

        if len(user_unique_labels) != 1:
            user_train_data, user_train_labels = prepare_training_data_for_user(new_invdata, uid)
            if user_train_data.shape[0] > 0:
                #regr = linear_model.LogisticRegression()
                regr = svm.SVC(kernel= param['kernel'], C= param['C'], gamma= param['gamma'], probability=True)
                regr.fit(user_train_data, user_train_labels)
                
        user_val_data = get_val_data_for_user(new_valdata, uid)
        user_val_trimmed_data = user_val_data.drop(["qid", "uid"], axis = 1)

        if len(user_unique_labels) != 1 and user_train_data.shape[0] > 0:
            predicted_proba = regr.predict_proba(user_val_trimmed_data)
        else:
            if len(user_unique_labels) == 0:
                user_unique_labels = [0]
            predicted_proba = np.array([[0.0, 1.0] if user_unique_labels[0] == 1 else [1.0, 0.0] for i in range(user_val_data.shape[0])])

        results_queue.put({"uid": uid, "labels": predicted_proba[:, 1]})
        
def handle_question(question_queue, invdata, new_valdata, results_queue):
     while True:
        qid = question_queue.get()
        if qid is None:
            break
            
        q_result = new_valdata[new_valdata.qid == qid]
        q_result.reset_index(drop = True, inplace = True)
        sorted_q_result = q_result.sort_values(['label'], axis=0, ascending=False)
        sorted_users = sorted_q_result['uid']
        r = []
        for uid in sorted_users:
            r.append(invdata[(invdata.qid == qid) & (invdata.uid == uid)].values[0][2])
        results_queue.put({"qid": qid, "val": (ndcg.ndcg_at_k(r,5) * 0.5) + (ndcg.ndcg_at_k(r,10) * 0.5)})

# Reduce function
def handle_user_result(new_valdata, result):
    uid = result["uid"]
    new_valdata.ix[new_valdata.uid == uid, 'label'] = result["labels"]
    
# Reduce function
def handle_question_result(q_ndcg, result):
    qid = result["qid"]
    q_ndcg.ix[q_ndcg.qid == qid, 'val'] = result["val"]

skf = StratifiedKFold(n_splits = 5)
invdata_label = invdata['label']
for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
    for gamma in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        param = {}
        param['kernel'] = 'rbf'
        param['C'] = c
        param['gamma'] = gamma
        accuracy = []
        for train_invdata, test_valdata in skf.split(invdata,invdata_label):
            new_invdata = invdata.loc[train_invdata]
            new_valdata = invdata.loc[test_valdata]
            new_invdata.reset_index(drop = True, inplace = True)
            new_valdata.reset_index(drop = True, inplace = True)
            q_ndcg = pd.DataFrame()
            q_ndcg['qid'] = np.unique(new_valdata.qid)
            q_ndcg['val'] = 0

            unique_users = np.unique(new_valdata.uid)
            num_unique = len(unique_users)
            # Queue up the users
            for uid in unique_users:
                tasks.put(uid)

            # Put poison pills
            for i in range(numproc):
                tasks.put(None)

            procs = []
            for i in range(numproc):
                p = mp.Process(target=handle_user, args=(tasks, new_invdata, new_valdata, param, results,))
                procs.append(p)
                p.start()

            start = time.time()
            num_results = 0
            while True:
                res = results.get()
                handle_user_result(new_valdata, res)
                num_results += 1
                if num_results == len(unique_users):
                    break
            end = time.time()

            print "Time elapsed = ", end - start


            unique_questions = np.unique(new_valdata.qid)
            num_unique = len(unique_questions)
            # Queue up the users
            for qid in unique_questions:
                tasks.put(qid)

            # Put poison pills
            for i in range(numproc):
                tasks.put(None)

            procs = []
            for i in range(numproc):
                p = mp.Process(target=handle_question, args=(tasks, invdata, new_valdata, results,))
                procs.append(p)
                p.start()

            start = time.time()
            num_results = 0
            while True:
                res = results.get()
                handle_question_result(q_ndcg, res)
                num_results += 1
                if num_results == len(unique_questions):
                    break
            end = time.time()

            print "Time elapsed = ", end - start

            accuracy.append(q_ndcg['val'].sum() / q_ndcg.index.size)
            print accuracy[-1]
        print accuracy



