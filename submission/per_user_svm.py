import numpy as np
import pandas as pd
from sklearn import svm
import sys

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

# Insert a column in valdata to store the predicted label probabilities
valdata.insert(2, "label", value = 0.0)
if len(sys.argv) > 1:
    param1 = sys.argv[1]
    if param1 == '1':
        print "Running for Test set"
        valdata = pd.read_csv(TEST)
    



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


def prepare_training_data_for_user(uid):
    # Get entries for the user from invited data
    user_invdata = invdata[invdata.uid == uid]
    # Merge with processed qdata to get the training data for the user
    user_data = user_invdata.merge(proc_qdata, on="qid", how="inner").drop(["qid", "uid", "wseq"], axis = 1)
    user_train_labels = user_data.answered
    user_train_data = user_data.drop(["answered"], axis = 1)
    return user_train_data, user_train_labels

def get_val_data_for_user(uid):
    user_valdata = valdata[valdata.uid == uid]
    user_valdata = user_valdata.merge(proc_qdata, on="qid", how="inner").drop(["wseq", "label"], axis = 1)
    return user_valdata

from sklearn import linear_model
import multiprocessing as mp
import time

tasks = mp.Queue()
results = mp.Queue()
numproc = mp.cpu_count()

# Map function
def handle_user(users_queue, results_queue):
    while True:
        uid = users_queue.get()
        if uid is None:
            break
        user_unique_labels = np.unique(invdata[invdata.uid == uid].answered)

        if len(user_unique_labels) != 1:
            user_train_data, user_train_labels = prepare_training_data_for_user(uid)
            if user_train_data.shape[0] > 0:
                #regr = linear_model.LogisticRegression()
                regr = svm.SVC(C=1, gamma=0.1, probability=True)
                regr.fit(user_train_data, user_train_labels)

        user_val_data = get_val_data_for_user(uid)
        user_val_trimmed_data = user_val_data.drop(["qid", "uid"], axis = 1)

        if len(user_unique_labels) != 1 and user_train_data.shape[0] > 0:
            predicted_proba = regr.predict_proba(user_val_trimmed_data)
        else:
            if len(user_unique_labels) == 0:
                user_unique_labels = [0]
            predicted_proba = np.array([[0.0, 1.0] if user_unique_labels[0] == 1 else [1.0, 0.0] for i in range(user_val_data.shape[0])])

        results_queue.put({"uid": uid, "labels": predicted_proba[:, 1]})

# Reduce function
def handle_result(result):
    uid = result["uid"]
    valdata.ix[valdata.uid == uid, 'label'] = result["labels"]

unique_users = np.unique(valdata.uid)
num_unique = len(unique_users)

# Queue up the users
for uid in unique_users:
    tasks.put(uid)

# Put poison pills
for i in range(numproc):
    tasks.put(None)

procs = []
for i in range(numproc):
    p = mp.Process(target=handle_user, args=(tasks, results,))
    procs.append(p)
    p.start()

start = time.time()
num_results = 0
while True:
    res = results.get()
    handle_result(res)
    num_results += 1
    if num_results == len(unique_users):
        break
end = time.time()

print "Time elapsed = ", end - start

# Write output as CSV
valdata.to_csv("per_user_svm.csv")