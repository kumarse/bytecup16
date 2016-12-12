import numpy as np
import pandas as pd
import sklearn as sk
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
wseq_matrix = cseq_vec.fit_transform(qdata.wseq).toarray()

# Do 1-of-K encoding for tags
qtags = qdata["qtag"].apply(str)
qtag_vec = CountVectorizer(tokenizer=tokenize)
qtag_matrix = qtag_vec.fit_transform(qtags).toarray()

# Convert the numpy arrays to dataframes
cseq_pd = pd.DataFrame(cseq_matrix)
wseq_pd = pd.DataFrame(wseq_matrix)
qtag_pd = pd.DataFrame(qtag_matrix)

# Merge
proc_qdata = pd.concat([qdata.qid, wseq_pd, qtag_pd, qdata.nvotes, qdata.nans, qdata.ntqans], axis = 1)


# Process the udata

# Convert the character sequence column into a bag of words kind of vector
# Refer: http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
cseq_matrix = cseq_vec.fit_transform(udata.cseq).toarray()
wseq_matrix = cseq_vec.fit_transform(udata.wseq).toarray()
exptag_matrix = qtag_vec.fit_transform(udata.exptag).toarray()

# Convert the numpy arrays to dataframes
cseq_pd = pd.DataFrame(cseq_matrix)
wseq_pd = pd.DataFrame(wseq_matrix)
exptag_pd = pd.DataFrame(exptag_matrix)

# Merge
proc_udata = pd.concat([udata.uid, wseq_pd, exptag_pd], axis = 1)


def prepare_training_data_for_user(uid):
    # Get entries for the user from invited data
    user_invdata = invdata[invdata.uid == uid]
    # Merge with processed qdata to get the training data for the user
    user_data = user_invdata.merge(proc_qdata, on="qid", how="inner").drop(["qid", "uid", "cseq"], axis = 1)
    user_train_labels = user_data.answered
    user_train_data = user_data.drop(["answered"], axis = 1)
    return user_train_data, user_train_labels


def prepare_training_data():

    # Merge with processed udata to get the training data for the user
    user_invdata = invdata.merge(proc_udata, on="uid", how="inner").drop(["wseq"], axis = 1)
    user_ques_invdata = user_invdata.merge(proc_qdata, on="qid", how="inner").drop(["uid","qid","cseq"], axis = 1)
    train_labels = user_ques_invdata.answered
    train_data = user_ques_invdata.drop(["answered"], axis = 1)
    return train_data, train_labels


def prepare_testing_data():
    # Merge with processed udata to get the testing data for the user
    user_valdata = valdata.merge(proc_udata, on="uid", how="inner").drop(["wseq"], axis = 1)
    user_ques_valdata = user_valdata.merge(proc_qdata, on="qid", how="inner").drop(["uid","qid","cseq"], axis = 1)
    test_data = user_ques_valdata.drop(["label"], axis = 1)
    return test_data

from sklearn import linear_model
#%%timeit -n 1
train_data, train_labels = prepare_training_data()
test_data = prepare_testing_data()

regr = linear_model.LogisticRegression()
regr.fit(train_data, train_labels)

predicted_proba = regr.predict_proba(test_data)

valdata['label'] = predicted_proba[:, 1]


def get_val_data_for_user(uid):
    user_valdata = valdata[valdata.uid == uid]
    user_valdata = user_valdata.merge(proc_qdata, on="qid", how="inner").drop(["cseq", "label"], axis = 1)
    return user_valdata


# Write output as CSV
valdata.to_csv("logistic_model_full_data.csv")