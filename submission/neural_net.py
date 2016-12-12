
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sys

# In[2]:

import os


# # Data loading

# In[25]:

BASE  = './bytecup2016data'
IINFO = os.path.join(BASE, 'invited_info_train.txt')
QINFO = os.path.join(BASE, 'question_info.txt')
UINFO = os.path.join(BASE, 'user_info.txt')
VAL   = os.path.join(BASE, 'validate_nolabel.txt')
TEST  = os.path.join(BASE, 'test_nolabel.txt')

# In[26]:

invdata = pd.read_csv(IINFO, delim_whitespace=True, header=None, names=["qid", "uid", "label"])
qdata = pd.read_csv(QINFO, delim_whitespace=True, header=None,
                    names=["qid", "qtag", "wseq", "cseq", "nvotes", "nans", "ntqans"])
udata = pd.read_csv(UINFO, delim_whitespace=True, header=None, names=["uid", "exptag", "wseq", "cseq"])
valdata = pd.read_csv(VAL)

# Insert a column in valdata to store the predicted label probabilities
valdata.insert(2, "label", value = 0.0)
if len(sys.argv) > 1:
    param1 = sys.argv[1]
    if param1 == '1':
        print "Running for Test set"
        valdata = pd.read_csv(TEST)

print valdata.shape

# In[27]:

# Normalize the data
for col in ['nvotes', 'nans', 'ntqans']:
    qdata[col] = (qdata[col] - qdata[col].min())/(qdata[col].max() - qdata[col].min())


# In[28]:

from sklearn.feature_extraction.text import CountVectorizer


# In[29]:

def tokenize(text):
    return text.split("/")

count_vectorizer = CountVectorizer(tokenizer=tokenize)


# # Form Question Matrix

# In[30]:

# Convert the character sequence column into a bag of words kind of vector
# Refer: http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
cseq_matrix = count_vectorizer.fit_transform(qdata.cseq).toarray()

# Do 1-of-K encoding for tags
qtags = qdata["qtag"].apply(str)
qtag_matrix = count_vectorizer.fit_transform(qtags).toarray()

# Convert the numpy arrays to dataframes
cseq_pd = pd.DataFrame(cseq_matrix)
qtag_pd = pd.DataFrame(qtag_matrix)

# Merge
proc_qdata = pd.concat([qdata.qid, cseq_pd, qtag_pd, qdata.nvotes, qdata.nans, qdata.ntqans], axis=1)


# # Form User Matrix

# In[31]:

# Convert the character sequence column into a bag of words kind of vector
# Refer: http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
ucseq_matrix = count_vectorizer.fit_transform(udata.cseq).toarray()

# Do 1-of-K encoding for tags
utags = udata["exptag"].apply(str)
utag_matrix = count_vectorizer.fit_transform(utags).toarray()

# Convert the numpy arrays to dataframes
ucseq_pd = pd.DataFrame(ucseq_matrix)
utag_pd = pd.DataFrame(utag_matrix)

# Merge
proc_udata = pd.concat([udata.uid, ucseq_pd, utag_pd], axis=1)


# # Generator for constructing batch data

# In[66]:

def generate_next_batch():
    batch_size = 256
    num_rows = len(invdata)
    num_batches = num_rows/batch_size
    while True:
        shuffled_invdata = invdata.iloc[np.random.permutation(num_rows)]
        for i in xrange(num_batches):
            batch_data = shuffled_invdata[i * batch_size : (i+1) * batch_size]
            qbatch = batch_data.merge(proc_qdata, on='qid', how='inner').drop(['qid', 'uid', 'label'], axis = 1)
            ubatch = batch_data.merge(proc_udata, on='uid', how='inner').drop(['qid', 'uid', 'label'], axis = 1)
            labels = batch_data['label']
            yield ([ubatch.values, qbatch.values], to_categorical(labels.values))


# # Model definition

# In[41]:

print proc_udata.shape
print proc_qdata.shape


# In[43]:

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge

import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical


# In[78]:

qinput_dim = proc_qdata.shape[1] - 1 # Except qid column
qbranch = Sequential()
qbranch.add(Dense(input_dim=qinput_dim, output_dim=1596, activation='relu', 
                W_regularizer=Reg.l2(l=5e-7), init='glorot_normal'))

qbranch.add(Dense(input_dim=1596, output_dim=1024, activation='relu', 
                W_regularizer=Reg.l2(l=5e-7), init='glorot_normal'))


uinput_dim = proc_udata.shape[1] - 1 # Except uid column
ubranch = Sequential()
ubranch.add(Dense(input_dim=uinput_dim, output_dim=2048, activation='relu', 
                W_regularizer=Reg.l2(l=5e-7), init='glorot_normal'))
ubranch.add(Dense(input_dim=uinput_dim, output_dim=1024, activation='relu', 
                W_regularizer=Reg.l2(l=5e-7), init='glorot_normal'))

merged = Merge([ubranch, qbranch], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(output_dim=1024, activation='relu', 
                W_regularizer=Reg.l2(l=5e-7), init='glorot_normal'))
final_model.add(Dense(output_dim=512, activation='relu', 
                W_regularizer=Reg.l2(l=5e-7), init='glorot_normal'))
final_model.add(Dense(2, activation='softmax'))


# # Compile and train model

# In[81]:

final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = final_model.fit_generator(generate_next_batch(), samples_per_epoch=len(invdata) - len(invdata)%256, nb_epoch = 50, verbose = 1)
print hist.history
final_model.save('neural_net_attempt1.model')


# # Run on validation and store CSV

# In[108]:

def predict_for_val_data():
    batch_size = 256
    num_rows = len(valdata)
    num_batches = num_rows/batch_size
    valdata['label'] = 0
    for i in xrange(num_batches):
        batch_data = valdata[i * batch_size : (i+1) * batch_size]
        qbatch = batch_data.merge(proc_qdata, on='qid', how='inner').drop(['qid', 'uid', 'label'], axis = 1)
        ubatch = batch_data.merge(proc_udata, on='uid', how='inner').drop(['qid', 'uid', 'label'], axis = 1)
        out = final_model.predict_proba([ubatch.values, qbatch.values], batch_size=batch_size)
        valdata.ix[i * batch_size : (i+1) * batch_size - 1, 'label'] = out[:, 1]
    if len(valdata) % batch_size != 0:
        i = len(valdata)/batch_size
        batch_data = valdata[i * batch_size : ]
        qbatch = batch_data.merge(proc_qdata, on='qid', how='inner').drop(['qid', 'uid', 'label'], axis = 1)
        ubatch = batch_data.merge(proc_udata, on='uid', how='inner').drop(['qid', 'uid', 'label'], axis = 1)
        out = final_model.predict_proba([ubatch.values, qbatch.values], batch_size=len(valdata)%256)
        valdata.ix[i * batch_size : , 'label'] = out[:, 1]


# In[ ]:

predict_for_val_data()


# In[112]:

valdata.to_csv("attempt_neural.csv")


# In[ ]:



