
# coding: utf-8

# In[2]:

import pandas as pd


# In[11]:

cf_results = pd.read_csv("CF-testresults.csv")
cb_results = pd.read_csv("CB-testresults.csv")
hybrid_results = pd.read_csv("CF-testresults.csv")


# In[17]:

hybrid_results = hybrid_results.drop('label',1)


# In[20]:

hybrid_results["label_0.1"] = ""
hybrid_results["label_0.2"] = ""
hybrid_results["label_0.3"] = ""
hybrid_results["label_0.4"] = ""
hybrid_results["label_0.5"] = ""
hybrid_results["label_0.6"] = ""
hybrid_results["label_0.7"] = ""
hybrid_results["label_0.8"] = ""
hybrid_results["label_0.9"] = ""


# In[6]:

alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


# In[10]:

cf_predictions = cf_results['label']
cb_predictions = cb_results['label']


# In[ ]:

for index, i in enumerate(alpha):
    n_validation_users = len(hybrid_results.index)
    for j in range(n_validation_users):
        qid_val = hybrid_results.iloc[j]['qid']
        uid_val = hybrid_results.iloc[j]['uid']
        label = 'label_0.' + str(index+1)
        hybrid_results.iloc[j, hybrid_results.columns.get_loc(label)] = (i * cf_predictions[j]) + ((1-i)*cb_predictions[j])


hybrid_results.to_csv("hybrid_results.csv")

