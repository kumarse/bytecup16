
# coding: utf-8

# In[21]:

import numpy as np
import pandas as pd
import ndcg
import glob
import multiprocessing as mp
import time

BASE  = './bytecup2016data'
IINFO = BASE + '/invited_info_train.txt'

invdata = pd.read_csv(IINFO, delim_whitespace=True, header=None, names=["qid", "uid", "label"])


# In[22]:

tasks = mp.Queue()
results = mp.Queue()
numproc = mp.cpu_count()
        
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
def handle_question_result(q_ndcg, result):
    qid = result["qid"]
    q_ndcg.ix[q_ndcg.qid == qid, 'val'] = result["val"]


# In[23]:

validation_files =  glob.glob("cross-validation-attempt*.csv")
for validation_file in validation_files:
    new_valdata = pd.read_csv(validation_file)
    q_ndcg = pd.DataFrame()
    q_ndcg['qid'] = np.unique(new_valdata.qid)
    q_ndcg['val'] = 0
    
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
print accuracy.sum() / len(accuracy)


# In[ ]:



