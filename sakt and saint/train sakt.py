#!/usr/bin/env python
# coding: utf-8

# # config

# In[14]:


import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn import metrics
import pickle

from train import cv_train, train
from processingData import processingData
from model_sakt import SAKT
from model_saint import SAINT
from utils import match_seq_len, collate_fn, reset_weights

config = {
    "train_config": {
        "batch_size": 256,
        "num_epochs": 100,
        "train_ratio": 0.8,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "seq_len": 31
    },
    "sakt": {
        "n": 31,
        "d": 50,
        "num_attn_heads": 5,
        "dropout": 0.2
    },
    "saint": {
        "dim_model": 20,
        "num_en": 2    ,
        "num_de": 2    ,
        "heads_en": 5  ,
        "heads_de": 5  ,
        "total_ex": 50 ,
        "total_cat": 1 ,
        "total_in": 2  ,
        "seq_len": 31  ,
        "dropout": 0.2 
    }
}


# # Spring + Fall train data

# ## tune hyperparameters

# In[2]:


lr_list = [0.0001, 0.001]
d_list = [20, 50, 100]
dropout_list = [0.2, 0.4]


# In[17]:


aucs_list=[]
for lr in lr_list:
    config['train_config']['learning_rate'] = lr
    for d in d_list:
        config['sakt']['d'] = d
        for dropout in dropout_list:
            config['sakt']['dropout'] = dropout
            print('lr: {}, d: {}, dropout: {}'.format(lr, d, dropout))
            torch.manual_seed(202209)
            aucs = cv_train("sakt", config = config, for_what = "train")
            aucs = pd.DataFrame(aucs).mean().to_list()
            aucs_list.append(aucs)
            with open("ckpts/sakt/cv_aucs.pkl", "wb") as f:
                pickle.dump(aucs_list, f)


# ## get the best model

# In[5]:


aucs = pd.read_pickle("ckpts/sakt/cv_aucs.pkl")
aucs = np.array(aucs)


# In[6]:


par_df = np.array([(x,y,z) for x in lr_list for y in d_list for z in dropout_list])
par_df = pd.DataFrame(par_df, columns = ["lr", "d", "drop_out"])


# In[7]:


aucs_df = par_df.copy()
for i in range(9, 100, 10):
    aucs_df["ep"+str(i+1)] = aucs[:, i]
aucs_df.reset_index(drop=True, inplace=True)


# In[11]:


aucs_df.iloc[:,3:].idxmax()


# In[12]:


aucs_df.iloc[:,3:].max()


# In[13]:


aucs_df.iloc[7,:]


# ## train best model

# In[38]:


# batch size 128
config['train_config']['num_epochs'] = 60
config['train_config']['learning_rate'] = 0.001
config['sakt']['d'] = 20
config['sakt']['dropout'] = 0.4


# In[49]:


torch.manual_seed(202209)
train("sakt", config = config, for_what = "train")


# ## test

# In[6]:


model_name = 'sakt'

model_config = config[model_name]
train_config = config["train_config"]

seq_len = train_config["seq_len"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"  


# In[7]:


test_dataset = processingData(seq_len, for_what = "test")
test_dataset_loader = DataLoader(
    test_dataset, batch_size=len(test_dataset.u_list), shuffle=False,
    collate_fn=collate_fn, 
    generator=torch.Generator(device=device)
)


# In[9]:


model_config = config["sakt"]
model = SAKT(test_dataset.num_q, **model_config).to(device)

checkpoint = torch.load("ckpts/sakt/train_final_model.ckpt")
model.load_state_dict(checkpoint)


# In[10]:


for data in test_dataset_loader:
    q, r, qshft, rshft, m = data

model.eval()
p, _ = model.forward(q.long(), r.long(), qshft.long())
p_last = p[:,-1].detach().cpu()
t_last = rshft[:,-1].detach().cpu()

auc = metrics.roc_auc_score(
    y_true=t_last.numpy(), y_score=p_last.numpy()
)


# In[11]:


auc


# # Spring train data

# ## train best model

# In[39]:


torch.manual_seed(202209)
train("sakt", config = config, for_what = "S19train")


# ## test

# In[40]:


model_name = 'sakt'

model_config = config[model_name]
train_config = config["train_config"]

seq_len = train_config["seq_len"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"  


# In[41]:


test_dataset = processingData(seq_len, for_what = "test")
test_dataset_loader = DataLoader(
    test_dataset, batch_size=len(test_dataset.u_list), shuffle=False,
    collate_fn=collate_fn, 
    generator=torch.Generator(device=device)
)


# In[42]:


model_config = config["sakt"]
model = SAKT(test_dataset.num_q, **model_config).to(device)

checkpoint = torch.load("ckpts/sakt/S19train_final_model.ckpt")
model.load_state_dict(checkpoint)


# In[43]:


for data in test_dataset_loader:
    q, r, qshft, rshft, m = data

model.eval()
p, _ = model.forward(q.long(), r.long(), qshft.long())
p_last = p[:,-1].detach().cpu()
t_last = rshft[:,-1].detach().cpu()

auc = metrics.roc_auc_score(
    y_true=t_last.numpy(), y_score=p_last.numpy()
)


# In[44]:


auc

