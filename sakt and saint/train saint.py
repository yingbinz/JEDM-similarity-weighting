#!/usr/bin/env python
# coding: utf-8

# # config

# In[1]:


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

# ## Tune hyperparameters

# In[2]:


lr_list = [0.0001, 0.001]
d_list = [20, 50]
nlayer_list = [2, 4, 6]
dropout_list = [0.2, 0.4]


# ### lr 0.0001

# In[3]:

aucs_list = []
lr = lr_list[0]
config['train_config']['learning_rate'] = lr
for d in d_list:
    config['saint']['dim_model'] = d
    for nlayer in nlayer_list:
        config['saint']['num_en'] = nlayer
        config['saint']['num_de'] = nlayer
        for dropout in dropout_list:
            config['saint']['dropout'] = dropout
            print('lr: {}, d: {}, nlayer: {}, dropout: {}'.format(lr, d, nlayer, dropout))
            torch.manual_seed(202209)
            aucs = cv_train("saint", config = config, for_what = "train")
            # get the cv average auc  
            aucs = pd.DataFrame(aucs).mean().to_list()
            aucs_list.append(aucs)
            with open("ckpts/saint/cv_aucs1.pkl", "wb") as f:
                pickle.dump(aucs_list, f)


# ### lr 0.001 

# In[6]:


aucs_list = []
lr = lr_list[1]
config['train_config']['learning_rate'] = lr
for d in d_list:
    config['saint']['dim_model'] = d
    for nlayer in nlayer_list:
        config['saint']['num_en'] = nlayer
        config['saint']['num_de'] = nlayer
        for dropout in dropout_list:
            config['saint']['dropout'] = dropout
            print('lr: {}, d: {}, nlayer: {}, dropout: {}'.format(lr, d, nlayer, dropout))
            torch.manual_seed(202209)
            aucs = ("saint", config = config, for_what = "train")
            # get the cv average auc  
            aucs = pd.DataFrame(aucs).mean().to_list()

            aucs_list.append(aucs)
            with open("ckpts/saint/cv_aucs2.pkl", "wb") as f:
                pickle.dump(aucs_list, f)


# ## get the best par

# In[3]:


par_df = np.array([(x,y,z) for x in d_list for y in nlayer_list for z in dropout_list])
par_df = pd.DataFrame(par_df, columns = ["d", "nlayer", "drop_out"])


# In[4]:


par_df1 = par_df.copy()
par_df1['lr'] = 0.0001
par_df2 = par_df.copy()
par_df2['lr'] = 0.001
aucs_df = pd.concat([par_df1, par_df2])


# In[5]:


aucs1 = pd.read_pickle("ckpts/saint/cv_aucs1.pkl")
aucs1 = np.array(aucs1)
aucs2 = pd.read_pickle("ckpts/saint/cv_aucs2.pkl")
aucs2 = np.array(aucs2)
aucs = np.concatenate((aucs1, aucs2), axis=0)


# In[6]:


for i in range(9, 100, 10):
    aucs_df["ep"+str(i+1)] = aucs[:, i]
aucs_df.reset_index(drop=True, inplace=True)


# In[7]:


aucs_df.reset_index(drop=True, inplace=True)


# In[8]:


aucs_df


# In[9]:


aucs_df.iloc[:,4:].idxmax()


# In[13]:


aucs_df.iloc[:,4:].max()


# ## train the best

# In[52]:


config['train_config']['num_epochs'] = 80
config['train_config']['learning_rate'] = 0.0001
config['saint']['num_en'] = 4
config['saint']['num_de'] = 4
config['saint']['dim_model'] = 50
config['saint']['dropout'] = 0.2


# In[53]:


torch.manual_seed(202209)
train("saint", config = config, for_what = "train")


# ## test

# In[54]:


model_name = 'saint'

model_config = config[model_name]
train_config = config["train_config"]

seq_len = train_config["seq_len"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"  


# In[55]:


test_dataset = processingData(seq_len, for_what = "test")
test_dataset_loader = DataLoader(
    test_dataset, batch_size=len(test_dataset.u_list), shuffle=False,
    collate_fn=collate_fn, 
    generator=torch.Generator(device=device)
)


# In[56]:


model_config = config["saint"]
model = SAINT(**model_config).to(device)

checkpoint = torch.load("ckpts/saint/train_final_model.ckpt")
model.load_state_dict(checkpoint)


# In[57]:


for data in test_dataset_loader:
    q, r, qshft, rshft, m = data
    
cat = torch.zeros(q.shape, dtype=torch.int32, device = "cuda")

model.eval()
p = model(q.long(), cat, r.long())

p_last = p[:,-1].detach().cpu()
t_last = rshft[:,-1].detach().cpu()

auc = metrics.roc_auc_score(
    y_true=t_last.numpy(), y_score=p_last.numpy()
)


# In[58]:


auc


# # Spring train

# ## train the best

# In[59]:


torch.manual_seed(202209)
train("saint", config = config, for_what = "S19train")


# ## test

# In[60]:


model_name = 'saint'

model_config = config[model_name]
train_config = config["train_config"]

seq_len = train_config["seq_len"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"  


# In[61]:


test_dataset = processingData(seq_len, for_what = "test")
test_dataset_loader = DataLoader(
    test_dataset, batch_size=len(test_dataset.u_list), shuffle=False,
    collate_fn=collate_fn, 
    generator=torch.Generator(device=device)
)


# In[62]:


model_config = config["saint"]
model = SAINT(**model_config).to(device)

checkpoint = torch.load("ckpts/saint/S19train_final_model.ckpt")
model.load_state_dict(checkpoint)


# In[63]:


for data in test_dataset_loader:
    q, r, qshft, rshft, m = data
    
cat = torch.zeros(q.shape, dtype=torch.int32, device = "cuda")

model.eval()
p = model(q.long(), cat, r.long())

p_last = p[:,-1].detach().cpu()
t_last = rshft[:,-1].detach().cpu()

auc = metrics.roc_auc_score(
    y_true=t_last.numpy(), y_score=p_last.numpy()
)


# In[64]:


auc

