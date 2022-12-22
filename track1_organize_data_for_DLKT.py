# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 00:05:05 2022

@author: yingb
"""

# %% load library
#from operator import index
import pandas as pd
import numpy as np
import os
#%% load data
####### if only use S19 train data, i.e., phase 1 train data, set term = "S19"
term = "S19"
early_train_F = pd.read_csv('data/F19/Train/early.csv')
late_train_F = pd.read_csv('data/F19/Train/late.csv')
raw_data_early_train_F = pd.read_csv('data/ours/raw_train_early_F19.csv')
raw_data_late_train_F = pd.read_csv('data/ours/raw_train_late_F19.csv')

X_train_base_F = late_train_F.copy().drop('Label', axis=1)
y_train_F = late_train_F['Label'].values

early_train_S = pd.read_csv('data/S19/ALL/early.csv')
late_train_S = pd.read_csv('data/S19/All/late.csv')
raw_data_early_train_S = pd.read_csv('data/ours/raw_train_early_S19_All.csv')
raw_data_late_train_S = pd.read_csv('data/ours/raw_train_late_S19_All.csv')

X_train_base_S = late_train_S.copy().drop('Label', axis=1)
y_train_S = late_train_S['Label'].values

early_train = pd.concat([early_train_F, early_train_S], axis = 0).reset_index(drop=True)
late_train = pd.concat([late_train_F, late_train_S], axis = 0).reset_index(drop=True)
raw_data_early_train = pd.concat([raw_data_early_train_F, raw_data_early_train_S], axis = 0).reset_index(drop=True)
raw_data_late_train = pd.concat([raw_data_late_train_F, raw_data_late_train_S], axis = 0).reset_index(drop=True)
X_train_base = pd.concat([X_train_base_F, X_train_base_S], axis = 0, join = 'inner').reset_index(drop=True)
y_train = np.concatenate((y_train_F, y_train_S))

TEST_PATH = "data/F19/Test"
early_test = pd.read_csv(TEST_PATH+'/early.csv')

raw_data_early_test = pd.read_csv('data/ours/raw_test_early_F19.csv')
late_test = pd.read_csv(TEST_PATH+'/late.csv')

# only s19 data
if term != "":
    early = pd.concat([early_train, early_test], axis = 0, join = 'inner').reset_index(drop=True)
    SubjectIDs = early['SubjectID'].values
    ProblemIDs = early['ProblemID'].values
    early_train = pd.read_csv('data/S19/Train/early.csv')
    late_train = pd.read_csv('data/S19/Train/late.csv')
    raw_data_early_train = pd.read_csv('data/ours/raw_train_early_S19.csv')
    raw_data_late_train = pd.read_csv('data/ours/raw_train_late_S19.csv')
    
    X_train_base = late_train.copy().drop('Label', axis=1)
    y_train = late_train['Label'].values
    
def get_attempt_order(raw_data_early):
    
    order = raw_data_early.groupby(['SubjectID', "ProblemID"]).agg(
        probStartTime = ('ServerTimestamp', lambda x: np.min(x))).reset_index(drop=False)
    order = order.sort_values(by = ['SubjectID', 'probStartTime']).reset_index(drop=True)
    order['AttemptID'] = order.index.values+1
    return order[['SubjectID', 'ProblemID', 'AttemptID']]

train_order = get_attempt_order(raw_data_early_train)
early_train = early_train.merge(train_order, how = "left", on = ['SubjectID', "ProblemID"])

early_order = get_attempt_order(raw_data_early_train)
early_test = early_test.merge(early_order, how = "left", on = ['SubjectID', "ProblemID"])

# %% create sequneces of early problem + 1 late problem 

def early_and_one_late(early, late):
    early = early.copy()
    late = late.copy()
    
    early = early.sort_values(by = ['SubjectID', 'AttemptID']).reset_index(drop=True)
    
    cols = ['SubjectID', "ProblemID","AttemptID","Label"]
    ids = list(set(late.SubjectID))
    individual_list = []
    for SubjectID in ids:
    
        early_individual = early.loc[early.SubjectID == SubjectID,cols].copy()
        if early_individual.shape[0] == 0:
            continue
        individual = late.loc[late.SubjectID == SubjectID,:].copy()
        individual["AttemptID"] = [early_individual["AttemptID"].iloc[-1]+rindex+1 for rindex in range(individual.shape[0])]
        individual["SubjectID"] = [individual["SubjectID"].iloc[rindex]+"pid"+str(individual["ProblemID"].iloc[rindex]) for rindex in range(individual.shape[0])]
        individual = individual[cols]
        
        temp_list = []
        for pid in individual["ProblemID"].values:
            temp = early_individual.copy()
            temp["SubjectID"]= temp["SubjectID"]+"pid"+str(pid)
            temp_list.append(temp)
        temp = pd.concat(temp_list, axis = 0).reset_index(drop=True)
        individual = pd.concat([temp,individual], axis = 0).reset_index(drop=True)
        individual_list.append(individual)
    early_1_late = pd.concat(individual_list, axis = 0).reset_index(drop=True)
    early_1_late = early_1_late.sort_values(by = ['SubjectID', 'AttemptID']).reset_index(drop=True)
    early_1_late['AttemptID'] = early_1_late.index.values+1
    return early_1_late

train = early_and_one_late(early_train, late_train)
test = early_and_one_late(early_test, late_test)
train['Label'] = train['Label']*1
test['Label'] = test['Label']*1

# %% output
path = os.path.join('data', 'DLKT')
os.makedirs(path, exist_ok=True)
train.to_csv(os.path.join(path, term+'train.csv'), index=False)
test.to_csv(os.path.join(path, 'test.csv'), index=False)
