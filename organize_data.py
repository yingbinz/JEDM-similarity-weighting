# %%
import pandas as pd
import numpy as np
from ProgSnap2 import ProgSnap2Dataset
from ProgSnap2 import PS2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
from os import path
import re
# %% cross semester
train_semester = 'S19/Train' 
TRAIN_PATH = os.path.join('data', train_semester)

test_semester = 'F19'
BASE_PATH = os.path.join('data', test_semester)
TEST_PATH = os.path.join(BASE_PATH, 'Test')

# %%
def get_raw_data(path, is_train):

    train_ps2 = ProgSnap2Dataset(os.path.join(path, 'Data'))


    # The early dataset will help us to feature extraction,
    # but we're not actually predicting anything here
    # Note: we could still use this for model training if desired.
    early_train = pd.read_csv(os.path.join(path, 'early.csv'))
    #early_train.head()

    def get_code_table(ps2_dataset):
        events_table = ps2_dataset.get_main_table()
        code_states = ps2_dataset.get_code_states_table()
        runs = events_table.merge(code_states, on=PS2.CodeStateID)
        runs = runs[runs[PS2.EventType] == 'Run.Program']
        runs = runs[[PS2.Order, PS2.SubjectID, PS2.ProblemID, 'Code']]
        return runs

    code_table_train = get_code_table(train_ps2)

    events_table = train_ps2.get_main_table()
    code_states = train_ps2.get_code_states_table()

    def isNaN(string):
        return string != string
    if not is_train:
        events_table["SourceLocation"] = [np.nan if isNaN(x) else x.split(":")[0].replace("line ", "Text:") for x in events_table["CompileMessageData"]]


# =============================================================================
#     events_table_wError = events_table[events_table['EventType'] == 'Compile.Error'][['ParentEventID', 'CompileMessageType', 'CompileMessageData', 'SourceLocation']]
#     
#     # Organize compiler error data per error
#     events_table_wError['CompileError'] = events_table_wError.apply(lambda row: {'CompileMessageType': row['CompileMessageType'], 'CompileMessageData': row['CompileMessageData'], 'SourceLocation': row['SourceLocation']}, axis=1)
#     events_table_wError['CompileErrorNew'] = events_table_wError.apply(lambda row: re.sub('line \d{0,3}: error: ', "",row['CompileMessageData']), axis=1)
# =============================================================================

    events_table_wError = events_table[events_table['EventType'] == 'Compile.Error'][['ParentEventID', 'CompileMessageType', 'CompileMessageData']]
    

    #events_table_wError = events_table[events_table['EventType'] == 'Compile.Error'][['ParentEventID', 'CompileMessageType', 'CompileMessageData', 'SourceLocation']]

    # Organize compiler error data per error
    events_table_wError['CompileError'] = events_table_wError.apply(lambda row: {'CompileMessageType': row['CompileMessageType'], 'CompileMessageData': row['CompileMessageData']}, axis=1)
    events_table_wError['CompileErrorNew'] = events_table_wError.apply(lambda row: re.sub('line \d{0,3}: error: ', "",row['CompileMessageData']), axis=1)


    # Aggregate compiler error data at submission level

    errors = pd.DataFrame(columns=['EventID', 'CompileErrors'])

    tot = len(events_table_wError['ParentEventID'].unique())
    i = 1

    for parent in events_table_wError['ParentEventID'].unique():
        df_temp = events_table_wError[events_table_wError['ParentEventID'] == parent]
        errors_detail_list = list(df_temp['CompileError'].values)
        errors_list = list(df_temp['CompileErrorNew'].values)
        grandparent = events_table[events_table['EventID'] == parent]['ParentEventID'].iloc[0]

        errors = errors.append({'EventID': grandparent,
                                'CompileErrorsDetail': errors_detail_list,
                                'CompileErrors': errors_list}, ignore_index=True)

        if i % 500 == 0:
            print(i, '/', tot)
        i = i + 1

    # Create single dataframe with all pertinent data
    if is_train:
        runs = events_table.merge(errors, on='EventID', how='left')
        runs = runs[runs['EventType'] == 'Run.Program']
        runs = runs.merge(code_states, on='CodeStateID')
        runs = runs[['SubjectID', 'ServerTimestamp', 'AssignmentID', 'ProblemID', 'Score', 'CompileErrorsDetail', 'CompileErrors', 'Code']]

        # extract submissions on late problems to compute features about the late problems
        runs_late = runs[~runs['AssignmentID'].isin(early_train.AssignmentID.unique())]

        # Keep only submissions on early problems (to keep only the data we'll have for the challenge)
        runs_early = runs[runs['AssignmentID'].isin(early_train.AssignmentID.unique())]

        return runs_early, runs_late
    else:
        runs = events_table.merge(errors, on='EventID', how='left')
        runs = runs[runs['EventType'] == 'Run.Program']
        runs = runs.merge(code_states, on='CodeStateID')
        runs = runs[['SubjectID', 'ServerTimestamp', 'AssignmentID', 'ProblemID', 'Score', 'CompileErrorsDetail', 'CompileErrors', 'Code']]

        return runs
# %%
train_runs_early, train_runs_late = get_raw_data(TRAIN_PATH, is_train=True)
test_runs_early = get_raw_data(TEST_PATH, is_train=False)


train_runs_early.to_csv('data/ours/raw_train_early_'+'S19'+'.csv', index=False)
train_runs_late.to_csv('data/ours/raw_train_late_'+'S19'+'.csv', index=False)


test_runs_early.to_csv('data/ours/raw_test_early_F19.csv', index=False)
# %%
# Added block for Phase 2
## S19 All
S19_all_path = 'data/S19/All'

S19_all_train_runs_early, S19_all_train_runs_late = get_raw_data(S19_all_path, is_train=True)

S19_all_train_runs_early.to_csv('data/ours/raw_train_early_'+'S19_All'+'.csv', index=False)
S19_all_train_runs_late.to_csv('data/ours/raw_train_late_'+'S19_All'+'.csv', index=False)

## F19 train
F19_path = 'data/F19/Train'
F19_train_runs_early, F19_train_runs_late = get_raw_data(F19_path, is_train=True)

F19_train_runs_early.to_csv('data/ours/raw_train_early_F19.csv', index=False)
F19_train_runs_late.to_csv('data/ours/raw_train_late_F19.csv', index=False)
