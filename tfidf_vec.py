import pandas as pd
import numpy as np
from ProgSnap2 import ProgSnap2Dataset
from ProgSnap2 import PS2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

from scipy.sparse import vstack

def get_code_table(ps2_dataset):
    events_table = ps2_dataset.get_main_table()
    code_states = ps2_dataset.get_code_states_table()
    runs = events_table.merge(code_states, on=PS2.CodeStateID)
    runs = runs[runs[PS2.EventType] == "Run.Program"]
    runs = runs[[PS2.Order, PS2.SubjectID, PS2.ProblemID, "Code"]]
    return runs
# %%
test_semester = "F19"
TEST_PATH = os.path.join("data", test_semester, "Test")

# read S19 all
train_semester = "S19"
TRAIN_PATH = os.path.join("data", train_semester, "All")
train_ps2 = ProgSnap2Dataset(os.path.join(TRAIN_PATH, "Data"))
# The early dataset will help us to feature extraction,
# but we're not actually predicting anything here
# Note: we could still use this for model training if desired.
early_train_S = pd.read_csv(os.path.join(TRAIN_PATH, "early.csv"))
# The late dataset contains the problems that we're actually predicting for.
# The training portion of it includes labels.
late_train_S = pd.read_csv(os.path.join(TRAIN_PATH, "late.csv"))
# late_train.head()
code_table_train_S = get_code_table(train_ps2)

# read F19
train_semester = "F19"
TRAIN_PATH = os.path.join("data", train_semester, "Train")
train_ps2 = ProgSnap2Dataset(os.path.join(TRAIN_PATH, "Data"))
# The early dataset will help us to feature extraction,
# but we're not actually predicting anything here
# Note: we could still use this for model training if desired.
early_train_F = pd.read_csv(os.path.join(TRAIN_PATH, "early.csv"))
# The late dataset contains the problems that we're actually predicting for.
# The training portion of it includes labels.
late_train_F = pd.read_csv(os.path.join(TRAIN_PATH, "late.csv"))
# late_train.head()
code_table_train_F = get_code_table(train_ps2)
code_table_train = pd.concat([code_table_train_F, code_table_train_S], axis = 0).reset_index(drop=True)

# combine S19 and F19
early_train = pd.concat([early_train_F, early_train_S], axis = 0).reset_index(drop=True)
late_train = pd.concat([late_train_F, late_train_S], axis = 0).reset_index(drop=True)

X_train_base = late_train.copy().drop("Label", axis=1)
y_train = late_train["Label"].values

problem_encoder = OneHotEncoder().fit(X_train_base[PS2.ProblemID].values.reshape(-1, 1))

problem_encoder.transform(X_train_base[PS2.ProblemID].values.reshape(-1, 1)).toarray()





# assign "" to rows where "Code" is na
code_table_train.loc[code_table_train['Code'].isna(), "Code"] = ""
# code_table_train

# We want to find a consistent, common vocabulary across all problems
# so we first build our vocabulary for all code submissions


# Note this approach is _very_ naive, since it's using NLP assumptions
# about tokenizing, among other things, but it is good enough for a demonstration.
code_vectorizer = TfidfVectorizer(max_features=50)
code_vectorizer.fit(code_table_train["Code"])
top_vocab = code_vectorizer.vocabulary_
# top_vocab

# We want to create a separate encoder for each problem, since the
# "document frequency" part of TF-IDF should be calibrated separately
# for each problem.
code_problem_encoders = {}


def create_encoder(rows):
    code = rows["Code"]
    problem_id = rows[PS2.ProblemID].iloc[0]
    code_vectorizer = TfidfVectorizer(vocabulary=top_vocab)
    code_vectorizer.fit(code)
    code_problem_encoders[problem_id] = code_vectorizer


code_table_train.groupby(PS2.ProblemID).apply(create_encoder)

# len(code_problem_encoders)


# test_code = code_table_train['Code'].iloc[0]
# print(test_code)
# print(code_problem_encoders[1].transform([test_code]))


# def extract_instance_features(instance, early_df):
#     instance = instance.copy()
#     subject_id = instance[PS2.SubjectID]
#     early_problems = early_df[early_df[PS2.SubjectID] == subject_id]
#     # Extract very naive features about the student
#     # (without respect to the problem bring predicted)
#     # Number of early problems attempted
#     instance['ProblemsAttempted'] = early_problems.shape[0]
#     # Percentage of early problems gotten correct eventually
#     instance['PercCorrectEventually'] = np.mean(early_problems['CorrectEventually'])
#     # Median attempts made on early problems
#     instance['MedAttempts'] = np.median(early_problems['Attempts'])
#     # Max attempts made on early problems
#     instance['MaxAttempts'] = np.max(early_problems['Attempts'])
#     # Percentage of problems gotten correct on the first try
#     instance['PercCorrectFirstTry'] = np.mean(early_problems['Attempts'] == 1)

#     instance = instance.drop('SubjectID')
#     return instance


# extract_instance_features(X_train_base.iloc[0], early_train)


def extract_instance_code_features(instance, early_df, code_table):
    subject_id = instance[PS2.SubjectID]
    problem_id = instance[PS2.ProblemID]

    # Get all attempts for this problem by this subject
    attempts = code_table[
        (code_table[PS2.SubjectID] == subject_id)
        & (code_table[PS2.ProblemID] == problem_id)
    ]
    # Get the code of the last attempt (we could use others but don't here)
    encoder = code_problem_encoders[problem_id]
    # If for some reason there were no attempts, return 0s
    if attempts.shape[0] == 0:
        return encoder.transform([""])
    last_attempt = attempts.sort_values("Order")["Code"].iloc[-1]
    code_features = encoder.transform([last_attempt])

    return code_features


# print(extract_instance_code_features(X_train_base.iloc[0], early_train, code_table_train))

# Test how to stack code features across instances
# import functools

# code_features = X_train_base.iloc[:5].apply(
#     lambda instance: extract_instance_code_features(
#         instance, early_train, code_table_train
#     ),
#     axis=1,
# )

# vstack(code_features)


def extract_features(X, early_df, code_table, scaler):
    # First extract performance features for each row
    # features = X.apply(lambda instance: extract_instance_features(instance, early_df), axis=1)
    # Then get code features
    # print(X)
    code_features = early_df.apply(
        lambda instance: extract_instance_code_features(instance, early_df, code_table),
        axis=1,
    )
    # print(code_features)
    code_features = vstack(code_features).toarray()
    tfidf_df = early_df[['SubjectID', 'AssignmentID', 'ProblemID']].copy()
    tfidf_df = pd.concat([tfidf_df, pd.DataFrame(code_features)], axis = 1)
    # Then one-hot encode the problem_id and append it
    # problem_ids = problem_encoder.transform(features[PS2.ProblemID].values.reshape(-1, 1)).toarray()
    # Then get rid of nominal features
    # features.drop([PS2.AssignmentID, PS2.ProblemID], axis=1, inplace=True)
    # Then scale the continuous features, fitting the scaler if this is training
    # if is_train:
    #     scaler.fit(features)
    # features = scaler.transform(features)

    # Return continuous and one-hot features together
    # return np.concatenate([features, code_features, problem_ids], axis=1)
    # return np.concatenate([features, code_features], axis=1)
    return tfidf_df


scaler = StandardScaler()
train_df = extract_features(X_train_base, early_train, code_table_train, scaler)
X_train = train_df.drop(['SubjectID', 'AssignmentID', 'ProblemID'], axis = 1).copy()

train_df_subject =  train_df.drop(['AssignmentID', 'ProblemID'], axis = 1).groupby(['SubjectID']).mean().reset_index()
cor_table = train_df_subject.drop(['SubjectID'], axis=1).corr()

# print(X_train.shape)
# X_train[:2,]


## Test

early_test = pd.read_csv(os.path.join(TEST_PATH, "early.csv"))
late_test = pd.read_csv(os.path.join(TEST_PATH, "late.csv"))

test_ps2 = ProgSnap2Dataset(os.path.join(TEST_PATH, "Data"))
code_table_test = get_code_table(test_ps2)

test_df = extract_features(late_test, early_test, code_table_test, scaler)
X_test = test_df.drop(['SubjectID', 'AssignmentID', 'ProblemID'], axis = 1).copy()
print(X_test.shape)

train_df.to_csv('data/ours/tfidf_All_train.csv', index=False)
test_df.to_csv('data/ours/tfidf_'+test_semester+'_test.csv', index=False)


# %%
########################################################
########################################################
# Aggregate at subject-level to get late vectors

# =============================================================================
# early_train = pd.read_csv('data/S19/Train/early.csv')
# late_train = pd.read_csv('data/S19/Train/late.csv')
# early_test = pd.read_csv('data/F19/Test/early.csv')
# late_test = pd.read_csv('data/F19/Test/late.csv')
# 
# tfidf_early_train = pd.read_csv('data/ours/tfidf_All_train.csv')
# tfidf_early_test = pd.read_csv('data/ours/tfidf_F19_test.csv')
# 
# =============================================================================

# %%
# =============================================================================
# train_agg = tfidf_early_train.drop(['AssignmentID', 'ProblemID'], axis=1).groupby('SubjectID').mean()
# 
# tfidf_late_train = late_train.drop('Label', axis=1).merge(train_agg, on='SubjectID')
# 
# 
# # %%
# test_agg = tfidf_early_test.drop(['AssignmentID', 'ProblemID'], axis=1).groupby('SubjectID').mean()
# 
# tfidf_late_test = late_test.merge(test_agg, on='SubjectID')
# 
# 
# =============================================================================
# %%
# =============================================================================
# tfidf_late_train.to_csv('data/ours/tfidf_late_train.csv', index=False)
# tfidf_late_test.to_csv('data/ours/tfidf_late_test.csv', index=False)
# 
# =============================================================================
# %%
