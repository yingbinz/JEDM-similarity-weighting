# %%
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, metrics
from sklearn.model_selection import cross_validate, GroupKFold, GridSearchCV, KFold

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb

from os import listdir
from os.path import isfile, join
# %%
onlyS19 = True
cv = GroupKFold(5)

comb_list = [['difficulty','prompt'],
    ['difficulty','code2vec'],
    ['prompt','code2vec'],
    ['order','difficulty','prompt'],
    ['order','difficulty','code2vec'],
    ['order','prompt','code2vec'],
    ['difficulty','prompt','code2vec']]
# =============================================================================
# f = weight_files[0]
# =============================================================================
# %% functions for evaluation
## Evaluate the Performance of the Model
def eva_performance(X, y, model):    
    predictions = model.predict_proba(X)[:,1]
    AUC = roc_auc_score(y, predictions)
    predictions = model.predict(X)
    acc = np.mean(y == predictions)
    macroF1 = f1_score(y,  predictions,  average='macro')
    return acc, AUC, macroF1
## Evaluate the CV Performance of the Model
def cv_performance(X_train, y_train, model, groups):
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=['accuracy', 'f1_macro', 'roc_auc'], n_jobs=-1, groups=groups)
    
    AUC = np.mean(cv_results["test_roc_auc"])
    acc = np.mean(cv_results["test_accuracy"])
    macroF1 = np.mean(cv_results["test_f1_macro"])
    return acc, AUC, macroF1
    
# %% define models
modrf = RandomForestClassifier(max_depth=5,
                               min_samples_split=10,
                               n_estimators=500, n_jobs=-1,
                               random_state=42)
modlasso = LogisticRegression(
                        C=1,
                        penalty='l1',
                        solver='liblinear',
                        tol=1e-6,
                        max_iter=int(1e6),
                        warm_start=True,
                        random_state=42)
# check standardized coef
# =============================================================================
# model.fit(X_train / np.std(X_train, 0), y_train)
# model.coef_
# =============================================================================
modxgb = xgb.XGBClassifier(learning_rate=0.01, max_depth=3,
                              subsample = 0.5,
                              objective = 'binary:logistic',
                              n_estimators = 500, random_state=42, verbosity = 0,
                              use_label_encoder=False)

# %% evaluation
df_cols = ['weighting',
            'features',
            'model',
            'train accuracy',
            'train AUC',
            'train macro F1',
            'test accuracy',
            'test AUC',
            'test macro F1']
performance = pd.DataFrame(columns=df_cols)

for comb in comb_list:
    if onlyS19:
        X_train = pd.read_csv('data/ours/additional_weighting_comb/S19_X_train_'+"_".join(comb)+'_cosine.csv')
        y_train = pd.read_csv('data/ours/S19_y_train_all.csv')['label'].values
        X_train_base = pd.read_csv('data/S19/Train/late.csv').drop('Label', axis=1)
       
        # remove some rows with missing values in training dataset
        notnull_index = X_train.loc[X_train.notnull().all(1),:].index
        X_train = X_train.loc[notnull_index,:].reset_index(drop=True)
        y_train = y_train[notnull_index]
        X_train_base = X_train_base.loc[notnull_index,:].reset_index(drop=True)
        
        # encode true-false label to 1-0 to avoid xgb warning
        y_train = y_train*1
        
        X_test = pd.read_csv('data/ours/additional_weighting_comb/S19_X_test_'+"_".join(comb)+'_cosine.csv')
        late_test = pd.read_csv('data/F19/Test/late.csv')
        y_test = late_test['Label'].values*1        
    else:
        X_train = pd.read_csv('data/ours/additional_weighting_comb/X_train_'+"_".join(comb)+'_cosine.csv')
        y_train = pd.read_csv('data/ours/y_train_all.csv')['label'].values
        X_train_base_S = pd.read_csv('data/S19/All/late.csv').drop('Label', axis=1)
        X_train_base_F = pd.read_csv('data/F19/Train/late.csv').drop('Label', axis=1)
        X_train_base = pd.concat([X_train_base_F, X_train_base_S], axis = 0, join = 'inner').reset_index(drop=True)
        
        # remove some rows with missing values in training dataset
        notnull_index = X_train.loc[X_train.notnull().all(1),:].index
        X_train = X_train.loc[notnull_index,:].reset_index(drop=True)
        y_train = y_train[notnull_index]
        X_train_base = X_train_base.loc[notnull_index,:].reset_index(drop=True)
        
        # encode true-false label to 1-0 to avoid xgb warning
        y_train = y_train*1
        
        X_test = pd.read_csv('data/ours/additional_weighting_comb/X_test_'+"_".join(comb)+'_cosine.csv')
        late_test = pd.read_csv('data/F19/Test/late.csv')
        y_test = late_test['Label'].values*1
        
    X_train = X_train.drop(['AssignmentID'], axis=1)
    groups = X_train_base['SubjectID']
    
    
    X_test =  X_test[X_train.columns]
    #X_train = X_train.drop(X_train.columns[X_train.columns.str.contains('tfidf')], axis=1) 
        
    drop_list = ['ALL']
    for drop in drop_list:
        if len(drop) == 2:
            f_train = X_train.drop(X_train.columns[X_train.columns.str.contains(drop[0])], axis=1).copy()
            f_train = f_train.drop(f_train.columns[f_train.columns.str.contains(drop[1])], axis=1)
            drop="tfidf and code2vec"
        else:
            f_train = X_train.drop(X_train.columns[X_train.columns.str.contains(drop)], axis=1).copy()
        f_test = X_test[f_train.columns].copy()
        for model in [modxgb, modrf, modlasso]:
            
            model.fit(f_train, y_train)
            train_performance = eva_performance(f_train, y_train, model)
            test_performance = eva_performance(f_test, y_test, model)
            temp_performance = ["_".join(comb)+'_cosine', "No"+drop, type(model).__name__]+list(train_performance)+list(test_performance)
            performance.loc[len(performance.index)] = temp_performance
            
performance['features'] = ["All" if x == "NoALL" else x for x in performance.features]
performance['features'] = [x.replace("No", "No ") for x in performance.features]

if onlyS19:
    performance.to_csv('result/track1_results_onlyS19train_additional_weighting_comb.csv', index=False)
else:
    performance.to_csv('result/track1_results_additional_weighting_comb.csv', index=False)