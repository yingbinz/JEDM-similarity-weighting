# %%
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GroupKFold


from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import xgboost as xgb

from os import listdir
from os.path import isfile, join
# %%
onlyS19 = True
cv = GroupKFold(5)

weight_path = "data/ours/weights/"
weight_files = [f for f in listdir(weight_path) if isfile(join(weight_path, f))]
if onlyS19:
    weight_files = [x for x in weight_files if ((("Sp19" in x) or ("prompts" in x)) & ("cosine" in x)) ]
else:
    weight_files = [x for x in weight_files if ((("all" in x) or ("prompts" in x)) & ("cosine" in x)) ]

weight_files = ["difficulty.csv"] + weight_files
weight_files = weight_files + ["".join(["order_", f]) for f in weight_files]
weight_files = ["no_weight.csv", "order.csv"] + weight_files
weight_files = weight_files + ['order_cosine_difficulty.csv']

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
            'num_features',
            'model',
            'train accuracy',
            'train AUC',
            'train macro F1',
            'test accuracy',
            'test AUC',
            'test macro F1']
performance = pd.DataFrame(columns=df_cols)
late_test = pd.read_csv('data/F19/Test/late.csv')
# =============================================================================
# predictions_df = late_test.copy()
# =============================================================================
for f in weight_files:
    f = f.replace("_Sp19", "")
    if onlyS19:
        X_train = pd.read_csv('data/ours/S19_X_train_'+f)
        y_train = pd.read_csv('data/ours/S19_y_train_all.csv')['label'].values
        X_train_base = pd.read_csv('data/S19/Train/late.csv').drop('Label', axis=1)
       
        # remove some rows with missing values in training dataset
        notnull_index = X_train.loc[X_train.notnull().all(1),:].index
        X_train = X_train.loc[notnull_index,:].reset_index(drop=True)
        y_train = y_train[notnull_index]
        X_train_base = X_train_base.loc[notnull_index,:].reset_index(drop=True)
        
        # encode true-false label to 1-0 to avoid xgb warning
        y_train = y_train*1
        
        X_test = pd.read_csv('data/ours/S19_X_test_'+f)
        y_test = late_test['Label'].values*1        
    else:
        X_train = pd.read_csv('data/ours/X_train_'+f)
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
        
        X_test = pd.read_csv('data/ours/X_test_'+f)
        y_test = late_test['Label'].values*1
        
    X_train = X_train.drop(['AssignmentID'], axis=1)
    groups = X_train_base['SubjectID']
    
    
    X_test =  X_test[X_train.columns]       
    


    for model in [modxgb, modrf, modlasso]:
        for num_feature in [5, 10, 20, 30, 40, 50, 60]:
            f_train = X_train.copy()
            f_test = X_test[f_train.columns].copy()
            
            sfm = SelectFromModel(model, threshold=-np.inf, max_features=num_feature)
            sfm.fit(f_train, y_train)
            f_train = sfm.transform(f_train)
            f_test = sfm.transform(f_test) 
            model.fit(f_train, y_train)
            
            train_performance = eva_performance(f_train, y_train, model)
            test_performance = eva_performance(f_test, y_test, model)
            temp_performance = [f.replace(".csv",""), num_feature, type(model).__name__]+list(train_performance)+list(test_performance)
            performance.loc[len(performance.index)] = temp_performance
            
if onlyS19:
    performance.to_csv('result/feature_selection_track1_results_onlyS19train.csv', index=False)
else:
    performance.to_csv('result/feature_selection_track1_results.csv', index=False)

# =============================================================================
#         model.fit(X_train, y_train)
#         predictions = model.predict_proba(X_test)[:,1]
#         predictions_df[f+str(model.__class__)] = predictions
 
# if onlyS19:
#     predictions_df.to_csv('result/prediction_track1_results_onlyS19train.csv', index=False)
# else:
#     predictions_df.to_csv('result/prediction_selection_track1_results.csv', index=False)
# =============================================================================
# %% which feature contribute large improvement in no weighting
auc_change_list = []
late_test = pd.read_csv('data/F19/Test/late.csv')
for onlyS19 in [True, False]:
    weight_files = [f for f in listdir(weight_path) if isfile(join(weight_path, f))]
    if onlyS19:
        weight_files = [x for x in weight_files if ((("Sp19" in x) or ("prompts" in x)) & ("cosine" in x)) ]
    else:
        weight_files = [x for x in weight_files if ((("all" in x) or ("prompts" in x)) & ("cosine" in x)) ]

    weight_files = ["difficulty.csv"] + weight_files
    weight_files = weight_files + ["".join(["order_", f]) for f in weight_files]
    weight_files = ["no_weight.csv", "order.csv"] + weight_files
    weight_files = weight_files + ['order_cosine_difficulty.csv']
    
    for f in ['no_weight.csv', 'difficulty.csv']:

        f = f.replace("_Sp19", "")
        if onlyS19:
            X_train = pd.read_csv('data/ours/S19_X_train_'+f)
            y_train = pd.read_csv('data/ours/S19_y_train_all.csv')['label'].values
            X_train_base = pd.read_csv('data/S19/Train/late.csv').drop('Label', axis=1)
           
            # remove some rows with missing values in training dataset
            notnull_index = X_train.loc[X_train.notnull().all(1),:].index
            X_train = X_train.loc[notnull_index,:].reset_index(drop=True)
            y_train = y_train[notnull_index]
            X_train_base = X_train_base.loc[notnull_index,:].reset_index(drop=True)
            
            # encode true-false label to 1-0 to avoid xgb warning
            y_train = y_train*1
            
            X_test = pd.read_csv('data/ours/S19_X_test_'+f)
            y_test = late_test['Label'].values*1        
        else:
            X_train = pd.read_csv('data/ours/X_train_'+f)
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
            
            X_test = pd.read_csv('data/ours/X_test_'+f)
            y_test = late_test['Label'].values*1
            
        X_train = X_train.drop(['AssignmentID'], axis=1)
        
        X_test =  X_test[X_train.columns]    
        
        
        first_fives = []
        for model in [modxgb, modrf]:
            mod_temp = model
            f_train = X_train.copy()
            f_test = X_test[f_train.columns].copy()
            
            model.fit(f_train, y_train)
            
            f_importances = pd.Series(model.feature_importances_, index=f_train.columns).sort_values(ascending=False)
            temp=[]
            print(mod_temp)
            for i in range(5,31):
                print (i)
                temp_train = f_train[f_importances.index[:i]]
                model = mod_temp
                model.fit(temp_train, y_train)
                
                train_performance = eva_performance(temp_train, y_train, model)
                test_performance = eva_performance(f_test[temp_train.columns], y_test, model)
                
                temp.append(train_performance+test_performance)
            temp = pd.DataFrame(temp, columns=['train accuracy',
                                                'train AUC',
                                                'train macro F1',
                                                'test accuracy',
                                                'test AUC',
                                                'test macro F1'])
            temp['new_feature'] = f_importances.index.values[4:30]
            temp['model'] = type(model).__name__
            temp['weight'] = f
            temp['onlyS19'] =onlyS19
            first_fives.append(f_importances.index.values[:4])
            auc_change_list.append(temp)
            
        mod_temp = modlasso
        f_train = X_train.copy()
        f_test = X_test[f_train.columns].copy()   
        selected_featurs = [] 
        last_selected = []
        temp=[]
        print(mod_temp)
        for num_feature in range(5,31):
            print (num_feature)
            model = mod_temp
            
            sfm = SelectFromModel(model, threshold=-np.inf, max_features=num_feature)
            sfm.fit(f_train, y_train)
            selected_f = sfm.get_feature_names_out()
            temp_train = f_train[sfm.get_feature_names_out()].copy()
            model.fit(temp_train, y_train)
            
            train_performance = eva_performance(temp_train, y_train, model)
            test_performance = eva_performance(f_test[temp_train.columns], y_test, model)
            
            temp.append(train_performance+test_performance)
            for x in selected_f:
                if x not in last_selected:
                    selected_featurs.append(x)
                    break
            last_selected = selected_f.copy()
            
        temp = pd.DataFrame(temp, columns=['train accuracy',
                                            'train AUC',
                                            'train macro F1',
                                            'test accuracy',
                                            'test AUC',
                                            'test macro F1'])
        temp['new_feature'] = selected_featurs
        temp['model'] = type(model).__name__
        temp['weight'] = f
        temp['onlyS19'] =onlyS19
        
        auc_change_list.append(temp)
              

pd.concat(auc_change_list).to_csv('result/auc_change.csv', index=False)

#%% remove percSubjectStruggleA492minusA439 and percCorrectEventuallyA492minusA439
df_cols = ['weighting',
            'phase',
            'model',
            'train accuracy',
            'train AUC',
            'train macro F1',
            'test accuracy',
            'test AUC',
            'test macro F1']
performance = pd.DataFrame(columns=df_cols)
late_test = pd.read_csv('data/F19/Test/late.csv')
for onlyS19 in [True, False]:
    weight_files = [f for f in listdir(weight_path) if isfile(join(weight_path, f))]
    if onlyS19:
        weight_files = [x for x in weight_files if ((("Sp19" in x) or ("prompts" in x)) & ("cosine" in x)) ]
    else:
        weight_files = [x for x in weight_files if ((("all" in x) or ("prompts" in x)) & ("cosine" in x)) ]

    weight_files = ["difficulty.csv"] + weight_files
    weight_files = weight_files + ["".join(["order_", f]) for f in weight_files]
    weight_files = ["no_weight.csv", "order.csv"] + weight_files
    weight_files = weight_files + ['order_cosine_difficulty.csv']
    for f in weight_files[:5]:
        f = f.replace("_Sp19", "")
        if onlyS19:
            X_train = pd.read_csv('data/ours/S19_X_train_'+f)
            y_train = pd.read_csv('data/ours/S19_y_train_all.csv')['label'].values
            X_train_base = pd.read_csv('data/S19/Train/late.csv').drop('Label', axis=1)
           
            # remove some rows with missing values in training dataset
            notnull_index = X_train.loc[X_train.notnull().all(1),:].index
            X_train = X_train.loc[notnull_index,:].reset_index(drop=True)
            y_train = y_train[notnull_index]
            X_train_base = X_train_base.loc[notnull_index,:].reset_index(drop=True)
            
            # encode true-false label to 1-0 to avoid xgb warning
            y_train = y_train*1
            
            X_test = pd.read_csv('data/ours/S19_X_test_'+f)
            y_test = late_test['Label'].values*1        
        else:
            X_train = pd.read_csv('data/ours/X_train_'+f)
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
            
            X_test = pd.read_csv('data/ours/X_test_'+f)
            y_test = late_test['Label'].values*1
            
        X_train = X_train.drop(['AssignmentID'], axis=1)
        groups = X_train_base['SubjectID']
        
        
        X_test =  X_test[X_train.columns]      
        for model in [modxgb, modrf, modlasso]:
            f_train = X_train.copy()
            f_train = f_train.drop(['percSubjectStruggleA492minusA439', 'percCorrectEventuallyA492minusA439'],axis=1)
            f_test = X_test[f_train.columns].copy()
            model.fit(f_train, y_train)
            
            train_performance = eva_performance(f_train, y_train, model)
            test_performance = eva_performance(f_test, y_test, model)
            temp_performance = [f.replace(".csv",""), onlyS19, type(model).__name__]+list(train_performance)+list(test_performance)
            performance.loc[len(performance.index)] = temp_performance   
            
performance.to_csv('result/track1_results_no_A492minusA439.csv', index=False)