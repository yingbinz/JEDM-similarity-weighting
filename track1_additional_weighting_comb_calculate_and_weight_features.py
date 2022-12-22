# %% load library
#from operator import index
import pandas as pd
import numpy as np
from ast import literal_eval
import Levenshtein
from sklearn.decomposition import PCA
### use all data

#%% load data
####### if only use S19 train data, i.e., phase 1 train data
onlyS19 = True

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
if onlyS19:
    early = pd.concat([early_train, early_test], axis = 0, join = 'inner').reset_index(drop=True)
    SubjectIDs = early['SubjectID'].values
    ProblemIDs = early['ProblemID'].values
    early_train = pd.read_csv('data/S19/Train/early.csv')
    late_train = pd.read_csv('data/S19/Train/late.csv')
    raw_data_early_train = pd.read_csv('data/ours/raw_train_early_S19.csv')
    raw_data_late_train = pd.read_csv('data/ours/raw_train_late_S19.csv')
    
    X_train_base = late_train.copy().drop('Label', axis=1)
    y_train = late_train['Label'].values
    
del early_train_S, late_train_S, raw_data_early_train_F, raw_data_early_train_S, \
    raw_data_late_train_F, raw_data_late_train_S 
# %% late 20 problems' features
# Percentage of students struggling with that problem
late_problem_df = late_train.groupby(["ProblemID"]).agg(percProblemStruggle = ("Label", lambda x: 1-np.mean(x))).reset_index()
# Percentage of students with syntax error on that problem
temp = raw_data_late_train.groupby(['SubjectID', "ProblemID"]).any()['CompileErrorsDetail'].groupby(['ProblemID']).mean().reset_index(name="percProblemSyntaxError")
late_problem_df = late_problem_df.merge(temp, on = 'ProblemID')
# Percentage of students with semantic error on that problem
temp = raw_data_late_train[raw_data_late_train['CompileErrorsDetail'].isna()].groupby(['SubjectID', "ProblemID"])['Score']\
    .apply(lambda x: (x < 1).any()).groupby(['ProblemID']).mean().reset_index(name = 'percProblemSemanticError')
late_problem_df = late_problem_df.merge(temp, on = 'ProblemID')

# %% early 30 problems' features
early = pd.concat([early_train, early_test], axis = 0, join = 'inner').reset_index(drop=True)
raw_data_early = pd.concat([raw_data_early_train, raw_data_early_test], axis = 0, join = 'inner').reset_index(drop=True)

# Percentage of students struggling with that problem
early_problem_df = early.groupby(["ProblemID"]).agg(percProblemStruggle = ("Label", lambda x: 1-np.mean(x))).reset_index()
# Percentage of students with syntax error on that problem
temp = raw_data_early.groupby(['SubjectID', "ProblemID"]).any()['CompileErrorsDetail'].groupby(['ProblemID']).mean().reset_index(name="percProblemSyntaxError")
early_problem_df = early_problem_df.merge(temp, on = 'ProblemID')
# Percentage of students with semantic error on that problem
temp = raw_data_early[raw_data_early['CompileErrorsDetail'].isna()].groupby(['SubjectID', "ProblemID"])['Score']\
    .apply(lambda x: (x < 1).any()).groupby(['ProblemID']).mean().reset_index(name = 'percProblemSemanticError')
early_problem_df = early_problem_df.merge(temp, on = 'ProblemID')

# %% get correlation matrix between early problems and later problems
early_problems = list(set(early_train.ProblemID))
later_problems = list(set(late_train.ProblemID))

both_train = pd.concat([early_train[["SubjectID", "ProblemID", "Label"]],
                       late_train[["SubjectID", "ProblemID", "Label"]]])

both_train = both_train.pivot(index = 'SubjectID', columns = 'ProblemID', values = 'Label')

both_train = both_train*1

lor_df = pd.DataFrame([[0]*len(later_problems)]*len(early_problems),
                      columns = later_problems ,
                      index = early_problems)

for i in early_problems:
    for j in later_problems:
        tab = pd.crosstab(both_train.loc[:,i], both_train.loc[:,j])
              
        # adding 0.5 to reduce bias
        a = tab.iloc[0,0]+.5
        b = tab.iloc[0,1]+.5
        c = tab.iloc[1,0]+.5
        d = tab.iloc[1,1]+.5  
        lor = np.log(a*d/b/c)
        #sd = np.sqrt(1/a+1/b+1/c+1/d)
        lor_df.loc[i,j] = lor
        
lor_df[lor_df < 0] = np.nan
lor_df.fillna(value = lor_df.min(), inplace = True)

lor_df.reset_index(inplace=True)
lor_df.rename(columns = {'index': 'ProblemID'}, inplace=True)


# %% build functions for get_early_subject_df 
## Compute subject level features in here to avoid compute features for the same 
## subject multiple times in the extract_instance_features() and extract_features()
def get_last_sub_info(raw_data_early):
    raw_data_early = raw_data_early.copy()
    raw_data_early['ServerTimestamp'] = raw_data_early['ServerTimestamp'].apply(lambda x: pd.Timestamp(x, tz = "US/Eastern"))
    raw_data_early.sort_values(by = ['SubjectID', 'ProblemID', 'ServerTimestamp'], inplace = True)
    last_row = raw_data_early.iloc[:-1][['SubjectID', 'ProblemID', 'ServerTimestamp',
                             'Score','CompileErrors', 'num_of_syntax_errors', 'Code']]
    first_row = pd.DataFrame(raw_data_early.iloc[-1][['SubjectID', 'ProblemID', 'ServerTimestamp',
                             'Score','CompileErrors', 'num_of_syntax_errors', 'Code']])
    last_row = pd.concat([first_row.T, last_row], axis=0)
    last_row = last_row.set_index(raw_data_early.index)
    last_row = last_row.rename(columns={'SubjectID':'last_SubjectID', 
                                        'ProblemID':'last_ProblemID', 
                                        'ServerTimestamp':'last_ServerTimestamp', 
                                        'Score':'last_Score', 
                                        'CompileErrors':'last_CompileErrors',
                                        'num_of_syntax_errors':'last_num_of_syntax_errors',
                                        'Code':'last_Code'})
    ## the first row doesn't have last row information, so assign nan to the first row
    last_row.iloc[0] = np.nan
    raw_data_early = pd.concat([raw_data_early, last_row], axis = 1)
    raw_data_early.reset_index(drop = True, inplace = True)
    
    return raw_data_early

### Change in syntax errors
def get_error_change_series(series):
    ##If S1 had syntax errors, were any of these old syntax errors fixed in S2?
    FixingSyntaxError = -9
    ##Were there any new syntax errors in S2?
    MakingNewSyntaxError = -9
    ##If S1 had syntax errors, how many of these old syntax errors were fixed in S2?
    decrease_in_syntax_error = -9    
    ## If S1 had no syntax error, what's the change of test score
    ChangeTestScores = -9
    ## 
    if (series.ProblemID == series.last_ProblemID) & (
            series.SubjectID == series.last_SubjectID):
        # if information about the syntax errors of the last row or the current row is missing, we cannot calculate the syntax error                     
        if (series.last_num_of_syntax_errors == 0):

            FixingSyntaxError = np.nan
            MakingNewSyntaxError = np.nan
            decrease_in_syntax_error = np.nan
            
            ChangeTestScores = series.Score - series.last_Score
        else:
            ChangeTestScores = np.nan
            if series.num_of_syntax_errors == 0:
                FixingSyntaxError = 1
                MakingNewSyntaxError = 0
                decrease_in_syntax_error = series.last_num_of_syntax_errors
            else:
                syntax_errors = literal_eval(series.CompileErrors)
                syntax_errors.sort()            
                
                last_syntax_errors = literal_eval(series.last_CompileErrors)
                last_syntax_errors.sort()
                
                decrease_in_syntax_error = sum([error not in syntax_errors for error in last_syntax_errors])                                              
                # if the current row has syntax errors no less than the last row, students might fail to fix the syntax errors in the last row
                # but it might also be because the number of fixed syntax errors were no more than the number of new syntax errors 
                # thus, simply comparing the number of syntax errors between submissions is not sufficient for judging whether a student fix old syntax errors
                if decrease_in_syntax_error == 0:
                    FixingSyntaxError = 0
                else:
                    FixingSyntaxError = 1
                                   
                if all(error in last_syntax_errors for error in syntax_errors):
                    MakingNewSyntaxError = 0
                else:
                    MakingNewSyntaxError = 1
                                                      
    else:
        FixingSyntaxError = np.nan
        MakingNewSyntaxError = np.nan
        decrease_in_syntax_error = np.nan
        ChangeTestScores = np.nan
    return FixingSyntaxError, MakingNewSyntaxError, decrease_in_syntax_error, ChangeTestScores

def get_error_change_df(raw_data_early):
    raw_data_early = raw_data_early.copy()
    raw_data_early["num_of_syntax_errors"] = raw_data_early.apply(lambda row: len(literal_eval(row["CompileErrors"])) if type(row["CompileErrors"]) == str else 0, axis=1)

    raw_data_early = get_last_sub_info(raw_data_early)
    error_change = raw_data_early.apply(get_error_change_series, axis=1)
    
    raw_data_early = raw_data_early.assign(FixingSyntaxError=np.nan, 
                                    MakingNewSyntaxError=np.nan, 
                                    decrease_in_syntax_error=np.nan,
                                    ChangeTestScores=np.nan)
    raw_data_early[['FixingSyntaxError', 
                          'MakingNewSyntaxError', 
                          'decrease_in_syntax_error', 
                          'ChangeTestScores']] = [x for x in error_change]
    return raw_data_early

def get_code_state(instance):
    code_state_given = np.nan
    code_state_target = np.nan
    if instance.num_of_syntax_errors > 0:
        code_state_target = "syntax_error"               
    elif instance.Score == 1:
        code_state_target = "correct"
    else:
        code_state_target = "test_error"
    
    if (instance.SubjectID == instance.last_SubjectID) & (instance.last_ProblemID == instance.ProblemID):
        if instance.last_num_of_syntax_errors > 0:
            code_state_given = "syntax_error"               
        elif instance.last_Score == 1:
            code_state_given = "correct"
        else:
            code_state_given = "test_error" 
    
    return code_state_given, code_state_target
    
def get_code_state_transition(df):# input df should be processed by get_error_change_df        
    df[['code_state_given', 'code_state_target']] = [x for x in df.apply(lambda instance:get_code_state(instance), axis = 1)]     
    
    df['transition'] = [".".join([x, y]) if pd.notna(x) & pd.notna(y) & ('correct' != x) else np.nan \
                          for x, y in zip(df['code_state_given'], df['code_state_target'])]
    
    df = df[pd.notna(df.transition)].groupby(['SubjectID']).agg(
      SE_SE = ('transition', lambda x: np.sum(x == 'syntax_error.syntax_error')),
      SE_TE = ('transition', lambda x: np.sum(x == 'syntax_error.test_error')),
      SE_correct = ('transition', lambda x: np.sum(x == 'syntax_error.correct')),
      TE_SE = ('transition', lambda x: np.sum(x == 'test_error.syntax_error')),
      TE_TE = ('transition', lambda x: np.sum(x == 'test_error.test_error')),
      TE_correct = ('transition', lambda x: np.sum(x == 'test_error.correct')),
      given_SE = ('code_state_given', lambda x: np.sum(x == 'syntax_error')),
      given_TE = ('code_state_given', lambda x: np.sum(x == 'test_error')) ,  
      target_SE = ('code_state_target', lambda x: np.sum(x == 'syntax_error')),   
      target_TE = ('code_state_target', lambda x: np.sum(x == 'test_error'))  ,    
      target_correct = ('code_state_target', lambda x: np.sum(x == 'correct'))  ,
      total = ('SubjectID', len)       
      ).reset_index() 
    
    for given in ['SE', 'TE']:
        for target in ['SE', 'TE', 'correct']:
            a = df["_".join([given, target])]
            b = df["_".join(['given', given])] - a
            c = df["_".join(['target', target])] - a
            d = df['total'] - a - b - c
            df["_".join([given, target])] = np.log((a+0.5)*(b+0.5)/(c+0.5)/(d+0.5))
    return(df[['SubjectID','SE_SE', 'SE_TE', 'SE_correct', 'TE_SE', 'TE_TE',
       'TE_correct']])

def get_time_feature(raw_data_early, early, use_weight = False, median_center=False):
    # sort the data so that the next step can be done
    raw_data_early = raw_data_early.copy()
    raw_data_early = get_error_change_df(raw_data_early)
    
    # state transition
    raw_data_early[['code_state_given', 'code_state_target']] = [x for x in raw_data_early.apply(lambda instance:get_code_state(instance), axis = 1)]     
 
    raw_data_early.sort_values(by = ['SubjectID', 'ServerTimestamp'], inplace = True)
    
    raw_data_early = raw_data_early.drop(['last_SubjectID',
                  'last_ProblemID',
                  'last_ServerTimestamp'], axis = 1)
    # put the information in the last row to current row
    ## extract information in the last row 
    last_row = raw_data_early.iloc[:-1][['SubjectID', 'ServerTimestamp', 'ProblemID']]
    first_row = pd.DataFrame(raw_data_early.iloc[-1][['SubjectID', 'ServerTimestamp', 'ProblemID']])
    last_row = pd.concat([first_row.T, last_row], axis=0)
    last_row = last_row.set_index(raw_data_early.index)
    last_row = last_row.rename(columns={'SubjectID':'last_SubjectID',
                                        'ProblemID': 'last_ProblemID',
                             'ServerTimestamp':'last_ServerTimestamp'})
    ## the first row doesn't have last row information
    last_row.iloc[0] = np.nan
    ## add the last row information to raw_data_early
    raw_data_early = pd.concat([raw_data_early, last_row], axis = 1)
    # calculate the time interval between submissions    
    raw_data_early['time_diff'] = raw_data_early.ServerTimestamp - raw_data_early.last_ServerTimestamp
    raw_data_early['time_diff'] = [x.total_seconds() for x in raw_data_early['time_diff']]      
    # if submissions are on different problems, add 5s for switching
    raw_data_early.loc[(raw_data_early.ProblemID != raw_data_early.last_ProblemID) & (raw_data_early.SubjectID == raw_data_early.last_SubjectID), 'time_diff'] = \
        raw_data_early.loc[(raw_data_early.ProblemID != raw_data_early.last_ProblemID) & (raw_data_early.SubjectID == raw_data_early.last_SubjectID), 'time_diff'] - 5
    ## if minus 5 leads to negative value, add 5
    raw_data_early.loc[raw_data_early.time_diff < 0, 'time_diff'] = raw_data_early.loc[raw_data_early.time_diff < 0, 'time_diff'] + 5       
    ## code time differences of different students' submissions as missing    
    raw_data_early.loc[(raw_data_early.SubjectID != raw_data_early.last_SubjectID), 'time_diff'] = np.nan    
    
    raw_data_early.reset_index(inplace = True, drop = True)
    raw_data_early = raw_data_early.drop(['last_SubjectID',
                  'last_ProblemID',
                  'last_ServerTimestamp'], axis = 1)

    raw_data_early.loc[raw_data_early.time_diff > 300, 'time_diff'] = np.nan
    
    temp = raw_data_early.groupby(['SubjectID', "ProblemID"]).agg(
        syntax_error = ('code_state_given', lambda x: np.sum(x == 'syntax_error')),
        test_error = ('code_state_given', lambda x: np.sum(x == 'test_error')),
        quick_submission = ('time_diff', lambda x: np.any(x < 15))             
        )
    temp = temp.merge(early, how = 'left', on = ['SubjectID', 'ProblemID'])
    temp['percSubjectStateSyntaxError'] = temp['syntax_error'] / temp['Attempts']
    temp['percSubjectStateTestError'] = temp['test_error'] / temp['Attempts']
       
    if use_weight:
        temp = temp.assign(percSubjectStateSyntaxError = temp['percSubjectStateSyntaxError']*temp['weight'],
                           percSubjectStateTestError = temp['percSubjectStateTestError']*temp['weight']
                           )  
        
    temp = temp.groupby(['SubjectID']).agg(
        percSubjectStateSyntaxError = ('percSubjectStateSyntaxError', np.mean),
        percSubjectStateTestError = ('percSubjectStateTestError', np.mean),
        # the percent of problems that a student made a least one quick submission
        percSubjectQuickSubmission = ('quick_submission', np.mean)
        )    
    return temp

def get_struggle_change(early):
    temp = early.copy()
    early = early.copy()
    # assignment 439, 487, and 492 are released in order. each contains 10 problems.
    early = early.groupby(["SubjectID", 'AssignmentID']).agg(percSubjectStruggle = ("Label", lambda x: 1-np.mean(x))).reset_index()
    early = early.pivot(index = 'SubjectID', columns = 'AssignmentID', values = 'percSubjectStruggle').reset_index()
    early['percSubjectStruggleA492minusA439'] = early[492.0] - early[439.0]
    #early['percSubjectStruggleA492minusA487'] = early[492.0] - early[487.0]
    early.loc[early["percSubjectStruggleA492minusA439"].isna(),"percSubjectStruggleA492minusA439"]=np.nanmedian(early["percSubjectStruggleA492minusA439"])
    #early.loc[early["percSubjectStruggleA492minusA487"].isna(),"percSubjectStruggleA492minusA487"]=np.nanmedian(early["percSubjectStruggleA492minusA487"])
    output = early.copy()
    
    early = temp
    early = early.groupby(["SubjectID", 'AssignmentID']).agg(percCorrectEventually = ("CorrectEventually", lambda x: 1-np.mean(x))).reset_index()
    early = early.pivot(index = 'SubjectID', columns = 'AssignmentID', values = 'percCorrectEventually').reset_index()
    early['percCorrectEventuallyA492minusA439'] = early[492.0] - early[439.0]
   # early['percCorrectEventuallyA492minusA487'] = early[492.0] - early[487.0]
    early.loc[early["percCorrectEventuallyA492minusA439"].isna(),"percCorrectEventuallyA492minusA439"]=np.nanmedian(early["percCorrectEventuallyA492minusA439"])
    #early.loc[early["percCorrectEventuallyA492minusA487"].isna(),"percCorrectEventuallyA492minusA487"]=np.nanmedian(early["percCorrectEventuallyA492minusA487"])
    
    output = output.merge(early, how = "left", on = "SubjectID")
    return output[['SubjectID', 'percSubjectStruggleA492minusA439', 'percCorrectEventuallyA492minusA439']]

def get_num_lines_adm(series):
    c1 = series['last_Code']
    c2 = series['Code']
    if pd.isna(c1) | pd.isna(c2):
        return np.nan
    c1 = c1.splitlines()
    c2 = c2.splitlines()
    # removing the leading and ending whitespaces
    # so that adding/deleting whitespaces is ignored
    for n, l in enumerate(c1):
        c1[n] = l.strip()
    for n, l in enumerate(c2):
        c2[n] = l.strip()
    c1_disappear = []
    for x in c1:
        if x not in c2:
            c1_disappear.append(x)
    c2_new = []
    for x in c2:
        if x not in c1:
            c2_new.append(x)
    # find how many in c1_disappear are modified rather than deleted
    # if edit distance / the length of line <= 0.5, regard the line as modified
    num_line_modified=0 
    for l1 in c1_disappear:
        # ignore blank lines
        if len(l1)==0:
            num_line_modified += 1
            continue
        for l2 in c2_new:            
            ratio = Levenshtein.distance(l1, l2)/len(l1)
            if ratio <= 0.5:                
                num_line_modified += 1
                break
    num_lines_adm = len(c1_disappear)+len(c2_new)-num_line_modified
    return num_lines_adm

def get_all_lines_adm(raw_data_early):
    temp = raw_data_early.copy()
    
    # put last row code to the current row
    temp = temp.assign(last_SubjectID = np.nan,
                 last_ProblemID = np.nan,
                 last_Code = np.nan)
    
    temp.sort_values(by = ['SubjectID', 'ProblemID', 'ServerTimestamp'], inplace = True)
    temp.reset_index(drop = True, inplace = True)
    ## convert the series in the right to list so that the assign statement would not assign values based on index
    temp.loc[1:len(temp),"last_SubjectID"] = temp.loc[0:(len(temp) - 2),"SubjectID"].tolist()
    temp.loc[1:len(temp),"last_ProblemID"] = temp.loc[0:(len(temp) - 2),"ProblemID"].tolist()
    temp.loc[1:len(temp),"last_Code"] = temp.loc[0:(len(temp) - 2),"Code"].tolist()
    temp.loc[(temp.SubjectID != temp.last_SubjectID) | (temp.ProblemID != temp.last_ProblemID), "last_Code"] = np.nan
    
    # compute the number of lines added, deleted, and modified
    
    temp['num_lines_adm'] = temp[['last_Code', 'Code']].apply(lambda x: get_num_lines_adm(x), axis=1)
    temp = temp.groupby(['SubjectID','ProblemID']).\
        agg(num_lines_adm = ('num_lines_adm', lambda x: np.mean(x))).\
        groupby(['SubjectID']).\
        agg(avgNumLinesADM = ('num_lines_adm', lambda x: np.mean(x))).reset_index() 
    return temp

# %% weight feature 

def weight_feature(weight_matrix, early, raw_data_early, use_order_weight = False, weight_vector = False, onlyS19=False): 
    early = early.copy()
    raw_data_early = raw_data_early.copy()
    num_similiarity_weight = len(weight_matrix)
    if num_similiarity_weight != 30:
        temp = weight_matrix.copy()
        weight_matrix = weight_matrix[0]
        weight_matrix[later_problems] = weight_matrix[later_problems] / weight_matrix[later_problems].sum()
        for w in temp[1:]:
            w[later_problems] = w[later_problems] / w[later_problems].sum() 
            weight_matrix.loc[:,later_problems] = weight_matrix[later_problems] + w[later_problems]
    else:        
        num_similiarity_weight = 1
    # if all weights in weight_matrix are the same, it means not using weight
    not_use_similiarity_weight = np.all(weight_matrix.iloc[:,1:]==1)
    early['Label'] = early['Label'] * 1
    early['CorrectEventually'] = early['CorrectEventually'] * 1
    
    #### basic and error-related feature
    ### compute feature at the (early) problem level  
    # CorrectFirstTry
    early['CorrectFirstTry'] = (early['Attempts'] == 1) & (early['CorrectEventually'] == 1)
    early['CorrectFirstTry'] = early['CorrectFirstTry'] * 1
    
    #CompileErrorsDetail
    temp_early = raw_data_early.groupby(['SubjectID', "ProblemID"]).any()['CompileErrorsDetail'].reset_index()   
    early = early.merge(temp_early, on = ["SubjectID", "ProblemID"])   
    early['CompileErrorsDetail'] = early['CompileErrorsDetail'] * 1
    
    #Score
    temp_early = raw_data_early[raw_data_early['CompileErrorsDetail'].isna()].groupby(
        ['SubjectID', "ProblemID"])['Score'].apply(lambda x: (x < 1).any()).reset_index() 
    early = early.merge(temp_early, on = ["SubjectID", "ProblemID"], how="left")   
    early['Score'] = early['Score'] * 1
    
    ids = list(set(early.SubjectID))
    # num_syntax_errors
    for Subject in ids:
        for Problem in set(raw_data_early.loc[raw_data_early.SubjectID == Subject].ProblemID):
            individual = raw_data_early.loc[(raw_data_early.SubjectID == Subject) & (raw_data_early.ProblemID == Problem)]
            errors = []
            for error in individual.CompileErrors:
                if type(error) == str:
                    errors = errors + literal_eval(error)
            if errors:
                num_errors = len(set(errors))
            else:
                num_errors=0
            early.loc[(early.SubjectID == Subject) & (early.ProblemID==Problem), 'num_syntax_errors'] = num_errors
    # avgSubjectUniqueTestScore
    temp_early = raw_data_early.groupby(['SubjectID', 'ProblemID']).\
            agg(avgSubjectUniqueTestScore = ('Score', lambda x: len(set(x)))).reset_index() 
    early = early.merge(temp_early, on = ["SubjectID", "ProblemID"], how="left")           
    
    
    ##### weight features    
    used_cols = ['Label', 'CorrectEventually', 'CorrectFirstTry', 'Attempts',
                      'CompileErrorsDetail', 'Score', 'num_syntax_errors',
                      'avgSubjectUniqueTestScore']
    
    if use_order_weight:
        order_weight = raw_data_early.groupby(['SubjectID', "ProblemID"]).agg(
            probStartTime = ('ServerTimestamp', lambda x: np.min(x)))
        order_weight['order_weight'] = 1
        order_weight.sort_values(by = ['SubjectID', 'probStartTime'], inplace = True)
        order_weight.reset_index(inplace = True)
        order_weight['order_weight'] = order_weight[['SubjectID', "order_weight"]].groupby(['SubjectID']).transform(
            lambda x: np.cumsum(x))
        order_weight['order_weight'] = order_weight[['SubjectID', "order_weight"]].groupby(['SubjectID']).transform(
            lambda x: x/np.sum(x))
        early = early.merge(order_weight, how = 'left', on = ['SubjectID', 'ProblemID'])

        
    early = early.merge(weight_matrix, how = 'left', on = ['ProblemID'])    
    late_problem_subject = pd.DataFrame(columns=['SubjectID', 'ProblemID']+used_cols)
    for Subject in ids:
        early_sub = early.loc[early.SubjectID == Subject,:].copy()
        score = early_sub.loc[pd.notna(early_sub['Score']),:].copy()
        
        # normalize the log odds ratio to make it suitable as weight
        early_sub[later_problems] = early_sub[later_problems] / early_sub[later_problems].sum()
        if use_order_weight:
            # if not combine order and similiarity weight
            # replace similiarity weight with order weight
            if not_use_similiarity_weight:
                for lp in later_problems:
                    early_sub[lp] = early_sub['order_weight'].copy()
            # if combine order and similiarity weight, take their means as the final weight        
            else:
                for lp in later_problems:
                    early_sub[lp] = (early_sub[lp] + early_sub['order_weight'])/(num_similiarity_weight+1)
        # compute the weighted percSubjectStruggleWeightedCorr
        early_sub = np.matmul(np.asarray(early_sub[later_problems].transpose()),                              
                              early_sub[used_cols])
        # normalize the log odds ratio to make it suitable as weight
        score[later_problems] = score[later_problems] / score[later_problems].sum()
        if use_order_weight:
            # if not combine order and similiarity weight
            # replace similiarity weight with order weight
            if not_use_similiarity_weight:
                for lp in later_problems:
                    score[lp] = score['order_weight'].copy()
            # if combine order and similiarity weight, take their means as the final weight        
            else:
                for lp in later_problems:
                    score[lp] = (score[lp] + score['order_weight'])/(num_similiarity_weight+1)
        # compute the weighted percSubjectStruggleWeightedCorr
        score = np.matmul(np.asarray(score[later_problems].transpose()),                              
                              score['Score'])
        
        early_sub['SubjectID'] = Subject
        early_sub['ProblemID'] = later_problems
        early_sub['Score'] = score
        late_problem_subject = pd.concat([late_problem_subject, early_sub], axis = 0, ignore_index = True)
    # drop the first row
    late_problem_subject['Label'] = 1 - late_problem_subject['Label'] 
    late_problem_subject.columns = ['SubjectID', 'ProblemID', 
                    'percSubjectStruggle', 'percCorrectEventually', 
                    'percCorrectFirstTry', 'avgAttempts',
                    'percSubjectSyntaxErrors', 'percSubjectSemanticErrors', 
                    'avgSubjectNumSyntaxErrors',
                    'avgSubjectUniqueTestScore']     

    #### debugging features
    ### compute feature at the submission level  
    raw_data_early = get_error_change_df(raw_data_early)
    
    # state transition
    raw_data_early[['code_state_given', 'code_state_target']] = [x for x in raw_data_early.apply(lambda instance:get_code_state(instance), axis = 1)]     
    raw_data_early['transition'] = [".".join([x, y]) if pd.notna(x) & pd.notna(y) & ('correct' != x) else np.nan \
                          for x, y in zip(raw_data_early['code_state_given'], raw_data_early['code_state_target'])]
    raw_data_early = raw_data_early.assign(
      SE_SE = raw_data_early['transition']== 'syntax_error.syntax_error',
      SE_TE = raw_data_early['transition']== 'syntax_error.test_error',
      SE_correct = raw_data_early['transition']== 'syntax_error.correct',
      TE_SE = raw_data_early['transition']== 'test_error.syntax_error',
      TE_TE = raw_data_early['transition']== 'test_error.test_error',
      TE_correct = raw_data_early['transition']== 'test_error.correct',
      given_SE = raw_data_early['code_state_given']== 'syntax_error',
      given_TE = raw_data_early['code_state_given']== 'test_error',  
      target_SE = raw_data_early['code_state_target']== 'syntax_error',   
      target_TE = raw_data_early['code_state_target']== 'test_error',    
      target_correct = raw_data_early['code_state_target']== 'correct',
      total = 1
      )
    
    raw_data_early['IncreasedTestScore'] = raw_data_early['ChangeTestScores'] > 0
    
    ##### weight features
    used_cols = ['FixingSyntaxError', 'MakingNewSyntaxError',
                   'ChangeTestScores', 'IncreasedTestScore',
                   'SE_SE', 'SE_TE', 'SE_correct',
                   'TE_SE', 'TE_TE', 'TE_correct', 
                   'given_SE', 'given_TE', 
                   'target_SE', 'target_TE', 'target_correct', 
                   'total']
    raw_data_early[used_cols] = raw_data_early[used_cols]*1
    
    if use_order_weight:
        raw_data_early = raw_data_early.merge(order_weight, how = 'left', on = ['SubjectID', 'ProblemID'])
        
    raw_data_early= raw_data_early.merge(weight_matrix, how = "left", on = ["ProblemID"])
    debugging = pd.DataFrame(columns=['SubjectID', 'ProblemID']+used_cols+['npairs'])
    for Subject in ids:
        early_sub = raw_data_early.loc[raw_data_early.SubjectID == Subject,:].copy()
        
        ### transitions
        # normalize the log odds ratio to make it suitable as weight
        transition_col = ['SE_SE', 'SE_TE', 'SE_correct',
                                       'TE_SE', 'TE_TE', 'TE_correct', 
                                       'given_SE', 'given_TE', 
                                       'target_SE', 'target_TE', 'target_correct', 
                                       'total']
        transition = early_sub.loc[pd.notna(early_sub['transition']),:].copy()
        nrows = transition.shape[0]
        transition[later_problems] = transition[later_problems] / transition[later_problems].sum()
        if use_order_weight:
            transition['order_weight'] = transition['order_weight'] / transition['order_weight'].sum()
            if not_use_similiarity_weight:
                for lp in later_problems:
                    transition[lp] = transition['order_weight'].copy()
            # if combine order and similiarity weight, take their means as the final weight        
            else:
                for lp in later_problems:
                    transition[lp] = (transition[lp] + transition['order_weight'])/(num_similiarity_weight+1)    
        transition = np.matmul(np.asarray(transition[later_problems].transpose()),                              
                              transition[transition_col])
        transition['npairs'] = nrows
        if nrows == 0:
            transition.iloc[:,:]=np.nan
        ### 'FixingSyntaxError', 'MakingNewSyntaxError'
        syntaxError = early_sub.loc[pd.notna(early_sub['FixingSyntaxError']),:].copy()
        nrows = syntaxError.shape[0]
        syntaxError[later_problems] = syntaxError[later_problems] / syntaxError[later_problems].sum()            
        if use_order_weight:
            syntaxError['order_weight'] = syntaxError['order_weight'] / syntaxError['order_weight'].sum()
            if not_use_similiarity_weight:
                for lp in later_problems:
                    syntaxError[lp] = syntaxError['order_weight'].copy()
            # if combine order and similiarity weight, take their means as the final weight        
            else:
                for lp in later_problems:
                    syntaxError[lp] = (syntaxError[lp] + syntaxError['order_weight'])/(num_similiarity_weight+1)  
        syntaxError = np.matmul(np.asarray(syntaxError[later_problems].transpose()),                              
                              syntaxError[['FixingSyntaxError', 'MakingNewSyntaxError']])  
        if nrows == 0:
            syntaxError.iloc[:,:]=np.nan        
        ### 'ChangeTestScores', 'IncreasedTestScore'
        semanticError = early_sub.loc[pd.notna(early_sub['ChangeTestScores']),:].copy()
        nrows = semanticError.shape[0]
        semanticError[later_problems] = semanticError[later_problems] / semanticError[later_problems].sum()
        if use_order_weight:
            semanticError['order_weight'] = semanticError['order_weight'] / semanticError['order_weight'].sum()
            if not_use_similiarity_weight:
                for lp in later_problems:
                    semanticError[lp] = semanticError['order_weight'].copy()
            # if combine order and similiarity weight, take their means as the final weight        
            else:
                for lp in later_problems:
                    semanticError[lp] = (semanticError[lp] + semanticError['order_weight'])/(num_similiarity_weight+1)  
                    
        semanticError = np.matmul(np.asarray(semanticError[later_problems].transpose()),                              
                              semanticError[[ 'ChangeTestScores', 'IncreasedTestScore']])   
        if nrows == 0:
            semanticError.iloc[:,:]=np.nan
        # combine features
        early_sub = pd.concat([syntaxError, semanticError, transition], axis = 1)

        # add subject and problem id
        early_sub['SubjectID'] = Subject
        early_sub['ProblemID'] = later_problems
        
        debugging = pd.concat([debugging, early_sub], axis = 0, ignore_index = True)
    
    # compute transition strength    
    for given in ['SE', 'TE']:
        for target in ['SE', 'TE', 'correct']:
            a = debugging["_".join([given, target])].copy()
            b = debugging["_".join(['given', given])] - a
            c = debugging["_".join(['target', target])] - a
            d = debugging['npairs'] - a - b - c
            constant = 0.5/debugging['npairs']
            oddsratio = (a+constant)*(b+constant)/(c+constant)/(d+constant)
            debugging["_".join([given, target])] = np.log(oddsratio.astype('float'))
    debugging = debugging[['SubjectID', 'ProblemID', 
                    'FixingSyntaxError', 'MakingNewSyntaxError',
                   'ChangeTestScores', 'IncreasedTestScore',
                   'SE_SE', 'SE_TE', 'SE_correct',
                   'TE_SE', 'TE_TE', 'TE_correct']]        
    debugging.columns = ['SubjectID', 'ProblemID', 
                    'percFixingSyntaxError', 'percMakingNewSyntaxError',
                   'avgChangeTestScores', 'percIncreasedTestScore',
                   'SE_SE', 'SE_TE', 'SE_correct',
                   'TE_SE', 'TE_TE', 'TE_correct']    
    ## clear memory
    del raw_data_early, a, b, c, d, oddsratio, constant
    
    #### fill missing value
    ## Values in these columns are because the subject performed very well in all problems, 
    ## solving problems in either one attempt or few attempts without making syntax or test errors.
    late_problem_subject.loc[late_problem_subject["avgSubjectNumSyntaxErrors"].isna(),"avgSubjectNumSyntaxErrors"]=0
    debugging.loc[debugging["percMakingNewSyntaxError"].isna(),"percMakingNewSyntaxError"]=0
    debugging.loc[debugging["percFixingSyntaxError"].isna(),"percFixingSyntaxError"]=1
    debugging.loc[debugging["avgChangeTestScores"].isna(),"avgChangeTestScores"]=1
    debugging.loc[debugging["percIncreasedTestScore"].isna(),"percIncreasedTestScore"]=1
    debugging.loc[debugging["SE_SE"].isna(),"SE_SE"]=np.min(debugging["SE_SE"])
    debugging.loc[debugging["SE_TE"].isna(),"SE_TE"]=np.max(debugging["SE_TE"])
    debugging.loc[debugging["SE_correct"].isna(),"SE_correct"]=np.max(debugging["SE_correct"])
    debugging.loc[debugging["TE_SE"].isna(),"TE_SE"]=np.min(debugging["TE_SE"])
    debugging.loc[debugging["TE_TE"].isna(),"TE_TE"]=np.min(debugging["TE_TE"])
    debugging.loc[debugging["TE_correct"].isna(),"TE_correct"]=np.max(debugging["TE_correct"])
    
    #### vector
    tfidf_train = pd.read_csv('data/ours/tfidf_All_train.csv')
    tfidf_test = pd.read_csv('data/ours/tfidf_F19_test.csv')
    tfidf = pd.concat([tfidf_train, tfidf_test], axis=0)
    
    code2vec_train = pd.read_csv('data/ours/vectors.phase2_early_train.csv')
    code2vec_test = pd.read_csv('data/ours/vectors.phase2_early_test.csv')
    code2vec = pd.concat([code2vec_train, code2vec_test], axis=0)
    # clear memory
    del tfidf_train, tfidf_test, code2vec_train, code2vec_test
    
    ### preprocessing tfidf vector
    col_names = ["tfidf_"+str(i) for i in range(50)]
    col_names = ['SubjectID', 'AssignmentID', 'ProblemID'] + col_names 
    tfidf.columns = col_names
    ## columns had more than 90% none-zeros cells
    unused_columns = np.where((tfidf.iloc[:, 3:] > 0).mean() < 0.1)[0]
    unused_columns = ["tfidf_"+str(i) for i in unused_columns]
    tfidf = tfidf.drop(unused_columns, axis = 1)
    
    ### preprocessing code2vec vector

    cov = np.cov(code2vec.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov)
# =============================================================================
#     import matplotlib.pyplot as plt
#     print('The first 10 eigenvalues of the the Covariance Matrix')
#     print(eigenvalues[:20])
#     
#     # plot the eigenvalues
#     plt.figure(figsize=(20,10))
#     plt.plot(eigenvalues[:20])
#     plt.xlabel('Component')
#     plt.ylabel('Eigenvalue')
#     plt.show()
# =============================================================================
    
    ## 12 components are sufficient
    ## extract  scores of the first 15 PCA components
    n_components = 16
    pca = PCA(n_components=n_components, random_state=42)
    pca_scores = pca.fit_transform(code2vec)    
    pca_scores = pd.DataFrame(pca_scores, columns=["code2vecPCA"+str(i) for i in range(n_components)])
    #### add SubjectID and ProblemID
    # if only S19 train, i.e., phase 1 data
    if onlyS19:
        pca_scores["SubjectID"] = SubjectIDs 
        pca_scores["ProblemID"] = ProblemIDs
    else:
        pca_scores["SubjectID"] = early["SubjectID"].values 
        pca_scores["ProblemID"] = early["ProblemID"].values 
    

    
    vectors = tfidf.merge(pca_scores, on = ["SubjectID", "ProblemID"])
    del tfidf, pca_scores
    
    if not weight_vector:
        use_order_weight = False
        weight_matrix.iloc[:,1:] = 1
    ## weight vectors
    used_cols = vectors.drop(["SubjectID", "AssignmentID","ProblemID"],axis=1).columns.to_list()
    if use_order_weight:
        vectors = vectors.merge(order_weight, how = 'left', on = ['SubjectID', 'ProblemID'])
                
    
    vectors = vectors.merge(weight_matrix, how = "left", on = ["ProblemID"])    
    vector_df =  pd.DataFrame(columns=['SubjectID', 'ProblemID']+used_cols) 
    for Subject in ids:
        early_sub = vectors.loc[vectors.SubjectID == Subject,:].copy()
        # normalize the log odds ratio to make it suitable as weight
        early_sub[later_problems] = early_sub[later_problems] / early_sub[later_problems].sum()
        if use_order_weight:
            # if not combine order and similiarity weight
            # replace similiarity weight with order weight
            if not_use_similiarity_weight:
                for lp in later_problems:
                    early_sub[lp] = early_sub['order_weight'].copy()
            # if combine order and similiarity weight, take their means as the final weight        
            else:
                for lp in later_problems:
                    early_sub[lp] = (early_sub[lp] + early_sub['order_weight'])/(num_similiarity_weight+1)
        # compute the weighted percSubjectStruggleWeightedCorr
        early_sub = np.matmul(np.asarray(early_sub[later_problems].transpose()),                              
                              early_sub[used_cols])
        
        early_sub['SubjectID'] = Subject
        early_sub['ProblemID'] = later_problems
        vector_df = pd.concat([vector_df, early_sub], axis = 0, ignore_index = True)
    # drop the first row
    vector_df.columns = ['SubjectID', 'ProblemID'] + used_cols    
    
   

    if onlyS19:
        # only s19 train
        late_problem_subject = late_problem_subject.merge(vector_df, how = "left", on = ['SubjectID', 'ProblemID'])
        late_problem_subject = late_problem_subject.merge(debugging, how = "left", on = ['SubjectID', 'ProblemID'])
    else:
        # late_problem_subject, vector_df, debugging have the same SubjectID and ProblemID order
        late_problem_subject = late_problem_subject.join(vector_df.iloc[:,2:], how = 'left')
        late_problem_subject = late_problem_subject.join(debugging.iloc[:,2:], how = 'left')  
        
    return late_problem_subject

# %% function for subjects' features in the first 30 problems
def get_early_subject_df(early, raw_data_early):
    early = early.copy()
    raw_data_early = raw_data_early.copy()
    # get the change of percSubjectStruggle across assignments
    early_subject_df = get_struggle_change(early)   

    temp = early.groupby(['SubjectID']).size().reset_index(name="problemsAttempted")
    early_subject_df = early_subject_df.merge(temp, how = "left", on = "SubjectID")
    
    temp = early.groupby(['SubjectID']).agg(
        medAttempts = ('Attempts', np.median), # Median attempts made on early problems
        maxAttempts = ('Attempts', np.max)) # Max attempts made on early problems
    early_subject_df = early_subject_df.merge(temp, how = "left", on = "SubjectID")
       
    # time on a problem
    temp = get_time_feature(raw_data_early, early)
    early_subject_df = early_subject_df.merge(temp, how = "left", on = 'SubjectID')    
    # num of lines added, deleted, and modified
    temp = get_all_lines_adm(raw_data_early)
    early_subject_df = early_subject_df.merge(temp, how = "left", on = 'SubjectID') 
    
    # spacing
    raw_data_early['ServerTimestamp'] = raw_data_early['ServerTimestamp'].apply(lambda x: pd.Timestamp(x, tz = "US/Eastern"))
    raw_data_early['date'] = [x.date() for x in raw_data_early['ServerTimestamp']]       
    temp = raw_data_early.groupby(["SubjectID"]).agg(spacing = ('date', lambda x: len(set(x)))).reset_index()   
    early_subject_df = early_subject_df.merge(temp, how = "left", on = 'SubjectID')  
    
    early_subject_df.loc[early_subject_df["avgNumLinesADM"].isna(),"avgNumLinesADM"]=np.nanmedian(early_subject_df["avgNumLinesADM"])
    
    return early_subject_df

#%% extract weight features
def extract_weight_features(weight_matrix,
                     X_train_base, late_test,
                     early, 
                     raw_data_early, 
                     late_problem_df, 
                     use_order_weight = False,
                     weight_vector = False,
                     onlyS19=False):
    X_train = X_train_base.merge(late_problem_df, how = 'left', on = ['ProblemID'])
    X_test = late_test.merge(late_problem_df, how = 'left', on = ['ProblemID'])
    
    late_problem_subject = weight_feature(weight_matrix, early, raw_data_early,
                     use_order_weight = use_order_weight,
                     weight_vector = weight_vector,
                     onlyS19 = onlyS19)

    ## Subset late_problem_subject then merge it with X_train, otherwise merge will 
    ## consume too much memory and cause kernel to restart
    subjects_train = list(set(X_train.SubjectID))
    subjects_test = list(set(X_test.SubjectID))
    
    X_train = X_train.merge(late_problem_subject.loc[[x in subjects_train for x in late_problem_subject.SubjectID],:], how = 'left', on = ['SubjectID', 'ProblemID'])
    X_test = X_test.merge(late_problem_subject.loc[[x in subjects_test for x in late_problem_subject.SubjectID],:], how = 'left', on = ['SubjectID', 'ProblemID'])
    
    # diffProblemSubjectStruggle
    X_train['diffProblemSubjectStruggle'] = 1 - X_train['percSubjectStruggle'] - X_train['percProblemStruggle']
    # diffProblemSubjectSyntax
    X_train['diffProblemSubjectSyntaxError'] = 1 - X_train['percSubjectSyntaxErrors'] - X_train['percProblemSyntaxError']
    # diffProblemSubjectTest
    X_train['diffProblemSubjectSemanticError'] = 1 - X_train['percSubjectSemanticErrors'] - X_train['percProblemSemanticError']      
    
    # diffProblemSubjectStruggle
    X_test['diffProblemSubjectStruggle'] = 1 - X_test['percSubjectStruggle'] - X_test['percProblemStruggle']
    # diffProblemSubjectSyntax
    X_test['diffProblemSubjectSyntaxError'] = 1 - X_test['percSubjectSyntaxErrors'] - X_test['percProblemSyntaxError']
    # diffProblemSubjectTest
    X_test['diffProblemSubjectSemanticError'] = 1 - X_test['percSubjectSemanticErrors'] - X_test['percProblemSemanticError']      

    return X_train, X_test
    
# %% weight features 

early = pd.concat([early_train, early_test], axis = 0, join = 'inner').reset_index(drop=True)
raw_data_early = pd.concat([raw_data_early_train, raw_data_early_test], axis = 0, join = 'inner').reset_index(drop=True)
early_subject_df = get_early_subject_df(early, raw_data_early)

del raw_data_early_train, raw_data_early_test, raw_data_late_train, early_train, early_test
del early_train_F, late_train_F, X_train_base_F, X_train_base_S
#### by code and prompt
# read weight matrix
from os import listdir
from os.path import isfile, join
weight_path = "data/ours/weights/"
weight_files = [f for f in listdir(weight_path) if isfile(join(weight_path, f))]

if onlyS19:
    outpath='data/ours/additional_weighting_comb/S19_'
    weight_files = [x for x in weight_files if (("Sp19" in x) or ("prompts" in x)) ]
else:
    outpath='data/ours/additional_weighting_comb/'
    weight_files = [x for x in weight_files if (("all" in x) or ("prompts" in x)) ]


comb_list = [['difficulty','prompt'],
    ['difficulty','code2vec'],
    ['prompt','code2vec'],
    ['order','difficulty','prompt'],
    ['order','difficulty','code2vec'],
    ['order','prompt','code2vec'],
    ['difficulty','prompt','code2vec']]

similarity = 'cosine'
#### combine order, difficulty, prompt, and code weights
for comb in comb_list:
    weight_matrix = []
    use_order_weight = False
    for w in comb:            
        if w == 'difficulty':
            weight_matrix.append(lor_df.copy())
        elif w == 'order':
            use_order_weight = True
        else:            
            temp_file = [f for f in weight_files if (similarity in f) & (w in f)][0]
            temp_weight = pd.read_csv(weight_path+temp_file)
            temp_weight[temp_weight < 0] = np.nan
            temp_weight.fillna(value = temp_weight.min(), inplace = True)        
            temp_weight.columns = ['ProblemID']+later_problems
            weight_matrix.append(temp_weight)
    
    X_train, X_test = extract_weight_features(
                         weight_matrix,
                         X_train_base, late_test,
                         early, 
                         raw_data_early, 
                         late_problem_df, 
                         use_order_weight = use_order_weight,
                         weight_vector = True,
                         onlyS19 = onlyS19)
    X_train = X_train.merge(early_subject_df, how = 'left', on = ['SubjectID'])
    X_test = X_test.merge(early_subject_df, how = 'left', on = ['SubjectID'])
    X_train.drop(['SubjectID', 'ProblemID'],axis=1, inplace = True)
    X_test.drop(['SubjectID', 'ProblemID'],axis=1, inplace = True)
    
    X_train.to_csv(outpath+'X_train_'+"_".join(comb)+'_cosine.csv', index=False)
    X_test.to_csv(outpath+'X_test_'+"_".join(comb)+'_cosine.csv', index=False)
