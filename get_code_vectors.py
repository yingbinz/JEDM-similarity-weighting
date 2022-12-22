# %%
from codecs import ignore_errors
import pandas as pd

early_train_F = pd.read_csv('data/F19/Train/early.csv')
late_train_F = pd.read_csv('data/F19/Train/late.csv')
raw_data_early_train_F = pd.read_csv('data/ours/raw_train_early_F19.csv')
raw_data_late_train_F = pd.read_csv('data/ours/raw_train_late_F19.csv')

early_train_S = pd.read_csv('data/S19/ALL/early.csv')
late_train_S = pd.read_csv('data/S19/All/late.csv')
raw_data_early_train_S = pd.read_csv('data/ours/raw_train_early_S19_All.csv')
raw_data_late_train_S = pd.read_csv('data/ours/raw_train_late_S19_All.csv')

early_test = pd.read_csv('data/F19/Test/early.csv')
raw_data_early_test = pd.read_csv('data/ours/raw_test_early_F19.csv')
# late_test = pd.read_csv('data/F19/Test/late.csv')

early_train = pd.concat([early_train_F, early_train_S], axis = 0).reset_index(drop=True)
late_train = pd.concat([late_train_F, late_train_S], axis = 0).reset_index(drop=True)
raw_data_early_train = pd.concat([raw_data_early_train_F, raw_data_early_train_S], axis = 0).reset_index(drop=True)
raw_data_late_train = pd.concat([raw_data_late_train_F, raw_data_late_train_S], axis = 0).reset_index(drop=True)


# # Import student-problem-level data
# early_train = pd.read_csv('data/S19/Train/early.csv')
# late_train = pd.read_csv('data/S19/Train/late.csv')
# early_test = pd.read_csv('data/F19/Test/early.csv')


# # Import submission-level data
# raw_data_early_train = pd.read_csv('data/ours/raw_train_early_S19.csv')
# raw_data_late_train = pd.read_csv('data/ours/raw_train_late_S19.csv')
# raw_data_early_test = pd.read_csv('data/ours/raw_test_early_F19.csv')


# %%
def getLastSubCode(raw, main):
    main = main.copy()
    missing = 0
    for i in main.index:
        s = main.iloc[i]['SubjectID']
        p = main.iloc[i]['ProblemID']
        temp = raw[(raw['SubjectID'] == s) & (raw['ProblemID'] == p)]

        # Drop attempts with compiler errors (uncompilable code can't be vectorized)
        temp = temp[temp['CompileErrors'].isna()]

        # Skip empty subject-problem combinations
        if len(temp) == 0:
            missing = missing + 1
            continue

        # Take last attempt or last correct attempt
        if (temp['Score'].max() == 1) & (temp['Score'].iloc[-1] != 1):
            keep = temp[temp['Score'] == 1].iloc[-1]
        else:
            keep = temp.iloc[-1]

        # Add code to dataframe
        main.at[i, 'Code'] = keep['Code']

    print('missing:', missing)
    return main


# %%
# def exportCode(folder, data):
#     for i in data.iterrows():
#         with open('data/ours/code/' + folder + '/' + str(i[0]) + '.java', 'w') as output:
#             output.write(i[1]['Code'])

#     print(folder, 'exported')


# # %%
# exportCode('early_train', getLastSubCode(raw_data_early_train, early_train))
# exportCode('late_train', getLastSubCode(raw_data_late_train, late_train))
# exportCode('early_test', getLastSubCode(raw_data_early_test, early_test))


# %%
early_train = getLastSubCode(raw_data_early_train, early_train)
late_train = getLastSubCode(raw_data_late_train, late_train)
early_test = getLastSubCode(raw_data_early_test, early_test)

early_train.to_feather('data/ours/vector_processing/phase2_early_train.feather')
late_train.to_feather('data/ours/vector_processing/phase2_late_train.feather')
early_test.to_feather('data/ours/vector_processing/phase2_early_test.feather')


# %%
"""
To extract vectors, ensure modified `interactive_predict.py` is active and run this in terminal (while in code2vec folder):

/Users/juanpinto/miniforge3/envs/tensorflow_m1/bin/python3 code2vec.py --load models/java14_model/saved_model_iter8.release --predict --export_code_vectors
"""


# %%
# Preprocess code to prepare ready for vectorization with code2vec
# import subprocess

# def preprocessCode(code):
#     preprocessed = subprocess.check_output('echo "' + code.replace('"', '\\"') + '" | java -cp code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar JavaExtractor.App --max_path_length 8 --max_path_width 2 --num_threads 64 --file /dev/stdin', shell=True)

#     return preprocessed.decode('utf-8')

# early_train['PreprocessedCode'] = early_train['Code'].apply(preprocessCode)


# %%
# subprocess.check_output('echo "' + early_train['PreprocessedCode'].iloc[0] + '" | /Users/juanpinto/miniforge3/envs/tensorflow_m1/bin/python3 code2vec/code2vec.py --load code2vec/models/java14_model/saved_model_iter8.release --test /dev/stdin --predict --export_code_vectors', shell=True)
# %%
# import subprocess

# folder = 'early_test'
# for filename in os.listdir('data/ours/code/' + folder):
#     if not filename.endswith('.java'):
#         continue
#     print(folder, filename)
#     preprocessed = subprocess.check_output('java -cp code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar JavaExtractor.App --max_path_length 8 --max_path_width 2 --num_threads 64 --file data/ours/code/' + folder + '/' + filename, shell=True)

#     # return preprocessed.decode('utf-8')
#     with open('data/ours/code/' + folder + '-preprocessed/' + filename[:-5]):
#         asdf

#     break

# %%
# preprocessed = preprocessCode(early_train['Code'].iloc[0])

# with open('./preprocessed.c2v', 'w') as f:
#     f.write(pre_string)

# os.system('/Users/juanpinto/miniforge3/envs/tensorflow_m1/bin/python3 code2vec/code2vec.py --load code2vec/models/java14_model/saved_model_iter8.release --test ./preprocessed.c2v --export_code_vectors')

# %%
################################################################
################################################################

# vectors_early_train = pd.read_feather('data/ours/vectors_early_train.feather')
# vectors_late_train = pd.read_feather('data/ours/vectors_late_train.feather')
# vectors_early_test = pd.read_feather('data/ours/vectors_early_test.feather')
vectors_early_train = pd.read_csv('data/ours/vectors.phase2_early_train.csv')
vectors_late_train = pd.read_csv('data/ours/vectors.phase2_late_train.csv')
vectors_early_test = pd.read_csv('data/ours/vectors.phase2_early_test.csv')


# %%
print(vectors_early_train['Vector'].isna().sum())
print(vectors_late_train['Vector'].isna().sum())
print(vectors_early_test['Vector'].isna().sum())


# %%
vectors_early_train[vectors_early_train['Vector'].isna()]


# %%
raw_data_early_train[(raw_data_early_train['SubjectID'] == '2d66aed7a5328a988f77cbaec59fc047') & (raw_data_early_train['ProblemID'] == 101)]


# %%
################################################################
################################################################
"""
I had originally allowed attempts with compiler errors to count, and I found that apparently some can actually yield vectors! Still some were missing, so I re-extracted submission code while dropping attempts with compiler errors. I figured that some missing vectors could probably be calculated using these other submissions codes, so I identified which these were and re-exported those codes here. In the following cells I then import the vectors and insert them in with the rest.
"""
def getMissingCompilableCode(code, vectors):
    missing_df = pd.DataFrame()
    for i in vectors[vectors['Vector'].isna()].index:
        if i not in code[code['Code'].isna()].index:
            missing_df = missing_df.append(code.iloc[i])

    return missing_df

# %%
early_train_missing = getMissingCompilableCode(early_train, vectors_early_train)
late_train_missing = getMissingCompilableCode(late_train, vectors_late_train)
early_test_missing = getMissingCompilableCode(early_test, vectors_early_test)

early_train_missing.to_csv('data/ours/vector_processing/phase2_missing_early_train.csv')
late_train_missing.to_csv('data/ours/vector_processing/phase2_missing_late_train.csv')
early_test_missing.to_csv('data/ours/vector_processing/phase2_missing_early_test.csv')

# %%
early_train_missing_vectors = pd.read_csv('data/ours/vectors_missing_early_train.csv')
late_train_missing_vectors = pd.read_csv('data/ours/vectors_missing_late_train.csv')
early_test_missing_vectors = pd.read_csv('data/ours/vectors_missing_early_test.csv')

# %%
def insertNewVectors(missing, vectors):
    vectors = vectors.copy()
    for i in missing.index:
        j = missing.iloc[i]['Unnamed: 0.1']
        vectors.at[j, 'Code'] = missing.iloc[i]['Code']
        vectors.at[j, 'Vector'] = missing.iloc[i]['Vector']

    return vectors


# %%
vectors_early_train = insertNewVectors(early_train_missing_vectors, vectors_early_train)
vectors_late_train = insertNewVectors(late_train_missing_vectors, vectors_late_train)
vectors_early_test = insertNewVectors(early_test_missing_vectors, vectors_early_test)


# %%
################################################################
################################################################
# Convert vectors from str to list and insert 0s for missing vectors
import numpy as np

def convertVectors(main):
    vectors = main['Vector'].apply(lambda x: '' if type(x) != str else x).apply(lambda x: x.split(' '))
    vectors = vectors.apply(lambda x: [0] * 384 if len(x) == 1 else x)

    return vectors

v_early_train = convertVectors(vectors_early_train)
v_late_train = convertVectors(vectors_late_train)
v_early_test = convertVectors(vectors_early_test)


# %%
# Convert list vector to features
v_early_train_df = pd.DataFrame.from_dict(dict(zip(v_early_train.index, v_early_train.values))).T.add_prefix('codeVector_')
v_late_train_df = pd.DataFrame.from_dict(dict(zip(v_late_train.index, v_late_train.values))).T.add_prefix('codeVector_')
v_early_test_df = pd.DataFrame.from_dict(dict(zip(v_early_test.index, v_early_test.values))).T.add_prefix('codeVector_')


# %%
v_early_train_df.to_csv('data/ours/vectors.phase2_early_train.csv', index=False)
v_late_train_df.to_csv('data/ours/vectors.phase2_late_train.csv', index=False)
v_early_test_df.to_csv('data/ours/vectors.phase2_early_test.csv', index=False)


# %%
############################################################
############################################################
'''
Since we don't actually have the code for the late test submissions, we can't calculate vectors for them. Here I use the mean of vectors from corresponding students and attempt to replicate that scenario with the training data by creating a more realistic set of late train vectors to use.
'''

# Load base datasets
# early_train = pd.read_csv('data/S19/Train/early.csv')
# late_train = pd.read_csv('data/S19/Train/late.csv')
# early_test = pd.read_csv('data/F19/Test/early.csv')
# late_test = pd.read_csv('data/F19/Test/late.csv')
early_train = pd.read_feather('data/ours/vector_processing/phase2_early_train.feather')
late_train = pd.read_feather('data/ours/vector_processing/phase2_late_train.feather')
early_test = pd.read_feather('data/ours/vector_processing/phase2_early_test.feather')
late_test = pd.read_csv('data/F19/Test/late.csv')


# Load vectors
# v_early_train_df = pd.read_csv('data/ours/vectors.early_train.csv')
# v_late_train_df = pd.read_csv('data/ours/vectors.late_train.csv')
# v_early_test_df = pd.read_csv('data/ours/vectors.early_test.csv')
v_early_train_df = pd.read_csv('data/ours/vectors.phase2_early_train.csv')
v_late_train_df = pd.read_csv('data/ours/vectors.phase2_late_train.csv')
v_early_test_df = pd.read_csv('data/ours/vectors.phase2_early_test.csv')


# %%
# Add SubjectID and ProblemID to vector dfs
early_train_df = pd.concat([early_train[['SubjectID', 'ProblemID']], v_early_train_df], axis=1)
early_test_df = pd.concat([early_test[['SubjectID', 'ProblemID']], v_early_test_df], axis=1)


# %%
# Keep only F19 data
early_train_df = early_train_df[:10657]


# %%
# For cases of missing code, replace with problem-level mean
early_train_df_missing = early_train_df[
    (early_train_df['codeVector_0'] == 0) &
    (early_train_df['codeVector_1'] == 0) &
    (early_train_df['codeVector_2'] == 0) &
    (early_train_df['codeVector_3'] == 0)
]
early_test_df_missing = early_test_df[
    (early_test_df['codeVector_0'] == 0) &
    (early_test_df['codeVector_1'] == 0) &
    (early_test_df['codeVector_2'] == 0) &
    (early_test_df['codeVector_3'] == 0)
]

early_train_df_dropMissing = early_train_df.drop(early_train_df_missing.index)
early_test_df_dropMissing = early_test_df.drop(early_test_df_missing.index)

early_train_df_problemMeans = early_train_df_dropMissing.groupby('ProblemID').mean()
early_test_df_problemMeans = early_test_df_dropMissing.groupby('ProblemID').mean()

early_train_df_missingMeanReplacement = early_train_df_missing.iloc[:, :2].merge(early_train_df_problemMeans, on='ProblemID', how='left')
early_test_df_missingMeanReplacement = early_test_df_missing.iloc[:, :2].merge(early_test_df_problemMeans, on='ProblemID', how='left')

early_train_df_withReplacement = pd.concat([early_train_df_dropMissing, early_train_df_missingMeanReplacement])
early_test_df_withReplacement = pd.concat([early_test_df_dropMissing, early_test_df_missingMeanReplacement])


# %%
# Calculate mean for each vector grouped by SubjectID
v_mean_early_train_df = early_train_df_withReplacement.groupby('SubjectID').mean().drop('ProblemID', axis=1)
v_mean_early_test_df = early_test_df_withReplacement.groupby('SubjectID').mean().drop('ProblemID', axis=1)


# %%
# Create new late_train vector set
v_late_train_df_realistic = late_train[['SubjectID']].merge(v_mean_early_train_df, on='SubjectID').drop('SubjectID', axis=1)
v_late_test_df = late_test[['SubjectID']].merge(v_mean_early_test_df, on='SubjectID').drop('SubjectID', axis=1)


# %%
# v_late_train_df_realistic.to_csv('data/ours/vectors.late_train.realistic.csv', index=False)
# v_late_test_df.to_csv('data/ours/vectors.late_test.csv', index=False)
v_late_train_df_realistic.to_csv('data/ours/vectors.phase2_F19_late_train.realistic.csv', index=False)
v_late_test_df.to_csv('data/ours/vectors.phase2_late_test.csv', index=False)


# %%
############################################################
# Weigh later problems heavier
############################################################

# Get weights
problemReplacements = dict(list(zip(early_train_df_withReplacement.ProblemID.unique(), range(1,31))))
early_train_df_withReplacement['weight'] = early_train_df_withReplacement['ProblemID'].replace(problemReplacements)
early_test_df_withReplacement['weight'] = early_test_df_withReplacement['ProblemID'].replace(problemReplacements)

# Multiply by weights
early_train_df_withReplacement.iloc[:,2:-1] = early_train_df_withReplacement.iloc[:,2:-1].mul(early_train_df_withReplacement['weight'], axis=0)
early_test_df_withReplacement.iloc[:,2:-1] = early_test_df_withReplacement.iloc[:,2:-1].mul(early_test_df_withReplacement['weight'], axis=0)


# %%
# Calculate mean for each vector grouped by SubjectID
v_mean_early_train_df = early_train_df_withReplacement.groupby('SubjectID').mean().drop('ProblemID', axis=1)
v_mean_early_test_df = early_test_df_withReplacement.groupby('SubjectID').mean().drop('ProblemID', axis=1)


# %%
# Create new late_train vector set
v_late_train_df_realistic = late_train[['SubjectID']].merge(v_mean_early_train_df, on='SubjectID').drop('SubjectID', axis=1)
v_late_test_df = late_test[['SubjectID']].merge(v_mean_early_test_df, on='SubjectID').drop('SubjectID', axis=1)


# %%
v_late_train_df_realistic.to_csv('data/ours/vectors.phase2_late_train.realistic_weightedByLateness.csv', index=False)
v_late_test_df.to_csv('data/ours/vectors.phase2_late_test.realistic_weightedByLateness.csv', index=False)


# %%
############################################################
############################################################
'''
An alternative approach here is to calculate a vector that is the weighted mean of early vectors based on each problem's mean vector's similarity to the late problem mean vector.
'''

# Add SubjectID and ProblemID to vector dfs
early_train_df = pd.concat([early_train[['SubjectID', 'ProblemID']], v_early_train_df], axis=1)
late_train_df = pd.concat([late_train[['SubjectID', 'ProblemID']], v_late_train_df], axis=1)


# %%
# Keep only F19 data
early_train_df = early_train_df[:10657]
late_train_df = late_train_df[:7021]


# %%
# Remove cases of missing code to avoid miscalculating means
early_train_df_notMissing = early_train_df[
    ~(
    (early_train_df['codeVector_0'] == 0) &
    (early_train_df['codeVector_1'] == 0) &
    (early_train_df['codeVector_2'] == 0) &
    (early_train_df['codeVector_3'] == 0)
    )
]
late_train_df_notMissing = late_train_df[
    ~(
    (late_train_df['codeVector_0'] == 0) &
    (late_train_df['codeVector_1'] == 0) &
    (late_train_df['codeVector_2'] == 0) &
    (late_train_df['codeVector_3'] == 0)
    )
]

# Calculate vector element means grouped by problem
v_pmean_early_train_df = early_train_df_notMissing.groupby('ProblemID').mean()
v_pmean_late_train_df = late_train_df_notMissing.groupby('ProblemID').mean()


# %%
# For each late problem, rank neighbors among early problems
nearest_v_neighbors = pd.DataFrame()
for i in v_pmean_late_train_df.index:
    # Subtract vector elements
    temp = v_pmean_early_train_df - v_pmean_late_train_df.loc[i]

    # Take absolute values
    temp = temp.abs()

    # Get sum
    temp = temp.sum(axis=1)

    # Sort and invert values
    # temp = temp.sort_values().apply(lambda x: 1/x)
    temp = temp.apply(lambda x: 1/x)
    temp = pd.Series(temp, name=i)

    # Add sorted neighbors and weights to dataframe
    # nearest_v_neighbors = pd.concat([nearest_v_neighbors, pd.DataFrame({str(i) +'_neighbor': temp.index, str(i) + '_weight': temp.values})], axis=1)
    nearest_v_neighbors = nearest_v_neighbors.append(temp)

nearest_v_neighbors


# %%
# nearest_v_neighbors.to_csv('data/ours/phase2_nearest_v_neighbors.csv', index=False)

# %%
nearest_v_neighbors
# %%
# Create weighted vector for each instance in late df
def getWeightedVectors(weights, late_base_df, early_base_df, early_v_df):
    v_weighted_df = pd.DataFrame()
    for i in late_base_df.index:
        s = late_base_df.loc[i]['SubjectID']
        p = late_base_df.loc[i]['ProblemID']

        # Get early vectors for current subject
        early_df = pd.concat([early_base_df[['SubjectID', 'ProblemID']], early_v_df], axis=1)
        subject_v_df = early_df[early_df['SubjectID'] == s].drop('SubjectID', axis=1)

        # Align problem weights with corresponding vectors for subject
        temp = weights.loc[p]
        temp = temp.loc[subject_v_df['ProblemID'].values].values

        # Multiply vectors by weights and get mean for each vector element
        weighted_v = subject_v_df.drop('ProblemID', axis=1).mul(temp, axis=0).mean()

        v_weighted_df = v_weighted_df.append(weighted_v, ignore_index=True)

    return v_weighted_df


# %%
v_late_train_df_weighted = getWeightedVectors(nearest_v_neighbors, late_train, early_train, v_early_train_df)
v_late_test_df_weighted = getWeightedVectors(nearest_v_neighbors, late_test, early_test, v_early_test_df)

# Not sure if it makes much sense to weigh vectors in this way. Multiplying by inverse distance results in very different kinds of values than the original vectors.


# %%
# Arbitrary multiplication to avoid using such tiny values
v_late_train_df_weighted = v_late_train_df_weighted * 1000
v_late_test_df_weighted = v_late_test_df_weighted * 1000


# %%
# Export new weighted vectors
v_late_train_df_weighted.to_csv('data/ours/vectors.phase2_F19_late_train.weighted.csv', index=False)
v_late_test_df_weighted.to_csv('data/ours/vectors.phase2_F19_late_test.weighted.csv', index=False)


# %%
############################################################
# Same weighted idea as above, but using correlations as weights

# Create correlation matrix between early and late mean vectors
v_corr_df = pd.DataFrame()
for i in v_pmean_late_train_df.index:
    corr_i = v_pmean_early_train_df.append(v_pmean_late_train_df.loc[i]).T.corr()[i]
    corr_i = corr_i.abs().drop(i)

    v_corr_df = v_corr_df.append(corr_i)


# %%
v_late_train_df_weightedCorr = getWeightedVectors(v_corr_df, late_train, early_train, v_early_train_df)
v_late_test_df_weightedCorr = getWeightedVectors(v_corr_df, late_test, early_test, v_early_test_df)


# %%
# Arbitrary multiplication
v_late_train_df_weightedCorr = v_late_train_df_weightedCorr * 100
v_late_test_df_weightedCorr = v_late_test_df_weightedCorr * 100


# %%
# Export new weighted-by-correlation vectors
v_late_train_df_weightedCorr.to_csv('data/ours/vectors.phase2_F19_late_train.weightedCorr.csv', index=False)
v_late_test_df_weightedCorr.to_csv('data/ours/vectors.phase2_F19_late_test.weightedCorr.csv', index=False)


# %%
############################################################
############################################################
'''
WEIGHTED BY PROMPT SIMILARITY
'''

# Load base datasets
# early_train = pd.read_csv('data/S19/Train/early.csv')
# late_train = pd.read_csv('data/S19/Train/late.csv')
# early_test = pd.read_csv('data/F19/Test/early.csv')
# late_test = pd.read_csv('data/F19/Test/late.csv')
early_train = pd.read_feather('data/ours/vector_processing/phase2_early_train.feather')
late_train = pd.read_feather('data/ours/vector_processing/phase2_late_train.feather')
early_test = pd.read_feather('data/ours/vector_processing/phase2_early_test.feather')
late_test = pd.read_csv('data/F19/Test/late.csv')


# Load vectors
# v_early_train_df = pd.read_csv('data/ours/vectors.early_train.csv')
# v_late_train_df = pd.read_csv('data/ours/vectors.late_train.csv')
# v_early_test_df = pd.read_csv('data/ours/vectors.early_test.csv')
v_early_train_df = pd.read_csv('data/ours/vectors.phase2_early_train.csv')
v_late_train_df = pd.read_csv('data/ours/vectors.phase2_late_train.csv')
v_early_test_df = pd.read_csv('data/ours/vectors.phase2_early_test.csv')


# %%
# Add SubjectID and ProblemID to vector dfs
early_train_df = pd.concat([early_train[['SubjectID', 'ProblemID']], v_early_train_df], axis=1)
early_test_df = pd.concat([early_test[['SubjectID', 'ProblemID']], v_early_test_df], axis=1)


# %%
# Keep only F19 data
early_train_df = early_train_df[:10657]


# %%
# For cases of missing code, replace with problem-level mean
early_train_df_missing = early_train_df[
    (early_train_df['codeVector_0'] == 0) &
    (early_train_df['codeVector_1'] == 0) &
    (early_train_df['codeVector_2'] == 0) &
    (early_train_df['codeVector_3'] == 0)
]
early_test_df_missing = early_test_df[
    (early_test_df['codeVector_0'] == 0) &
    (early_test_df['codeVector_1'] == 0) &
    (early_test_df['codeVector_2'] == 0) &
    (early_test_df['codeVector_3'] == 0)
]

early_train_df_dropMissing = early_train_df.drop(early_train_df_missing.index)
early_test_df_dropMissing = early_test_df.drop(early_test_df_missing.index)

early_train_df_problemMeans = early_train_df_dropMissing.groupby('ProblemID').mean()
early_test_df_problemMeans = early_test_df_dropMissing.groupby('ProblemID').mean()

early_train_df_missingMeanReplacement = early_train_df_missing.iloc[:, :2].merge(early_train_df_problemMeans, on='ProblemID', how='left')
early_test_df_missingMeanReplacement = early_test_df_missing.iloc[:, :2].merge(early_test_df_problemMeans, on='ProblemID', how='left')

early_train_df_withReplacement = pd.concat([early_train_df_dropMissing, early_train_df_missingMeanReplacement]).reset_index(drop=True)
early_test_df_withReplacement = pd.concat([early_test_df_dropMissing, early_test_df_missingMeanReplacement]).reset_index(drop=True)


# %%
prompt_weights = pd.read_csv('data/ours/prompt_weights.csv').set_index('late')
prompt_weights = prompt_weights ** 50 * 10

# Weight later early problems heavier
prompt_weights = prompt_weights * range(1,31)


def getWeightedVectorsByPrompt(base, vectors, promptWeights):
    weightedVectors = pd.DataFrame(columns=vectors.columns)
    for row in base.iterrows():
        s = row[1]['SubjectID']
        p = row[1]['ProblemID']

        temp = vectors[vectors['SubjectID'] == s].copy()
        weights = temp['ProblemID'].apply(lambda x: promptWeights.at[p, str(x)])
        temp = pd.concat([temp.iloc[:,:2], temp.iloc[:,2:].multiply(weights, axis = 0)], axis=1)

        # Calculate mean of weighted vectors and append to base DF
        weightedVectors = pd.concat([weightedVectors, pd.concat([pd.DataFrame(pd.Series({'SubjectID': s, 'ProblemID': p})), pd.DataFrame(temp.iloc[:,2:].mean())]).T])

    return weightedVectors.reset_index(drop=True)


late_test_df_weightedByPromptSimilarity = getWeightedVectorsByPrompt(late_test, early_test_df_withReplacement, prompt_weights)
late_train_df_weightedByPromptSimilarity = getWeightedVectorsByPrompt(late_train, early_train_df_withReplacement, prompt_weights)


# %%
# late_test_df_weightedByPromptSimilarity.drop(['SubjectID', 'ProblemID'], axis=1).to_csv('data/ours/vectors.phase2_late_test.weightedByPrompt.csv', index=False)
# late_train_df_weightedByPromptSimilarity.drop(['SubjectID', 'ProblemID'], axis=1).to_csv('data/ours/vectors.phase2_late_train.weightedByPrompt.csv', index=False)
late_test_df_weightedByPromptSimilarity.drop(['SubjectID', 'ProblemID'], axis=1).to_csv('data/ours/vectors.phase2_late_test.weightedByPrompt+lateness.csv', index=False)
late_train_df_weightedByPromptSimilarity.drop(['SubjectID', 'ProblemID'], axis=1).to_csv('data/ours/vectors.phase2_late_train.weightedByPrompt+lateness.csv', index=False)

# %%
