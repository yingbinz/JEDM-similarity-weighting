# %%
import pandas as pd

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
late_train_df = pd.concat([late_train[['SubjectID', 'ProblemID']], v_late_train_df], axis=1)


# %%
# Keep only F19 data
# early_train_df = early_train_df[:10657]
# late_train_df = late_train_df[:7021]

# Keep only Sp19 data
early_train_df = early_train_df[10657:]
late_train_df = late_train_df[7021:]


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
import numpy as np

# Inverse Euclidean distance
def inverse_euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2)) ** -1

# Cosine similarity
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


# %%
def get_weights(func):
    weights_df = pd.DataFrame(index=v_pmean_early_train_df.index.values, columns=v_pmean_late_train_df.index)

    for i in weights_df.index:
        for c in weights_df.columns:
            weights_df.at[i, c] = func(v_pmean_early_train_df.loc[i], v_pmean_late_train_df.loc[c])

    return weights_df


# %%
cosine_weights = get_weights(cosine_similarity)
euclidean_weights = get_weights(inverse_euclidean_distance)
pearson_weights = get_weights(pd.Series.corr)


# %%
cosine_weights.to_csv('data/ours/weights/code2vec_cosineSimilarity_Sp19.csv', index_label='ProblemID')
euclidean_weights.to_csv('data/ours/weights/code2vec_inverseEuclideanDistance_Sp19.csv', index_label='ProblemID')
pearson_weights.to_csv('data/ours/weights/code2vec_pearsonCorrelation_Sp19.csv', index_label='ProblemID')


# %%
