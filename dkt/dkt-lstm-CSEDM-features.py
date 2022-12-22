# %%
import pandas as pd
import tensorflow as tf
import numpy as np

tf.keras.utils.set_random_seed(42)

verbose = 1
best_model_weights = "weights/bestmodel-CSEDM"
log_dir = "logs"
# learning_rate = 1e-6
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
optimizer = 'adam'
lstm_units = 32
batch_size = 32
epochs = 50
dropout_rate = 0.3
validation_fraction = 0.2
mask_value = -1.


# %%
# Load data
early_train_S = pd.read_csv('../data/S19/ALL/early.csv')
early_train_F = pd.read_csv('../data/F19/Train/early.csv')

late_train_S = pd.read_csv('../data/S19/All/late.csv')
late_train_F = pd.read_csv('../data/F19/Train/late.csv')

early_test_F = pd.read_csv('../data/F19/Test/early.csv')
late_test_F = pd.read_csv('../data/F19/Test/late.csv')

early = pd.concat([early_train_F, early_train_S, early_test_F], axis = 0)
late = pd.concat([late_train_F, late_train_S], axis = 0)

# Format df
df = pd.concat([early, late], axis=0).reset_index(drop=True)

# Create ProblemID mapping
pIDs = []
for a in sorted(df['AssignmentID'].unique()):
    for p in sorted(df[df['AssignmentID'] == a]['ProblemID'].unique()):
        pIDs.append(p)
pIDs = {v:k for k, v in enumerate(pIDs)}


# %%
new_train = pd.read_csv('../data/SAKT/train.csv')
new_test = pd.read_csv('../data/SAKT/test.csv')

df = pd.concat([new_train, new_test], axis=0).reset_index(drop=True)
# df = df.sort_values(by='AttemptID')

df['skill'] = df['ProblemID'].apply(lambda x: pIDs[x])

df = df[['SubjectID', 'skill', 'Label']]
df.columns = ['user_id', 'skill', 'correct']

# Remove users with a single answer
df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

# Cross skill id with answer to form a synthetic feature
df['skill_with_answer'] = df['skill'] * 2 + df['correct']


# %%
########################################################
# Get feature data
df_train = pd.read_csv('../data/ours/X_train_no_weight.csv')
df_test = pd.read_csv('../data/ours/X_test_no_weight.csv')

# Add subject and problem information
df_train = pd.concat([late.reset_index(drop=True)[['SubjectID', 'ProblemID', 'Label']], df_train.drop(['AssignmentID'], axis=1)], axis=1)
df_test = pd.concat([late_test_F.reset_index(drop=True)[['SubjectID', 'ProblemID', 'Label']], df_test.drop(['AssignmentID', 'Label'], axis=1)], axis=1)

# Remove rows with missing data
df_train = df_train.dropna()
# df_test = df_test.dropna()

# Remove users with a single answer
df_train = df_train.groupby('SubjectID').filter(lambda q: len(q) > 1).copy()

df_train = df_train.rename(columns={'SubjectID':'user_id', 'ProblemID':'skill', 'Label':'correct'})
df_test = df_test.rename(columns={'SubjectID':'user_id', 'ProblemID':'skill', 'Label':'correct'})

# Enumerate problem IDs
df_train['skill'] = df_train['skill'].apply(lambda x: pIDs[x]-30)
df_test['skill'] = df_test['skill'].apply(lambda x: pIDs[x]-30)

# Normalize continuous features
X_S = df_train.iloc[:,3:].apply(lambda c: (c - c.mean())/c.std())
X_test_S = df_test.iloc[:,3:].apply(lambda c: (c - c.mean())/c.std())

# Add user_id
X_S['user_id'] = df_train['user_id']
X_test_S['user_id'] = df_test['user_id']

# Group by student + offset by 1 timestep
X_S = X_S.groupby('user_id').apply(lambda row: row.drop('user_id', axis=1)[:-1].values)
X_test_S = X_test_S.groupby('user_id').apply(lambda row: row.drop('user_id', axis=1)[:-1].values)

y_S = df_train.groupby('user_id').apply(lambda row: (row['skill'].values[1:], row['correct'].values[1:]))
y_test_S = df_test.groupby('user_id').apply(lambda row: (row['skill'].values[1:], row['correct'].values[1:]))


assert len(X_S) == len(y_S) # Sanity check
num_students = len(X_S)


# %%
# Get unique students for appropriate shuffling and splitting
uStudents = pd.Series(X_S.index.unique())

# Shuffle unique students
np.random.seed(42)
uStudents_shuffled = np.random.permutation(uStudents)

# Get validation students
val_size = np.ceil(len(uStudents) * validation_fraction)
uStudents_val = uStudents_shuffled[:int(val_size)]

# Split train/val sets
uStudents_val_S = pd.Series(X_S.index).apply(lambda x: x in tuple(uStudents_val))
uStudents_val_S.index = X_S.index
X_val_S = X_S[uStudents_val_S]
y_val_S = y_S[uStudents_val_S]
X_train_S = X_S[~uStudents_val_S]
y_train_S = y_S[~uStudents_val_S]


# %%
skill_depth = df_train['skill'].max() + 1
features_depth = X_train_S[0].shape[1] + skill_depth - 1


# Create padded batches
def padded_batch(X, y, test=False):
    # Encode skill as one-hot in X
    # X_encoded = X.apply(lambda x: tf.keras.utils.to_categorical(x, num_classes=skill_depth)).to_numpy()

    X_encoded = X.apply(lambda x: np.concatenate((
        tf.keras.utils.to_categorical(x[:,0], num_classes=skill_depth),
        x[:,1:]
    ), axis=1)).to_numpy()

    # Encode y[0] (skill) as one-hot and concat with y[1] (correctness)
    y_encoded = y.apply(lambda x: np.concatenate((
        tf.keras.utils.to_categorical(x[0], num_classes=skill_depth),
        np.expand_dims(x[1], axis=1)
        ), axis=1)).to_numpy()

    # Shuffle X and y in unison (if training set)
    if not test:
        np.random.seed(42)
        ix = np.random.permutation(len(X))
        X_encoded = X_encoded[ix]
        y_encoded = y_encoded[ix]

    # Convert to lists of padded tensors
    X_encoded_tensors = [
        tf.pad(
            tf.constant(i),
            tf.constant([[0, 20-len(i)], [0, 0]]), # Should be 19?
            constant_values=mask_value
        ) for i in X_encoded]
    y_encoded_tensors = [
        tf.pad(
            tf.constant(i),
            tf.constant([[0, 20-len(i)], [0, 0]]),
            constant_values=mask_value
        ) for i in y_encoded]

    # Create empty y tensor if test set to ensure no real y data gets passed
    if test:
        y_encoded_tensors = [np.zeros(i.shape) for i in y_encoded]

    # Create tf dataset
    ds = tf.data.Dataset.from_generator(
        generator = lambda: [(i, j) for i, j in zip(X_encoded_tensors, y_encoded_tensors)],
        output_types = (tf.float32, tf.float32)
    )

    # Pad sequences per batch
    ds = ds.padded_batch(
        batch_size=batch_size,
        padding_values=(mask_value, mask_value),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=False
    )

    return ds


# %%
train_set = padded_batch(X_train_S, y_train_S, test=False)
val_set = padded_batch(X_val_S, y_val_S, test=False)
test_set = padded_batch(X_test_S, y_test_S, test=True)

length = num_students // batch_size


# %%
set_sz = length * batch_size
val_set_sz = set_sz * validation_fraction
train_set_sz = set_sz - val_set_sz
print("============= Data Summary =============")
print("Total number of students: %d" % set_sz)
print("Training set size: %d" % train_set_sz)
print("Validation set size: %d" % val_set_sz)
print("Number of skills: %d" % skill_depth)
print("Number of features in the input: %d" % features_depth)
print("========================================")


# %%
# Build model

inputs = tf.keras.Input(shape=(None, features_depth), name='inputs')

x = tf.keras.layers.Masking(mask_value=mask_value)(inputs)

x = tf.keras.layers.LSTM(lstm_units,
                            return_sequences=True,
                            dropout=dropout_rate)(x)

dense = tf.keras.layers.Dense(skill_depth, activation='sigmoid')
outputs = tf.keras.layers.TimeDistributed(dense, name='outputs')(x)

model = tf.keras.Model(inputs, outputs)


# %%
def get_target(y_true, y_pred):
    # Get skills and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true, mask_value), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred

class AUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(AUC, self).update_state(
            y_true=true,
            y_pred=pred,
            sample_weight=sample_weight)

def custom_loss(y_true, y_pred):
    y_true, y_pred = get_target(y_true, y_pred)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


# %%
# Compile model

model.compile(
    loss = custom_loss,
    optimizer = optimizer,
    metrics = [
        AUC()
    ]
)

model.summary()


# %%
# Train model

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks=[
    tf.keras.callbacks.CSVLogger(f"{log_dir}/train.log"),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=6),
    tf.keras.callbacks.ModelCheckpoint(
        best_model_weights,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_auc"),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

history = model.fit(
    train_set,
    epochs=epochs,
    verbose=verbose,
    validation_data=val_set,
    callbacks=callbacks)


# %%
######################################################################
######################################################################
# Test best model

model.load_weights(best_model_weights)

# result = model.evaluate(test_set, verbose=verbose)


# %%
preds = model.predict(test_set, verbose=verbose)

preds_df = pd.DataFrame(
    [i[-1] for i in preds],
    columns = range(0,20))
preds_df.insert(0, 'SubjectID', X_test_S.index)


# %%
# Melt to create one row per student-problem
preds_df = preds_df.melt(id_vars=['SubjectID']).sort_values(by=['SubjectID', 'variable']).reset_index(drop=True)
preds_df = preds_df.rename(columns={'variable':'ProblemID', 'value':'pred'})


# %%
# preds_df.to_csv('../data/Prediction/DKT-LSTM-df.csv', index=False)


# # %%
# preds_df = pd.read_csv('../data/Prediction/DKT-LSTM-df.csv')

y_true = pd.read_csv('../data/F19/Test/late.csv')
# Enumerate ProblemID to match preds
y_true['ProblemID'] = y_true['ProblemID'].apply(lambda x: pIDs[x]-30)

# Count trues
print(f"total: {len(preds_df)}")
print(f"predicted trues: {preds_df['pred'].round().sum()}")

comp_df = y_true.merge(preds_df, how='left', on=['SubjectID', 'ProblemID'])


# %%
from sklearn.metrics import roc_auc_score

roc_auc_score(comp_df['Label'], comp_df['pred'])


# %%
