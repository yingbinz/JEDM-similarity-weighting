# %%
import pandas as pd
import tensorflow as tf
import numpy as np

tf.keras.utils.set_random_seed(42)

fn = "./example_data/ASSISTments_skill_builder_data.csv"
verbose = 1
best_model_weights = "weights/bestmodel"
log_dir = "logs"
optimizer = "adam"
lstm_units = 100
batch_size = 32
epochs = 10
dropout_rate = 0.3
test_fraction = 0.2
validation_fraction = 0.2
mask_value = -1.


# %%
# Load and prepare data
df = pd.read_csv(fn)

if "skill_id" not in df.columns:
    raise KeyError(f"The column 'skill_id' was not found on {fn}")
if "correct" not in df.columns:
    raise KeyError(f"The column 'correct' was not found on {fn}")
if "user_id" not in df.columns:
    raise KeyError(f"The column 'user_id' was not found on {fn}")

if not (df['correct'].isin([0, 1])).all():
    raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

# Remove questions without skill
df.dropna(subset=['skill_id'], inplace=True)

# Remove users with a single answer
df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

# Enumerate skill id
df['skill'], _ = pd.factorize(df['skill_id'], sort=True)

# Cross skill id with answer to form a synthetic feature
df['skill_with_answer'] = df['skill'] * 2 + df['correct']


# %%
########################################################
########################################################
# Group by student + offset by 1 timestep
X_S = df.groupby('user_id').apply(lambda row: row['skill_with_answer'].values[:-1])
y_S = df.groupby('user_id').apply(lambda row: (row['skill'].values[1:], row['correct'].values[1:]))

assert len(X_S) == len(y_S) # Sanity check
num_students = len(X_S)


# %%
X_depth = df['skill_with_answer'].max() + 1
skill_depth = df['skill'].max() + 1

# Encode X as one-hot
X_encoded = X_S.apply(lambda x: tf.keras.utils.to_categorical(x, num_classes=X_depth)).to_numpy()

# Encode y[0] (skill) as one-hot and concat with y[1] (correctness)
y_encoded = y_S.apply(lambda x: np.concatenate((
    tf.keras.utils.to_categorical(x[0], num_classes=skill_depth),
    np.expand_dims(x[1], axis=1)
    ), axis=1)).to_numpy()

# Shuffle X and y in unison
np.random.seed(42)
ix = np.random.permutation(num_students)
X_encoded = X_encoded[ix]
y_encoded = y_encoded[ix]

# Convert encoded arrays to lists of tensors
X_encoded_tensors = [tf.constant(i) for i in X_encoded]
y_encoded_tensors = [tf.constant(i) for i in y_encoded]

# Create tf dataset
ds = tf.data.Dataset.from_generator(
    generator = lambda: [(i, j) for i, j in zip(X_encoded_tensors, y_encoded_tensors)],
    output_types = (tf.float32, tf.float32)
)


# %%
# Pad sequences per batch
ds = ds.padded_batch(
    batch_size=batch_size,
    padding_values=(mask_value, mask_value),
    padded_shapes=([None, None], [None, None]),
    drop_remainder=True
)

length = num_students // batch_size


# %%
# Split data

def split(ds, split_size):
    split_set = ds.take(split_size)
    ds = ds.skip(split_size)
    return ds, split_set

if not 0 < test_fraction < 1:
    raise ValueError("test_fraction must be between (0, 1)")

if validation_fraction is not None and not 0 < validation_fraction < 1:
    raise ValueError("validation_fraction must be between (0, 1)")

test_size = np.ceil(test_fraction * length)
train_size = length - test_size

if test_size == 0 or train_size == 0:
    raise ValueError(
        "The train and test datasets must have at least 1 element. Reduce the split fraction or get more data.")

train_set, test_set = split(ds, test_size)

val_set = None
if validation_fraction:
    val_size = np.ceil(train_size * validation_fraction)
    train_set, val_set = split(train_set, val_size)


# %%
set_sz = length * batch_size
test_set_sz = (set_sz * test_fraction)
val_set_sz = (set_sz - test_set_sz) * validation_fraction
train_set_sz = set_sz - test_set_sz - val_set_sz
print("============= Data Summary =============")
print("Total number of students: %d" % set_sz)
print("Training set size: %d" % train_set_sz)
print("Validation set size: %d" % val_set_sz)
print("Testing set size: %d" % test_set_sz)
print("Number of skills: %d" % skill_depth)
print("Number of features in the input: %d" % X_depth)
print("========================================")


# %%
# Build model

inputs = tf.keras.Input(shape=(None, X_depth), name='inputs')

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
# Compile model with standard loss

# model.


# %%
# Train model

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks=[
    tf.keras.callbacks.CSVLogger(f"{log_dir}/train.log"),
    tf.keras.callbacks.ModelCheckpoint(
        best_model_weights,
        save_best_only=True,
        save_weights_only=True),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

history = model.fit(
    train_set,
    epochs=epochs,
    verbose=verbose,
    validation_data=val_set,
    callbacks=callbacks)


# %%
# Test best model

model.load_weights(best_model_weights)

result = model.evaluate(test_set, verbose=verbose)


# %%
