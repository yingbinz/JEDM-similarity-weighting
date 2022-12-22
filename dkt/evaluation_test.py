'''
There seems to be a discrepancy between the way TF is calculating AUC on the test data vs sklearn's approach. I'm still not sure what's going on here.
'''
# %%
from sklearn.metrics import roc_auc_score


# %%
model.load_weights(best_model_weights)
result = model.evaluate(test_set, verbose=verbose)
print(result)

# %%
# Convert test_set to list
test_X = []
test_y = []
for i in test_set:
    test_X.append(i[0])
    test_y.append(i[1])

# Shape = (batches, students per batch (32), attempts (padded to max), feature/skill depth)

# %%
# %%
for i in range(len(test_X)):
    pred = model(test_X[i], training=False)
    y_true, y_pred = get_target(test_y[i], pred)

    auc = tf.keras.metrics.AUC()
    auc.update_state(y_true, y_pred)
    print(i, auc.result().numpy())
    break

# %%
preds = model.predict(test_set, verbose=verbose, steps=1)

y_true, y_pred = get_target(test_y[i], preds)

auc = tf.keras.metrics.AUC()
auc.update_state(y_true, y_pred)
print(i, auc.result().numpy())


# %%
scores = []
for batch in range(len(test_X)):
    y_pred = model(test_X[batch], training=False)
    y_true, y_pred = get_target(test_y[batch], y_pred)

    scores.append([])
    for i in range(32):
        # Remove dimension
        temp_true = y_true.numpy()[i][:,0]
        temp_pred = y_pred.numpy()[i][:,0]

        # Remove mask
        temp_true = temp_true[:(temp_pred != 0).sum()]
        temp_pred = temp_pred[:(temp_pred != 0).sum()]

        if len(pd.Series(temp_true).unique()) > 1:
            score = roc_auc_score(temp_true, temp_pred)

            # auc = tf.keras.metrics.AUC()
            # auc = AUC()
            # auc.update_state(temp_true, temp_pred)
            # score = auc.result().numpy()

            # print(batch, i, score)
            scores[batch].append(score)
    print(batch, np.mean(scores[batch]))

print(f"\nfinal mean:{np.mean([np.mean(i) for i in scores])}")


# %%
