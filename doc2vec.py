# %%
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

# Import prompts
prompts = pd.read_csv('data/ours/given_KCs.csv')
prompts = prompts[['ProblemID', 'Requirement']]
prompts.index = prompts.index + 1
prompts.columns = ['ProblemID', 'raw']


# %%
# Tokenize
prompts['tokenized'] = prompts['raw'].apply(lambda x: word_tokenize(x.lower()))

# Remove stop words
stop_words = set(stopwords.words('english'))
prompts['tokenized'] = prompts['tokenized'].apply(lambda x: [w for w in x if not w in stop_words])


# %%
# Tag late prompts
prompts['tagged'] = prompts.apply(lambda i: TaggedDocument(words=i['tokenized'], tags=[str(i['ProblemID'])]), axis=1)


# %%
# Create and train model
model = Doc2Vec(vector_size=50, min_count=2, epochs=120)
model.random.seed(42)

# # Train using only late problem prompts
# model.build_vocab(prompts.loc[31:]['tagged'])
# model.train(prompts.loc[31:]['tagged'], total_examples=model.corpus_count, epochs=model.epochs)

# Train using all problem prompts
model.build_vocab(prompts['tagged'])
model.train(prompts['tagged'], total_examples=model.corpus_count, epochs=model.epochs)


# %%
# # Calculate vectors for early prompts
# model.random.seed(42)
# prompts['vectors'] = prompts['tokenized'].apply(lambda x: model.infer_vector(x, steps=1000, alpha=0.01))

# prompts['vectors'].head()


# %%
# Identify most similar late problems
# prompts['mostSimilar'] = prompts['vectors'].apply(lambda x: model.docvecs.most_similar([x], topn=20))


# %%
def inverse_euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2)) ** 1

def inverse_euclidean_distances(x):
    distances = []
    for p in prompts['ProblemID']:
        distances.append((str(p), inverse_euclidean_distance(x, model.docvecs[str(p)])))
    return distances

def pearson_correlations(x):
    correlations = []
    for p in prompts['ProblemID']:
        correlations.append((str(p), np.corrcoef(x, model.docvecs[str(p)])[0, 1]))
    return correlations

# Calculate similarities based on docvecs embeddings
# Cosine similarity
prompts['cosineSimilarity'] = prompts['ProblemID'].apply(lambda x: model.docvecs.most_similar([model.docvecs[str(x)]], topn=50))

# Inverse Euclidean distance
prompts['inverseEuclideanDistance'] = prompts['ProblemID'].apply(lambda x: inverse_euclidean_distances(model.docvecs[str(x)]))

# Pearson correlation
prompts['pearsonCorrelation'] = prompts['ProblemID'].apply(lambda x: pearson_correlations(model.docvecs[str(x)]))


# %%
def get_weights(measure):
    # Create weights dataframe (late instances, early columns)
    prompt_weights_df = pd.DataFrame()
    for i in range(1, 31):
        p = prompts.loc[i]['ProblemID']
        weights = pd.Series({k: v for k, v in prompts[measure].loc[i]}, name=i)
        prompt_weights_df[p] = weights

    prompt_weights_df.index = prompt_weights_df.index.astype('int')
    # prompt_weights_df = prompt_weights_df.sort_index()
    prompt_weights_df = prompt_weights_df.loc[prompts.ProblemID.loc[31:].values]
    prompt_weights_df.columns = sorted(prompt_weights_df.columns)

    return prompt_weights_df.T

cosine_weights = get_weights('cosineSimilarity')
euclidean_weights = get_weights('inverseEuclideanDistance')
pearson_weights = get_weights('pearsonCorrelation')


# %%
# prompt_weights_df.to_csv('data/ours/prompt_weights.csv', index_label='late')


# %%
# New code for final run

# Transpose to match weighting function
# prompt_weights_df = prompt_weights_df.T

cosine_weights.to_csv('data/ours/weights/prompts_doc2vec_cosineSimilarity.csv', index_label='ProblemID')
euclidean_weights.to_csv('data/ours/weights/prompts_doc2vec_inverseEuclideanDistance.csv', index_label='ProblemID')
pearson_weights.to_csv('data/ours/weights/prompts_doc2vec_pearsonCorrelation.csv', index_label='ProblemID')


# %%
