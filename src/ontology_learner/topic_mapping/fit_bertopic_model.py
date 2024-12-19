# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
#

# %%
from dotenv import load_dotenv
from pathlib import Path
import os
import pickle
import json
load_dotenv()

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
#from umap import UMAP
#from hdbscan import HDBSCAN
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

datadir = Path(os.getenv('DATADIR'))
print(datadir)

device = 'cuda'

# %% [markdown]
# ### Load data
#

# %%

# write text to a json file for sentence transformers
fulltext_file = datadir / 'fulltext_sections.json'

with open(fulltext_file, 'r') as f:
    fulltext = json.load(f)

# Create training examples
sentences = []
sentence_keys = []
cutoff = 1e10 #200
minlength = 20

ctr = 0
for k, entry in fulltext.items():
    for sections in ['TITLE', 'INTRO', 'DISCUSS']:
        text = [i.lower().strip() for i in entry[sections].split('\n') if len(i) > minlength]
        sentences.extend(text)
        sentence_keys.extend([k] * len(text))
    ctr += 1
    if ctr > cutoff:
        break

print(f'found {len(sentences)} sentences')
assert len(sentences) == len(sentence_keys)

with open(datadir / 'sentence_keys_for_bertopic.json', 'w') as f:
    json.dump(sentence_keys, f)


# %% [markdown]
# ### Fit bertopic model

# %%
def get_embeddings(sentences, datdir, overwrite=False,
                   model_name='all-mpnet-base-v2', # 'all-MiniLM-L6-v2',
                   device=None):
    embedding_model = SentenceTransformer(model_name, device=None)
    if os.path.exists('data/embeddings.pkl') and not overwrite:
        print('using existing embeddings from data/embeddings.pkl')
        with open(datdir / 'embeddings_for_bertopic.pkl', 'rb') as f:
            embeddings = pickle.load(f)
    else:
        
        embeddings = embedding_model.encode(sentences, show_progress_bar=False)
        with open(datdir / 'embeddings_for_bertopic.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings, embedding_model


# %%

# Step 1 - Extract embeddings
model_name = ( datadir / 'embedding_models').as_posix()
embeddings, embedding_model = get_embeddings(sentences, datadir, model_name=model_name, overwrite=True, device=device)
embeddings = normalize(embeddings)

# %%
n_neighbors = 15
min_cluster_size = 50

# Step 2 - Reduce dimensionality
# ala https://maartengr.github.io/BERTopic/faq.html#i-have-too-many-topics-how-do-i-decrease-them
umap_model = UMAP(
    n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine'
)

# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(
    min_cluster_size=min_cluster_size,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
)

# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words='english')

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

# Step 6 - (Optional) Fine-tune topic representations with
# a `bertopic.representation` model
representation_model = KeyBERTInspired()


# %%
# All steps together

reduce_topics = True
if reduce_topics:
    nr_topics = 'auto'
else:
    nr_topics = None

topic_model = BERTopic(
    verbose=True,
    embedding_model=embedding_model,  # Step 1 - Extract embeddings
    umap_model=umap_model,  # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,  # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,  # Step 5 - Extract topic words
    representation_model=representation_model,  # Step 6 - Fine-tune topic represenations
    nr_topics=nr_topics,
    calculate_probabilities=True
)

topics, probs = topic_model.fit_transform(sentences)



# %%
topicmodeldir = datadir / f'topic_models/bertopic_intro-dicuss_nn-{n_neighbors}_minclust-{min_cluster_size}'
topicmodeldir.mkdir(exist_ok=True, parents=True)
topic_model.save(
    topicmodeldir.as_posix(),
    serialization='pytorch',
    save_ctfidf=True,
    save_embedding_model=True,
)


# %%
print(topic_model.get_topic_info())


# %%
