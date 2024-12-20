# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#

from dotenv import load_dotenv
from pathlib import Path
import os
import pickle
import json

import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
if torch.cuda.is_available():
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
    from cuml.preprocessing import normalize
else:
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

import argparse

def get_embeddings(sentences, datadir, overwrite=False,
                    model_name='all-mpnet-base-v2', # 'all-MiniLM-L6-v2',
                    device=None):
    embedding_file = datadir / 'embeddings_for_bertopic.pkl'
    embedding_model = SentenceTransformer(model_name, device=None)
    try:
        assert os.path.exists(embedding_file), f'{embedding_file} does not exist'
        assert not overwrite, f'{embedding_file} already exists but overwrite is True'
        print('using existing embeddings from data/embeddings.pkl')
        with open(embedding_file, 'rb') as f:
            embeddings = pickle.load(f)
        assert embeddings.shape[0] == len(sentences), f'{embeddings.shape[0]} != {len(sentences)}'
    except:
        embeddings = embedding_model.encode(sentences, show_progress_bar=False)
        with open(embedding_file, 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings, embedding_model

def main(n_neighbors, min_cluster_size, reduce_topics, cutoff):
    load_dotenv()
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = None

    datadir = Path(os.getenv('DATADIR'))
    print(datadir)

    print(f'using device: {device}')

    # ### Load data

    fulltext_file = datadir / 'fulltext_sections.json'

    with open(fulltext_file, 'r') as f:
        fulltext = json.load(f)

    # Create training examples
    sentences = []
    sentence_keys = []
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

 
    # ### Fit bertopic model


    # Step 1 - Extract embeddings
    model_name = ( datadir / 'embedding_models').as_posix()
    embedding_file = datadir / 'embeddings_for_bertopic.pkl'
    #model_name = '/Users/poldrack/data_unsynced/ontology_learner/embedding_models'
    embeddings, embedding_model = get_embeddings(sentences, datadir,
                                                 model_name=model_name, 
                                                 overwrite=True, 
                                                 device=device)
    embeddings = normalize(embeddings)

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

    # All steps together

    if reduce_topics:
        nr_topics = 'auto'
        reduce_topics_flag = 'auto'
    else:
        nr_topics = None
        reduce_topics_flag = 'none'

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

    topicmodeldir = datadir / f'topic_models/bertopic_intro-dicuss_' \
                                f'nn-{n_neighbors}_minclust-{min_cluster_size}_' \
                                f'cutoff-{cutoff}_reduce-{reduce_topics_flag}'
    topicmodeldir.mkdir(exist_ok=True, parents=True)
    topic_model.save(
        topicmodeldir.as_posix(),
        serialization='pytorch',
        save_ctfidf=True,
        save_embedding_model=True,
    )

    with open(topicmodeldir / 'sentence_keys_for_bertopic.json', 'w') as f:
        json.dump(sentence_keys, f)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(topicmodeldir / 'topic_info.csv')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit a BERTopic model.')
    parser.add_argument('--n_neighbors', type=int, default=15, help='Number of neighbors for UMAP.')
    parser.add_argument('--min_cluster_size', type=int, default=50, help='Minimum cluster size for HDBSCAN.')
    parser.add_argument('--cutoff', type=float, default=1e10, help='Cutoff for the number of entries to process.')
    parser.add_argument('--no_reduce', action='store_false', dest='reduce_topics', help='Flag to not reduce topics automatically.')

    args = parser.parse_args()
    main(args.n_neighbors, args.min_cluster_size, args.reduce_topics, args.cutoff)



