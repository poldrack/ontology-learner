# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Use llm to annotate the topics

#

from bertopic import BERTopic
from pathlib import Path
import os
import openai
from bertopic.representation import OpenAI
import json
from sentence_transformers import SentenceTransformer
import pandas as pd
import dotenv
import numpy as np
import torch
import argparse


def main(n_neighbors, min_cluster_size, reduce_topics, cutoff):
    
    dotenv.load_dotenv()

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = None

    datadir = Path(os.getenv('DATADIR'))
    print(datadir)

    if reduce_topics:
        reduce_topics_flag = 'auto'
    else:
        reduce_topics_flag = 'none'

    print(f'using device: {device}')
    # load model
    embedding_model_dir = datadir / 'embedding_models'
    def get_embedding_model(model_name='all-mpnet-base-v2', 
                            device=None):
        return SentenceTransformer(model_name, device=None)

    embedding_model = get_embedding_model(embedding_model_dir.as_posix(), device)


    topicmodeldir = datadir / f'topic_models/bertopic_intro-dicuss_' \
                                f'nn-{n_neighbors}_minclust-{min_cluster_size}_' \
                                f'cutoff-{cutoff}_reduce-{reduce_topics_flag}'

    topic_model = BERTopic.load(topicmodeldir.as_posix(), embedding_model=embedding_model)

    topic_model.get_topic_info()

    # load sentences
    fulltext_file = datadir / 'fulltext_sections.json'

    with open(fulltext_file, 'r') as f:
        fulltext = json.load(f)

    # Create training examples
    sentences = []
    sentence_keys = []
    minlength = 20
    cutoff = None

    ctr = 0
    for k, entry in fulltext.items():
        for sections in ['TITLE', 'INTRO', 'DISCUSS']:
            text = [i.lower().strip() for i in entry[sections].split('\n') if len(i) > minlength]
            sentences.extend(text)
            sentence_keys.extend([k] * len(text))
        ctr += 1
        if cutoff is not None and ctr > cutoff:
            break

    print(f'found {len(sentences)} sentences')
    assert len(sentences) == len(sentence_keys)

    # use custom prompt that calls for a shorter summary
    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]
    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short topic label of four words or less in the following format:
    topic: <topic label>
    """
    llm = 'llama3'

    if llm == 'gpt4':
        openai_client = openai.Client(api_key=os.getenv('OPENAI'))
    elif llm == 'llama3':
        #openai_client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        from together import Together

        openai_client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

    llmnames = {'gpt4': 'gpt-4', 
                'llama3': "meta-llama/Llama-3.3-70B-Instruct-Turbo"}
    llmname = llmnames[llm]

    representation_model = OpenAI(
        client=openai_client,
        model=llmname, chat=True, exponential_backoff=True,
        prompt=prompt
    )

    topic_model.update_topics(sentences, representation_model=representation_model)


    modeldir_llm = topicmodeldir.as_posix() + f'{llmname}' 
    topics, probs = topic_model.transform(sentences)
    output_dir_llm = Path(modeldir_llm)

    np.save(output_dir_llm / 'probs.npy', probs)
    df = pd.DataFrame({"Document": sentences, "Topic": topics}) # , 'Probs': probs})
    topic_model.save(
        output_dir_llm.as_posix(),
        serialization='pytorch',
        save_ctfidf=True,
        save_embedding_model=False,
    )

    df.to_csv(output_dir_llm / 'topic_probs.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate a BERTopic model.')
    parser.add_argument('--n_neighbors', type=int, default=15, help='Number of neighbors for UMAP.')
    parser.add_argument('--min_cluster_size', type=int, default=50, help='Minimum cluster size for HDBSCAN.')
    parser.add_argument('--cutoff', type=float, default=None, help='Cutoff for the number of entries to process.')
    parser.add_argument('--no_reduce', action='store_false', dest='reduce_topics', help='Flag to not reduce topics automatically.')

    args = parser.parse_args()
    main(args.n_neighbors, args.min_cluster_size, args.reduce_topics, args.cutoff)

