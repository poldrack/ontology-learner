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
import json
import numpy as np
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from chromadb.utils import embedding_functions
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset

load_dotenv()

device = 'mps'
datadir = Path(os.getenv('DATADIR'))
print(datadir)


# %% [markdown]
# ### Load data
#

# %%

# load jsonl format
fulltext_file = datadir / 'fulltext_sections.json'
flat_file = datadir / 'fulltext_sections_flat.json'

if not flat_file.exists():
    with open(fulltext_file, 'r') as f:
        dataset_orig = json.load(f)

    sections = ['INTRO', 'ABSTRACT', 'CONCL', 
                'TITLE', 'DISCUSS', 'RESULTS', 
                'METHODS']

    dataset_flat = {}
    for k, v in dataset_orig.items():
        text = ' '.join([v[section].lower() for section in sections])
        dataset_flat[k] = text

    with open(flat_file, 'w') as f:
        json.dump(dataset_flat, f)

else:
    print(f'Loading flat dataset from {flat_file}')
    with open(fulltext_file, 'r') as f:
        dataset_orig = json.load(f)
    with open(flat_file, 'r') as f:
        dataset_flat = json.load(f)

# create training examples
# ala https://www.ionio.ai/blog/fine-tuning-embedding-models-using-sentence-transformers-code-included
# for each section, create a training example with the 'query' key set to the text from the "DISCUSS" section,
#  the 'pos' key set to the text from the "TITLE" section, and the 'neg' key set to the 'TITLE' section
# from a randomly selected other line

# %% setup vector database


# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
dbdir = Path('/Users/poldrack/data_unsynced/ontology_learner/chroma_db')
dbdir.mkdir(exist_ok=True)

# use mac GPU via 'mps' device
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2", device=device)

client = chromadb.PersistentClient(path=dbdir.as_posix())

# remove collection if it exists
if "fmripapers" not in [c.name for c in client.list_collections()]:

    collection = client.create_collection(
        "fmripapers",
        embedding_function=sentence_transformer_ef
    )

    for k, v in tqdm(dataset_flat.items()):
        collection.add(documents=[v], ids=[k])
else:
    print(f'Loading collection from {dbdir}')
    collection = client.get_collection("fmripapers")


# %%

# Load the dataset

# Create training examples
train_examples = []
test_examples = []
ctr = 0
train_cutoff = 250

# some pilot tests showed that predicting results from intro was hardest
query_section = 'INTRO'
target_section = 'RESULTS'
for k, v in dataset_orig.items():
    query = v[query_section].lower()
    pos = v[target_section].lower()
        
    # Select a close example from the database
    results = collection.query(
        query_texts=[dataset_flat[k]],
        n_results=2,
    )
    neg_key = results['ids'][0][1]
    neg = dataset_orig[neg_key][target_section].lower()

    if ctr < train_cutoff:
        #train_examples.append(InputExample(texts=[query,pos,neg]))
        train_examples.append({'anchor': query, 'positive': pos, 'negative': neg})
    else:
        #test_examples.append(InputExample(texts=[query,pos,neg]))
        test_examples.append({'anchor': query, 'positive': pos, 'negative': neg})
    ctr += 1
    if ctr > (train_cutoff * 2):
        break



# %%
# fit the model
fit_model = True
encoder_model_name = 'all-mpnet-base-v2'
n_train_epochs = 1
batch_size = 64

model = SentenceTransformer(encoder_model_name, device=device)
#train_dataloader = DataLoader(train_examples, batch_size=64, shuffle=True, drop_last=True)
# train_loss = losses.TripletLoss(model)
train_loss = losses.MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="/Users/poldrack/data_unsynced/ontology_learner/checkpoints",
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    num_train_epochs=n_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=Dataset.from_list(train_examples),
    eval_dataset=Dataset.from_list(test_examples),
    loss=train_loss,
)
trainer.train()


# %%
# test on similarity of titles from intro

def get_class_accuracy(model, test_examples):
    test_dists = []
    for example in test_examples:
        anchor_embed = model.encode(example['anchor'])
        positive_embed = model.encode(example['positive'])
        negative_embed = model.encode(example['negative'])
        test_dists.append([np.linalg.norm(anchor_embed - positive_embed), np.linalg.norm(anchor_embed - negative_embed)])

    dist_array = np.array(test_dists)
    dist_diff = dist_array[:,0] - dist_array[:,1]
    print(f'mean accuracy: {np.mean(dist_diff < 0)}')

model_pretrained = SentenceTransformer(encoder_model_name, device=device)
print(f'pretrained model ({query_section} -> {target_section}):')
get_class_accuracy(model_pretrained, test_examples)
#print(f'finetuned model ({query_section} -> {target_section}):')
#get_class_accuracy(model, test_examples)

# %%
model_output_dir = datadir / 'embedding_models'
model_output_dir.mkdir(exist_ok=True)
model.save(model_output_dir.as_posix())



# %%
