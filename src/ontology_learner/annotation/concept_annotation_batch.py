# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# add autoreload
# %load_ext autoreload
# %autoreload 2

import json
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from llm_query.chat_client import ChatClientFactory
from openai import OpenAI
api_key = os.environ.get("OPENAI")


def lower_list(l):
    return [x.lower() for x in l]

# %%
@dataclass
class PaperResults:
    results_dir: Path
    output_dir: Path
    
    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.constructs = []
        self.tasks = []
        self.conditions = {}
        self.contrasts = {}
        self.brain_regions = []
        self.results_files = []

    # get results files
    def get_results_files(self):
        self.results_files = list(self.results_dir.glob('*.json'))
        self.results_files.sort()
        print(f'found {len(self.results_files)} results files')

    # load data files
    def parse_results_files(self):

        for filename  in self.results_files:
            pmcid = filename.stem
            with open(filename, 'r') as f:
                paper_results = json.load(f)
            if 'construct' in paper_results:
                self.constructs.extend(lower_list(paper_results['construct']))
            if 'task' in paper_results:
                self.tasks.extend(lower_list(paper_results['task']))
            # these are dicts
            if 'condition' in paper_results:
                for task, value in paper_results['condition'].items():
                    if len(value) == 0:
                        continue
                    if task in self.conditions:
                        self.conditions[task].extend(value)
                    else:
                        self.conditions[task] = value
            if 'contrast' in paper_results:
                for task, value in paper_results['contrast'].items():
                    if len(value) == 0:
                        continue
                    if task in self.contrasts:
                        self.contrasts[task].extend(value)
                    else:
                        self.contrasts[task] = value
            if 'brain_region' in paper_results:
                self.brain_regions.extend(paper_results['brain_region'])
        # remove duplicates
        self.constructs = list(set(self.constructs))
        self.tasks = list(set(self.tasks))
        self.brain_regions = list(set(self.brain_regions))



# %%
datadir = Path('/Users/poldrack/Dropbox/code/ontology_learner/data/json')
outdir_fulltext = Path('/Users/poldrack/Dropbox/data/ontology-learner/data/results_fulltext')

results = PaperResults(results_dir=outdir_fulltext, output_dir=outdir_fulltext)
results.get_results_files()
results.parse_results_files()


# %% [markdown]
# ### Clean up results
#
# create a batch job for gpt4 to clean up the different results 

# %%
system_msg = """
You are an expert in psychology and neuroscience.
You should be as specific and as comprehensive as possible in your responses.
Your response should be a JSON object with no additional text.  
"""

client = ChatClientFactory.create_client("openai", api_key, 
                                            system_msg=system_msg,
                                            model="gpt-4o")


# %%
def get_construct_prompt(construct):
    prompt = f"""
# CONTEXT #
Researchers in the field of cognitive neuroscience and psychology study specific 
psychological constructs, which are the building blocks of the mind, such as 
memory, attention, theory of mind, and so on.  

# OBJECTIVE #
Your job is to analyze a specific construct: {construct}.

- You should first determine whether it is truly a psychological construct, or whether it is some 
other kind of thing.  For example, "working memory" is a psychological construct, 
but "n-back task" is not.  Include a 'type' key in your response with the value 'construct' if it is 
truly a psychological construct or 'other' if it is not.

If it is a psychological construct, please do the following:
- provide a short description of the construct.
- provide a short list of widely cited publications that describe the construct. Include a 
'references' key in your response with a list of the references.
- provide a list of commonly used tasks or surveys that measure the construct. 
Include a 'tasks' key in your response with a list of the tasks.

Be as specific as possible, using names that are as specific as possible.

# RESPONSE #
Please return the results in JSON format.  Use the following keys:
- type: 'construct', or 'other'
- description: a short description of the construct
- references: a list of references that use the construct
- tasks: a list of tasks used to measure the construct

Respond only with JSON, without any additional text or description.
"""
    return prompt



# %% [markdown]
# Make a batch job for gpt4 to clean up the different results 

# %%
concept_batch_file = outdir_fulltext.parent / 'concept_batch.jsonl'
if concept_batch_file.exists():
    concept_batch_file.unlink()

ids = []
for construct in results.constructs:
    prompt = get_construct_prompt(construct)
    id = f'construct-{construct.replace(" ", "_")}'.lower()
    if id in ids:
        print(f"skipping duplicate construct {construct}")
        continue
    ids.append(id)
    batch_request = client.create_batch_request(id, prompt)
    with open(concept_batch_file, 'a') as f:
        f.write(json.dumps(batch_request) + '\n')



# %%

valid_jsonl = True
good_lines = []
bad_lines = []
with open(concept_batch_file, 'rb') as file:
    for line_num, line in enumerate(file, 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        try:
            json.loads(line)
            good_lines.append(line)
        except json.JSONDecodeError:
            print(f"Invalid JSON on line {line_num}: {line}")
            bad_lines.append(line)

with open(concept_batch_file, 'wb') as file:
    for line in good_lines:
        file.write(line + b"\n")

print(len(good_lines), "valid JSON lines written to", concept_batch_file)



# %%
batch_client = OpenAI(api_key=api_key)


batch_input_file = batch_client.files.create(file=open(concept_batch_file, "rb"),
                                        purpose="batch")

batch_input_file_id = batch_input_file.id

batch_metadata = batch_client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "annotation of constructs"
    }
)


# %%
print(batch_client.batches.retrieve(batch_metadata.id).status)
import time
while batch_client.batches.retrieve(batch_metadata.id).status != 'completed':
    time.sleep(60)
    print(batch_client.batches.retrieve(batch_metadata.id).status)
os.system('say "your program has finished"')

# %%
# NOTE: This step is run once the batch job has completed.
from gpt4_batch_utils import get_batch_results, save_batch_results
batch_results = get_batch_results(batch_client, batch_metadata.id)
outdir = outdir_fulltext.parent / 'concept_results'
outdir.mkdir(exist_ok=True, parents=True)

outfile = save_batch_results(batch_results, batch_metadata.id, outdir)
# %%

# process the results


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

results_raw = load_jsonl(outfile)

results_content = {}
for result in results_raw:
    task = result['custom_id'].split('-')[1]
    content = result['response']['body']['choices'][0]['message']['content']
    content = content.replace('json', '').replace("```", "").replace("\n", "")
    try:
        results_content[task] = json.loads(content)
    except json.JSONDecodeError:
        print(f"error decoding {task}")

# %%

content_by_type= defaultdict(dict)

for content_type in ['task', 'survey', 'other']:
    content_by_type[content_type] = {
        k: v
        for k, v in results_content.items() 
        if 'type' in v
        and v['type'] == content_type
    }

    print(f"{content_type}: {len(content_by_type[content_type])}")
# %%
