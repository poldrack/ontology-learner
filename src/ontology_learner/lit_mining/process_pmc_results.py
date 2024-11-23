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
                self.constructs.extend(paper_results['construct'])
            if 'task' in paper_results:
                self.tasks.extend(paper_results['task'])
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
datadir = Path('/Users/poldrack/data_unsynced/ontology-learner/data/json')
outdir_fulltext = Path('/Users/poldrack/data_unsynced/ontology-learner/data/results_fulltext')

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
def get_task_prompt(task):
    prompt = f"""
# CONTEXT #
Researchers in the field of cognitive neuroscience and psychology study specific 
psychological constructs, which are the building blocks of the mind, such as 
memory, attention, theory of mind, and so on.  To study them, researchers use 
experimental tasks or surveys, which are meant to measure behavior related to the 
constructs.  These are experiment measures in which an individual is presented with
stimuli and behavior is measured.  A measure is referred to as a "survey" if it involves
presenting specific questions to the participant and recording their self-reported responses. 
For example, the "Beck Depression Inventory" is a survey that measures depression.  
Any other kind of psychological measure that involves presenting stimuli and recording behavior
is referred to as a "task".  For example, the "go/no-go task" is a task that involves 
presenting stimuli and recording whether the participant makes a response or not on each trial.


# OBJECTIVE #
Your job is to analyze a specific term: {task}.

- You should first determine whether it is truly a psychological task or survey, or whether it is some other 
kind of thing.  For example, "picture story task" is a psychological task, but 'magnetic resonance imaging' and 'EEG'
are measurement tools that do not directly measure behavior or psychological constructs.  Include a 'type' key in your
response with the value 'task', 'survey', or 'other' depending on your determination.  If the term is generic, such as "follow-up survey", 
then set the type to 'other'.
- if it is a psychological task or survey, please provide a short description of the task or survey, in a key called 'description'
- If it is a psychological task or survey, you should next determine which psychological constructs
are measured by the task or survey.  For example, "picture story task" measures theory of mind.  Include a 'constructs' key in your
response with a list of the constructs measured by the task or survey.
- If it is a psychological task or survey, you should next provide a short list of widely cited publications that use the task or survey.
Include a 'references' key in your response with a list of the references.
- If it is a psychological task or survey, you should next provide a list of commonly used experimental conditions the task or survey. 
Include a 'conditions' key in your response with a list of the conditions.
- If it is a psychological task or survey, you should next provide a list of disorders that are commonly studied in relation to the task or survey.
Include a 'disorders' key in your response with a list of the disorders.

Be as specific as possible, using names that are as specific as possible.

# RESPONSE #
Please return the results in JSON format.  Use the following keys:
- type: 'task', 'survey', or 'other'
- description: a short description of the task or survey
- constructs: a list of the constructs measured by the task or survey
- references: a list of references that use the task or survey
- conditions: a list of the experimental conditions used in the task or survey
- disorders: a list of the disorders that are commonly studied in relation to the task or survey

Respond only with JSON, without any additional text or description.
"""
    return prompt



# %% [markdown]
# Make a batch job for gpt4 to clean up the different results 

# %%
task_batch_file = outdir_fulltext.parent / 'task_batch.jsonl'
for task in results.tasks:
    prompt = get_task_prompt(task)
    id = f'task-{task.replace(" ", "_")}'
    batch_request = client.create_batch_request(id, prompt)
    with open(task_batch_file, 'a') as f:
        f.write(json.dumps(batch_request) + '\n')



# %%

valid_jsonl = True
good_lines = []
bad_lines = []
with open(task_batch_file, 'rb') as file:
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

print(len(good_lines), len(bad_lines))


# %%
line

# %%
batch_client = OpenAI(api_key=api_key)


batch_input_file = batch_client.files.create(file=open(task_batch_file, "rb"),
                                        purpose="batch")

batch_input_file_id = batch_input_file.id

batch_metadata = batch_client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "annotation of tasks"
    }
)

print(batch_client.batches.retrieve(batch_metadata.id))



# %%
batch_input_file

# %%
constructs = set()
tasks = set()
conditions = {}
contrasts = {}
brain_regions = set()

for filename  in results:
    pmcid = filename.stem
    with open(filename, 'r') as f:
        paper_results = json.load(f)
    if 'construct' in paper_results:
        constructs.update(paper_results['construct'])
    if 'task' in paper_results:
        tasks.update(paper_results['task'])
    # these are dicts
    if 'condition' in paper_results:
        for task, value in paper_results['condition'].items():
            if len(value) == 0:
                continue
            if condition in conditions:
                conditions[task].extend(value)
            else:
                conditions[task] = value
    if 'contrast' in paper_results:
        for task, value in paper_results['contrast'].items():
            if len(value) == 0:
                continue
            if task in contrasts:
                contrasts[task].extend(value)
            else:
                contrasts[task] = value
    if 'brain_region' in paper_results:
        brain_regions.update(paper_results['brain_region'])

len(constructs)



# %%
constructs

# %%
[c for c in constructs if ' and ' in c]

# %%
