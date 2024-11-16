# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
# add autoreload
# %load_ext autoreload
# %autoreload 2

import json
import os
from pathlib import Path
from llm_query.chat_client import ChatClientFactory
from publication import Publication

def get_prompt(text):
    return f"""
# CONTEXT #
Researchers in the field of cognitive neuroscience and psychology study specific 
psychological constructs, which are the building blocks of the mind, such as 
memory, attention, theory of mind, and so on.  To study them, researchers use 
experimental tasks or surveys, which are meant to measure behavior related to the 
constructs.  In many cases psychological tasks have several different experimental
conditions, which are meant to manipulate the construct in different ways.  These are 
commonly compared to one another in order to measure the effect of the manipulation; we 
refer to these comparisons as contrasts.

# TEXT #
Here is the text of the publication:

{text}

# OBJECTIVE #
Your job is to analyze the text to identify:
- which psychological constructs are discussed in the text (e.g., "theory of mind")
- which tasks or surveys are used to measure them (e.g., "picture story task")
- which experimental conditions are used to manipulate the construct (e.g., "false belief")
- which contrasts are made between experimental conditions (e.g., "false belief vs. true belief")
- which brain regions are discussed in relation to the construct (e.g., "left prefrontal cortex")

Be as specific as possible, using names that are as specific as possible.

# RESPONSE #
Please return the results in JSON format.  Use the following keys:
- construct: a list of strings that represent the constructs discussed in the text
- task: a list of strings that represent the tasks or surveys used to measure the constructs
- condition: a dict keyed by the task name containing a list of strings that represent the experimental conditions used within that task
- contrast: a dict keyed by the task name containing a list of strings that represent the contrasts made between experimental conditions within that task
- brain_region: a dict keyed by the task name containing a list of strings that represent the brain regions discussed in relation to that task
Respond only with JSON, without any additional text or description.
"""

def parse_pmcid_json(pmcid, datadir):

    pub = Publication(pmcid, datadir=datadir)
    pub.load_json()
    pub.parse_sections()    
    return pub.combine_text()


if __name__ == '__main__':
    api_key = os.environ.get("ANTHROPIC")

    datadir = Path('/Users/poldrack/data_unsynced/ontology-learner/data/json')
    outdir_fulltext = Path('/Users/poldrack/data_unsynced/ontology-learner/data/results_fulltext')

    if not outdir_fulltext.exists():
        outdir_fulltext.mkdir(parents=True)

    datafiles = list(datadir.glob('*.json'))
    datafiles.sort(reverse=True)

    print(f'found {len(datafiles)} data files')


    system_msg = """
    You are an expert in psychology and neuroscience.
    You should be as specific and as comprehensive as possible in your responses.
    Your response should be a JSON object with no additional text.
    """

    client = ChatClientFactory.create_client("anthropic", api_key, system_msg=system_msg)

    for file in datafiles:
        pmcid = file.stem
        text = parse_pmcid_json(pmcid, datadir)
        outfile = outdir_fulltext / f'{pmcid}.json'
        # check for existing file
        if outfile.exists():
            print(f'{pmcid} already processed')
            continue    

        response = client.chat(get_prompt(text))

        # save results to file
        with open(outfile, 'w') as f:
            json.dump(json.loads(response), f, indent=4)



