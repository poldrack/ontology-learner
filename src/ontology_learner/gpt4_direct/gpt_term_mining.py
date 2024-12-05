
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from ontology_learner.gpt4_batch_utils import get_batch_results, save_batch_results
from ontology_learner.json_utils import load_jsonl, parse_jsonl_results
from llm_query.chat_client import ChatClientFactory
from tqdm import tqdm
from pathlib import Path
import fasttext
import time
import hashlib
import secrets


def clean_task_name(taskname):
    taskname_clean = taskname.lower().replace('tasks', 'task')
    if '(' in taskname_clean:
        acronym = taskname_clean.split('(')[1].strip(')')
        taskname_clean = taskname_clean.split('(')[0].strip()
    else:
        acronym = []
    return taskname_clean, acronym

def clean_task_ontology(task_ontology):
    ontology_clean = {}
    for taskname, taskdict in task_ontology.items():
        taskdict = taskdict.copy()
        taskname_clean, acronym = clean_task_name(taskname)
        taskdict['acronym'] = acronym
        ontology_clean[taskname_clean] = taskdict
    return ontology_clean

def get_construct_task_dict_from_task_ontology(ontology_clean):
    constructs = {}
    for taskname, taskdict in ontology_clean.items():
        taskdict = taskdict.copy()
        for construct in taskdict['constructs']:
            if construct not in constructs:
                constructs[construct] = []
            constructs[construct].append(taskname)
    return constructs


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
but "n-back task" is not (it is a task, not a construct).  Include a 'type' key in your response with the value 'construct' if it is 
truly a psychological construct or 'other' if it is not.

If it is a psychological construct, please do the following:
- provide a short description of the construct.
- provide a short list of widely cited publications that describe the construct. Include a 
'references' key in your response with a list of the references.
- provide a list of commonly used tasks or surveys that measure the construct.  Include a 'tasks' key in your response with a list of the tasks.
- Provide a list of other constructs that are related to this construct.  Include a 'related_constructs' key in your response with a list of the related constructs.
Be as specific as possible, using names that are as specific as possible.

# RESPONSE #
Please return the results in JSON format.  Use the following keys:
- type: 'construct', or 'other'
- description: a short description of the construct
- references: a list of references that use the construct
- tasks: a list of tasks used to measure the construct
- related_constructs: a list of other constructs that are related to this construct
Respond only with JSON, without any additional text or description.
"""
    return prompt


def mk_batch_script(batchfile, termlist, prompt_func,
                    system_msg=None, custom_ids=None,
                    model='gpt-4o', overwrite=False):

    api_key = os.environ.get("OPENAI")
    client = OpenAI(api_key=api_key)

    if system_msg is None:
        system_msg = """
            You are an expert in psychology and neuroscience.
            You should be as specific and as comprehensive as possible in your responses.
            Your response should be a JSON object with no additional text.  
            """

    client = ChatClientFactory.create_client("openai", api_key, 
                                                system_msg=system_msg,
                                                model=model)

    if batchfile.exists() and overwrite:
        batchfile.unlink()
    elif batchfile.exists() :
        print(f'{batchfile} already exists and overwrite is False')
        return


    if isinstance(termlist, dict):

        termlist, ids = list(termlist.values()), list(termlist.keys())
    else:
        ids = termlist.copy()

    if custom_ids is None:
        custom_ids = ids
    
    for term, id in zip(termlist, custom_ids):

        prompt = prompt_func(term)
        kwargs = {'model': model,  'messages': [{"role": "user", "content": prompt}]}
        try:
            batch_request = client.create_batch_request(id, prompt)
        except Exception as e:
            print(f'error processing {term}: {e}')
            continue

        with open(batchfile, 'a') as f:
            f.write(json.dumps(batch_request) + '\n')
    print(f'wrote {batchfile}')


def run_batch_request(batchfile):
    api_key = os.environ.get("OPENAI")
    batch_client = OpenAI(api_key=api_key)

    batch_input_file = batch_client.files.create(file=open(batchfile, "rb"),
                                            purpose="batch")

    batch_input_file_id = batch_input_file.id

    batch_metadata = batch_client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "construct annotation"
        }
    )

    return batch_metadata, batch_client


def wait_for_batch_completion(batch_metadata, batch_client):
    print(batch_client.batches.retrieve(batch_metadata.id).status)
    while batch_client.batches.retrieve(batch_metadata.id).status != 'completed':
        time.sleep(60)
        print(batch_client.batches.retrieve(batch_metadata.id).status)


def get_main_construct_dict(construct_refinement_results, construct_task_dict):
    main_construct_dict = {}
    for k, v in construct_refinement_results.items():
        main_construct_dict[k.lower()] = v
    for construct, result in construct_refinement_results.items():
        for related_construct in result['related_constructs']:
            related_construct_clean = related_construct.lower()
            if related_construct_clean not in main_construct_dict:
                main_construct_dict[related_construct_clean] = {}
    return main_construct_dict


def get_task_cluster_prompt(task_list):
    return f"""
the following dictionary contains lists of psychological task or survey labels that were identified as 
being very similar and potentially referring to the same measure.  For example, "stop-signal task" 
and "stop signal task" refer to the same measure, whereas "color perception task" and 
"color generation task" refer to different tasks.  For each list, please determine whether the items 
refer to the same measure or different measures.  Each item includes the task name followed 
by a brief description of the task.

### LIST ###

{task_list}

# RESPONSE #
Please return the results in JSON format.  The result should be a dictionary.  Each element
of the dictionary should be a list of task names that belong to the same cluster.  The keys
should be a string that is a short label that summarizes the tasks in the cluster; if there is only
one task in the cluster, then use the task name.  Any proper nouns should be capitalized in the key.

Respond only with JSON, without any additional text or description.
"""