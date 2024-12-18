This project will use LLMs to learn an ontology from papers downloaded from pubmed central.

# Second pass refactor

I learned a couple of things from the first pass.  

- Many of the results from the LLM turn out to be very similar (e.g. "psychopathy" vs "psychopathic behavior", "stop-signal task" vs "stop-signal task (SST)").  In the first pass I did the harmonization late in the process, which led to unnecessary compute in earlier steps, so in the second pass I will move the harmonization to the second step after paper mining.
- I realized that I need unique identifiers for each concept/task earlier in the process, to make linking them more effective.  In the second pass I will assign each task/concept a semantically inert unique identifier (random hash) to make it easier to track them.
- I am going to add another 5000 papers to the literature base. (we have a much larger base, but the cost is about $34.50 per batch of 2500 so that would be expensive)


## Step 1: 

## Step 1: Download full text for fMRI papers

1a. Download the entire open access PMC database: [src/ontology_learner/lit_mining/get_all_bioc.sh](src/ontology_learner/lit_mining/get_all_bioc.sh).

1b. Using [src/ontology_learner/lit_mining/pmc_search.py]() we perform our query of PMC and
copy the matching files from the full set to `<datadir>/data/json`.

- searched on 11/16/2024, found 167447 results for query: brain AND open access[filter] AND "fMRI" OR "functional MRI" OR "functional magnetic resonance imaging"
- copied 132257 files
- missing 35190 files

## Step 2: Process papers

Using [src/ontology_learner/lit_mining/process_pmc_papers_batch.py]() and [src/ontology_learner/run_gpt4_batch.py](), run the full text of each paper through GPT-4o to annotate various aspects of the paper

This is performed using the OpenAI Batch API to save time and money.  Once the batch job has been completed, used [src/ontology_learner/lit_mining/download_batch_results.py]() to download the results and separate them into separate files for each paper, saved to `<datadir>/data/results_fulltext`.  A record of each run is stored in batch_tracking.json within the batch file input directory.


## Step 3: Extract and harmonize task and concept labels

### Aborted Step: cluster using document embeddings

We first tried to compute task and concept similarity using embeddings for the documents. In `src/ontology_learner/lit_mining/get_fulltext_embeddings.ipynb` we compared several possible embeddings, and found that fasttext embeddings generated from the full text corpus were highly correlated in terms of representational similarity with OpenAI embeddings on a subset of documents, so we will use those given that they are both free and reproducible.  These are stored in `<datadir>/fasttext_embeddings_fulltext.npy`.  However, clustering on these embeddings did not produce particularly coherent clusters of concepts, so we abandoned this and used GPT-4 instead (as we had with success in Pass 1).

### Step 3a: Create functions to work with the task and concept sets.

-   `refinement_utils.load_original_results` - loads results from original concept/task extraction

### Step 3c: Clustering of concepts/tasks

Using the mean embeddings, perform agglomerative clustering on the concepts/tasks in order to find similar items for further assessment using GPT-4.

### Step 3d: Refine concepts/tasks

Using the results from 3c, create a refined set of concepts/tasks that combine together the results from the initial annotation (e.g. conditions, contrasts, references, etc).


## Step 4: Linkage of tasks and concepts

Using the refined concepts/tasks, find all concept/task pairs that overlap within at least one document.


# Original pass

## Step 1: Download full text for fMRI papers

First tried to download individual papers using BioC API, but it was flaky, so decided
to download the full open access set, see `src/ontology_learner/lit_mining/get_all_bioc.sh`


Using `src/ontology_learner/lit_mining/pmc_search.py` we then perform our query of PMC and
copy the matching files from the full set to <datadir>/data/json.  

- searched on 11/16/2024, found 167447 results for query: brain AND open access[filter] AND "fMRI" OR "functional MRI" OR "functional magnetic resonance imaging"
- copied 132257 files
- missing 35190 files

## Step 2: Process papers

Using `src/ontology_learner/lit_mining/process_pmc_papers_batch.py` and `src/ontology_learner/run_gpt4_batch.py`, run the full text of each paper through GPT-4o to annotate various aspects of the paper. Here is the prompt:

```
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
- which disorders are discussed in relation to the construct (e.g., "autism")

Be as specific as possible, using names that are as specific as possible.

# RESPONSE #
Please return the results in JSON format.  Use the following keys:
- construct: a list of strings that represent the constructs discussed in the text
- task: a list of strings that represent the tasks or surveys used to measure the constructs
- condition: a dict keyed by the task name containing a list of strings that represent the experimental conditions used within that task
- contrast: a dict keyed by the task name containing a list of strings that represent the contrasts made between experimental conditions within that task
- brain_region: a dict keyed by the task name containing a list of strings that represent the brain regions discussed in relation to that task
- disorder: a list of strings that represent the disorders discussed in the text

Respond only with JSON, without any additional text or description.

```

This is performed using the OpenAI Batch API to save time and money.  Once the batch job has been completed, used `src/ontology_learner/lit_mining/download_batch_results.py` to download the results and separate them into separate files for each paper, saved to <datadir>/data/results_fulltext.

## Step 3: Task annotation

Using all of the tasks identified in Step 2, we ask GPT-4o for additional information; see `src/ontology_learner/annotation/task_annotation_batch.py`.  Here is the prompt:

```
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

```

results are stored into <datadir>/data/task_results

## Step 4: Task refinement

The tasks returned by the annotation had many instances of the same task with slightly different labels. We used a combination of clustering and GPT-4 to identify sets of labels that mapped to the same underlying term.

This was performed using `src/ontology_learner/refinement/task_refinement.ipynb`


## Step 5: Concept refinement

A similar refinment was performed for concepts, using `src/ontology_learner/refinement/concept_refinement.ipynb`


## Step 6: Combination of refined tasks and concepts

