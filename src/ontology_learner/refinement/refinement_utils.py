import contextlib
import io
from ontology_learner.json_utils import parse_jsonl_file, parse_jsonl_task_line


def load_original_results(results_dir):

    concept_jsonl = list(results_dir.glob('batch*.jsonl'))[0]

    # context manager to suppress print statements
    with contextlib.redirect_stdout(io.StringIO()) as f:
        concepts = list(parse_jsonl_file(concept_jsonl, parse_jsonl_task_line))
    print(f'Loaded {len(concepts)} concepts from {concept_jsonl}')

    return{t['custom_id']: t for t in concepts}
