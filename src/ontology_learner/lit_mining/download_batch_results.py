from openai import OpenAI
import argparse
from pathlib import Path
import json
import os
from gpt4_batch_utils import get_batch_results, save_batch_results


def parse_batch_results(outfile, results_dir):
    with open(outfile, 'r') as f:
        for line in f:
            data = json.loads(line)
            paper_id = data['id']
            paper_text = data['text']
            outfile = results_dir / f'{paper_id}.txt'
            with open(outfile, 'w') as f:
                f.write(paper_text)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Download results from a batch of GPT-4 requests')
    parser.add_argument('-b', '--batchfile', type=str, help='path to the batchfile')
    args = parser.parse_args()

    batchfile = Path(args.batchfile)

    api_key = os.getenv('OPENAI')
    client = OpenAI(api_key=api_key)


    tracking_file = batchfile.parent / 'batch_tracking.json'
    batchlabel = batchfile.stem

    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)

    batch_id = tracking_data[batchlabel]['id']

    outfile = batchfile.parent.parent / 'batch_results' / f'{batchlabel}.jsonl'
    print(f'downloading results to {outfile}')
    if not outfile.exists():
        batch_results = get_batch_results(client, batch_id)
        outdir =  batchfile.parent.parent / 'batch_results'
        if not outdir.exists():
            outdir.mkdir()
        save_batch_results(batch_results, batchlabel, outdir)
    else:
        print(f'results already downloaded for batch {batchlabel}')
    # 

    # parse the results into separate files
    results_dir = batchfile.parent.parent / 'results_fulltext'

    with open(outfile, 'r') as f:
        for line in f:
            data = json.loads(line)
            paper_id = data['custom_id']
            result = data['response']['body']['choices'][0]['message']['content']
            # clean up the result
            result = result.replace('```json', '').replace('```', '')
            try:
                result_json = json.loads(result)
            except json.JSONDecodeError:
                print(f'error parsing result for {paper_id}')
                continue
            outfile = results_dir / f'{paper_id}.json'
            with open(outfile, 'w') as f:
                json.dump(result_json, f, indent=4)
