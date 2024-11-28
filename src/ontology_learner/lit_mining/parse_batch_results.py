import argparse
from pathlib import Path
import json


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
    parser.add_argument('-t', '--trackingfile', type=str, help='path to the tracking file')
    args = parser.parse_args()

    batchfile = Path(args.batchfile)

    tracking_file = args.trackingfile
    batchlabel = batchfile.stem

    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)

    batch_id = tracking_data[batchlabel]['id']

    outfile = batchfile.parent.parent / 'batch_results' / f'{batchlabel}.jsonl'
    print(f'parsing results for {batchfile.stem}')

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
