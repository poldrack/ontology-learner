from openai import OpenAI
import os
import argparse
import json
from pathlib import Path

if __name__ == '__main__':

    # create a cli to take in the batchfile
    parser = argparse.ArgumentParser(
        description='Run a batch of requests with GPT-4')
    parser.add_argument('-b', '--batchfile', type=str, help='path to the batchfile')
    args = parser.parse_args()

    batchfile = Path(args.batchfile)
    tracking_file = batchfile.parent / 'batch_tracking.json'
    batchlabel = batchfile.stem

    api_key = os.getenv('OPENAI')
    client = OpenAI(api_key=api_key)

    batch_input_file = client.files.create(file=open(batchfile, "rb"),
                                           purpose="batch")

    batch_input_file_id = batch_input_file.id

    batch_metadata = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "annotation of PMC papers"
        }
    )

    # add the batch id to the tracking file
    if tracking_file.exists():
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    tracking_data[batchlabel] = {
        'batchfile': args.batchfile,
        'id': batch_metadata.id,
        'status': batch_metadata.status,
        'input_file_id': batch_input_file_id,
        'output_file_id': batch_metadata.output_file_id
    }
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f, indent=4)

    print(client.batches.retrieve(batch_metadata.id))
