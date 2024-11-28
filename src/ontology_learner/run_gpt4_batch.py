from openai import OpenAI
import os
import argparse
import json
from pathlib import Path
import time
from gpt4_batch_utils import get_batch_results, save_batch_results

if __name__ == "__main__":
    # create a cli to take in the batchfile
    parser = argparse.ArgumentParser(description="Run a batch of requests with GPT-4")
    parser.add_argument("-b", "--batchfile", type=str, help="path to the batchfile")
    parser.add_argument("-d", "--desc", type=str, help="description of the batch")
    parser.add_argument(
        "-o", "--outdir", type=str, help="output directory", default="batch_results"
    )

    args = parser.parse_args()

    batchfile = Path(args.batchfile)
    tracking_file = batchfile.parent / "batch_tracking.json"
    batchlabel = batchfile.stem

    api_key = os.getenv("OPENAI")
    client = OpenAI(api_key=api_key)

    batch_input_file = client.files.create(file=open(batchfile, "rb"), purpose="batch")

    batch_input_file_id = batch_input_file.id

    batch_metadata = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": args.desc},
    )

    # add the batch id to the tracking file
    if tracking_file.exists():
        with open(tracking_file, "r") as f:
            tracking_data = json.load(f)
    else:
        tracking_data = {}

    tracking_data[batchlabel] = {
        "batchfile": args.batchfile,
        "id": batch_metadata.id,
        "description": args.desc,
        "status": batch_metadata.status,
        "input_file_id": batch_input_file_id,
        "output_file_id": batch_metadata.output_file_id,
    }
    with open(tracking_file, "w") as f:
        json.dump(tracking_data, f, indent=4)

    print(client.batches.retrieve(batch_metadata.id))

    while client.batches.retrieve(batch_metadata.id).status != "completed":
        print(client.batches.retrieve(batch_metadata.id).status)
        time.sleep(60)
    try:
        os.system("say 'your program has finished'")
    except:  # noqa: E722
        pass

    batch_results = get_batch_results(client, batch_metadata.id)
    outdir = batchfile.parent.parent / args.outdir
    outdir.mkdir(exist_ok=True, parents=True)

    outfile = save_batch_results(batch_results, args.batchfile.stem, outdir)
