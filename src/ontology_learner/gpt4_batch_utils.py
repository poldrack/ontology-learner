def list_running_batches(client):
    batches = client.batches.list()
    for batch in batches:
        if batch.status in ['running', 'finalizing',
                            'validating', 'in_progress']:
            print(f'batch {batch.id} is {batch.status}')


def list_completed_batches(client):
    batches = client.batches.list()
    for batch in batches:
        if batch.status == 'completed':
            print(f'batch {batch.id} is completed')


def cancel_all_running_batches(client):
    batches = client.batches.list()
    for batch in batches:
        if batch.status == 'running':
            print(f'cancelling batch {batch.id}')
            client.batches.cancel(batch.id)


def get_batch_results(client, batch_id):
    md = client.batches.retrieve(batch_id)
    file_response = client.files.content(md.output_file_id)
    return file_response.text


def save_batch_results(batch_results, batch_id, outdir):
    outfile = outdir / f'{batch_id}.jsonl'
    with open(outfile, 'w') as f:
        f.write(batch_results)
    return outfile
