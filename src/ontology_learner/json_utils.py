import json

def parse_jsonl_file(jsonl_path, parser):
    """Reads and parses JSONL file, yielding documents."""
    with open(jsonl_path, 'r') as file:
        for line in file:
            try:
                parsed_entry = parser(line)
                yield parsed_entry
            except Exception as e:
                print(f"Error parsing line {line}: {e}")
                continue


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
