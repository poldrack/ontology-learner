import json


def parse_jsonl_file(jsonl_path, parser=None):
    """Reads and parses JSONL file, yielding documents."""
    with open(jsonl_path, "r") as file:
        for line in file:
            try:
                parsed_entry = parser(line)
                yield parsed_entry
            except Exception as e:
                print(f"Error parsing line {line}: {e}")
                continue


def parse_jsonl_results(results_raw):
    results_content = {}
    for result in results_raw:
        #task = result["custom_id"].split("-")[1]
        task = result["custom_id"]
        content = result["response"]["body"]["choices"][0]["message"]["content"]
        content = content.replace("json", "").replace("```", "").replace("\n", "")
        try:
            results_content[task] = json.loads(content)
        except json.JSONDecodeError:
            print(f"error decoding {task}")
    return results_content


def parse_jsonl_task_line(line):
    """Parses a single line of JSONL into a dictionary."""
    json_data = json.loads(line.strip())

    content = json_data["response"]["body"]["choices"][0]["message"]["content"]
    # clean up content
    content = content.replace("```json", "").replace("```", "")
    content_dict = json.loads(content)
    assert (
        content_dict["type"] != "other"
    ), f"type is other for {json_data['custom_id']}"
    content_dict["type"] = content_dict["type"].replace("task-", "")
    content_dict["type"] = content_dict["type"].replace("_", " ")
    content_dict["description"] = "".join(content_dict["description"])
    content_dict["model"] = json_data["response"]["body"]["model"]
    content_dict["custom_id"] = (
        json_data["custom_id"].replace("construct-", "").replace("/", "_")
    )
    content_dict["system_fingerprint"] = json_data["response"]["body"][
        "system_fingerprint"
    ]
    return content_dict


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data
