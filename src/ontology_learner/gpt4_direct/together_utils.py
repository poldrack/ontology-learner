import json
from together import Together


def chat(messages, model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
         seed=None, temperature=None, max_tokens=32000):
    result = Together().chat.completions.create(
        model=model,
        messages=messages,
        # response_format={"type": "json_object"},
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        )
    try:
        return json.loads(result.choices[0].message.content.replace("```json", "").replace("```", ""))
    except Exception as e:
        raise e
