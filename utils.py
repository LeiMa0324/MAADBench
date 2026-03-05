from datetime import datetime
import json

def generate_trace_id(dataset, mas_arch, query_id, label, temp = 0):
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    trace_id  = f"{time}_{dataset}_{mas_arch}_{query_id}_temp_{temp}_{label}"
    return trace_id

import re

def extract_json(text: str) -> dict | None:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    # 去掉 ```json ... ``` 或 ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # 去掉末尾的 ```
    text = text.replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 最后尝试：找第一个 { 到最后一个 }
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
    return None

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item))
            f.write("\n")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]