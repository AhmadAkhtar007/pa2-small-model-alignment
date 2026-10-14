"""Normalize source datasets into explicit single-turn examples."""

import random


def normalize_sft(row):
    messages = row["messages"]
    if len(messages) < 2 or [m["role"] for m in messages[:2]] != ["user", "assistant"]:
        return None
    return {"prompt": messages[0]["content"].strip(), "response": messages[1]["content"].strip()}


def normalize_uf(row):
    chosen, rejected = row["chosen"], row["rejected"]
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if any([m["role"] for m in side] != ["user", "assistant"] for side in (chosen, rejected)):
        return None
    if chosen[0]["content"] != rejected[0]["content"]:
        return None
    return {"prompt": chosen[0]["content"].strip(), "chosen": chosen[1]["content"].strip(), "rejected": rejected[1]["content"].strip()}


def normalize_hh(row):
    def parse(text):
        if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
            return None
        if not text.startswith("\n\nHuman:"):
            return None
        prompt, response = text[len("\n\nHuman:"):].split("\n\nAssistant:")
        return prompt.strip(), response.strip()

    chosen, rejected = parse(row["chosen"]), parse(row["rejected"])
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    return {"prompt": chosen[0], "chosen": chosen[1], "rejected": rejected[1]}


def split_rows(rows, train_size, val_size, seed):
    unique = {}
    for row in rows:
        if row and all(isinstance(v, str) and v.strip() for v in row.values()):
            unique.setdefault(row["prompt"].strip(), row)
    values = list(unique.values())
    random.Random(seed).shuffle(values)
    if len(values) < train_size + val_size:
        raise ValueError(f"Need {train_size + val_size} distinct prompts; only {len(values)} survived filtering")
    return values[:train_size], values[train_size:train_size + val_size]


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"
