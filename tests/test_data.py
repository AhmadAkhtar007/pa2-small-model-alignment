from alignment.data import normalize_hh, normalize_sft, normalize_uf, split_rows


def test_hh_keeps_only_matching_single_turn_pairs():
    row = {"chosen": "\n\nHuman: Help me\n\nAssistant: Sure", "rejected": "\n\nHuman: Help me\n\nAssistant: No"}
    assert normalize_hh(row) == {"prompt": "Help me", "chosen": "Sure", "rejected": "No"}
    assert normalize_hh({**row, "rejected": "\n\nHuman: Other\n\nAssistant: No"}) is None
    assert normalize_hh({k: v + "\n\nHuman: Again\n\nAssistant: No" for k, v in row.items()}) is None


def test_sft_extracts_first_turn_and_uf_checks_same_prompt():
    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    assert normalize_sft({"messages": messages + messages}) == {"prompt": "Q", "response": "A"}
    assert normalize_uf({"chosen": messages, "rejected": [messages[0], {"role": "assistant", "content": "B"}]}) == {"prompt": "Q", "chosen": "A", "rejected": "B"}


def test_split_is_deterministic_and_removes_duplicate_prompts():
    rows = [{"prompt": str(i)} for i in range(10)] + [{"prompt": "0"}]
    train, val = split_rows(rows, 5, 3, 42)
    assert (train, val) == split_rows(rows, 5, 3, 42)
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in val}
