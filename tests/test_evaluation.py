import json

import pytest

from alignment.evaluation import build_pairs, judge_pairs, judge_prompt, summarize
from alignment.io import read_rows


def outputs(name):
    return [{"id": str(i), "instruction": "Q", "output": f"{name} answer {i}", "generator": name, "dataset": "toy"} for i in range(3)]


def test_pairs_preserve_identity_but_judge_prompt_hides_it():
    pairs = build_pairs(outputs("sft"), outputs("dpo"), 42)
    assert pairs == build_pairs(outputs("sft"), outputs("dpo"), 42)
    for pair in pairs:
        prompt = judge_prompt([pair], "general")
        # Names in response text are content; metadata must never be serialized.
        assert "generator" not in prompt and "model_a" not in prompt
        assert pair["model_a"] != pair["model_b"]
    with pytest.raises(ValueError):
        build_pairs(outputs("sft"), outputs("dpo")[:-1], 42)


def test_judge_retries_checkpoints_and_resumes(tmp_path):
    pairs = build_pairs(outputs("sft"), outputs("dpo"), 42)
    path = tmp_path / "judgments.jsonl"
    attempts, sleeps = [], []
    def request(prompt):
        attempts.append(prompt)
        if len(attempts) == 1:
            raise RuntimeError("temporary denial")
        ids = [r["id"] for r in json.loads(prompt.split("INPUT JSON:\n")[1])]
        return json.dumps([{"id": i, "winner": "A", "reason": "better"} for i in ids])
    judge_pairs(pairs, path, request, "fake", batch_size=2, sleep_seconds=5, max_attempts=3, sleep=sleeps.append)
    assert len(read_rows(path)) == 3
    assert len(attempts) == 3 and sleeps == [5, 5]
    judge_pairs(pairs, path, request, "fake", sleep=sleeps.append)
    assert len(attempts) == 3


def test_invalid_decisions_are_unknown_and_failures_not_wins(tmp_path):
    pairs = build_pairs(outputs("sft"), outputs("dpo"), 42)
    path = tmp_path / "judgments.jsonl"
    judge_pairs(pairs, path, lambda _: "bad json", "fake", max_attempts=1, sleep=lambda _: None)
    result = summarize(pairs, [], read_rows(path.with_suffix(".failures.jsonl")), "dpo")
    assert result["failures"] == 3 and result["wins"] == 0
    assert result["win_rate_excluding_ties"] is None


def test_resume_rejects_changed_pairs(tmp_path):
    pairs = build_pairs(outputs("sft"), outputs("dpo"), 42)
    path = tmp_path / "judgments.jsonl"
    def request(prompt):
        rows = json.loads(prompt.split("INPUT JSON:\n")[1])
        return json.dumps([{"id": r["id"], "winner": "tie"} for r in rows])
    judge_pairs(pairs, path, request, "fake", sleep=lambda _: None)
    pairs[0]["response_a"] += " changed"
    with pytest.raises(ValueError, match="changed"):
        judge_pairs(pairs, path, request, "fake", sleep=lambda _: None)


def test_persona_scores_are_mapped_to_model_identity(tmp_path):
    pairs = build_pairs(outputs("sft"), outputs("dpo"), 42)
    def request(prompt):
        rows = json.loads(prompt.split("INPUT JSON:\n")[1])
        return json.dumps([{"id": r["id"], "winner": "tie", "style_a": 2, "style_b": 4,
                            "usefulness_a": 3, "usefulness_b": 3, "safe_a": True, "safe_b": False} for r in rows])
    rows = judge_pairs(pairs, tmp_path / "judgments.jsonl", request, "fake", mode="persona", sleep=lambda _: None)
    result = summarize(pairs, rows, [], "dpo")
    sides = ["a" if p["model_a"] == "dpo" else "b" for p in pairs]
    assert result["persona_ratings"]["dpo"]["mean_style"] == pytest.approx(sum(2 if s == "a" else 4 for s in sides) / 3)
    assert result["persona_ratings"]["dpo"]["safe_fraction"] == pytest.approx(sides.count("a") / 3)
