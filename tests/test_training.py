import json

import pytest
import torch

from alignment.core import build_sft_batch, response_scores
from alignment.training import optimizer_step, run_training, validate
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_validation_uses_token_weighted_loss():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=1)
    batches = [build_sft_batch(tokenizer, ["Q:"], [r], 30) for r in ["a", "longer"]]
    scores = [response_scores(model, b) for b in batches]
    expected = -sum(s.sum().item() for s, c in scores) / sum(c.sum().item() for s, c in scores)
    result = validate(model, None, batches, {"stage": "sft", "device": "cpu"})
    assert result["validation_loss"] == pytest.approx(expected)
    assert model.training


def test_gradient_clipping_and_scheduler_follow_actual_optimizer_updates():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    initial = model.weight.detach().clone()
    (model.weight * 1000).sum().backward()
    assert optimizer_step(model, optimizer, scaler, .1)
    assert (model.weight - initial).abs().item() == pytest.approx(.1, abs=1e-6)
    assert model.weight.grad is None


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    path = tmp_path / "base"
    tokenizer_backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2, "###": 3,
                                           "Instruction:": 4, "Response:": 5, "Q": 6, "a": 7, "b": 8}, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=9, n_positions=64, n_embd=16, n_layer=1, n_head=2,
                                     resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., eos_token_id=1, pad_token_id=0))
    model.save_pretrained(path)
    return path


def configuration(tmp_path, tiny_checkpoint, stage="sft"):
    rows = [{"prompt": "Q", "response": "a b"} for _ in range(4)] if stage == "sft" else [{"prompt": "Q", "chosen": "a", "rejected": "b"} for _ in range(4)]
    data = tmp_path / f"{stage}_data.json"
    data.write_text(json.dumps(rows), encoding="utf-8")
    return {"stage": stage, "model": str(tiny_checkpoint), "reference_model": str(tiny_checkpoint),
            "train_path": str(data), "val_path": str(data), "output_dir": str(tmp_path / stage),
            "device": "cpu", "reference_device": "cpu", "seed": 42, "max_length": 32,
            "max_prompt_length": 16, "batch_size": 1, "gradient_accumulation": 3,
            "learning_rate": .001, "weight_decay": 0., "warmup_ratio": 0., "scheduler": "constant",
            "epochs": 1, "gradient_clip": 1., "eval_steps": 1, "beta": .1}


@pytest.mark.parametrize("stage", ["sft", "dpo"])
def test_real_tiny_model_training_saves_best_final_and_partial_accumulation(tmp_path, tiny_checkpoint, stage):
    config = configuration(tmp_path, tiny_checkpoint, stage)
    model = run_training(config)
    output = tmp_path / stage
    assert (output / "best" / "config.json").exists()
    assert (output / "final" / "config.json").exists()
    assert json.loads((output / "complete.json").read_text())["optimizer_steps"] == 2
    assert all(p.dtype == torch.float32 for p in model.parameters())
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    if stage == "dpo":
        assert metrics[0]["reward_accuracy"] == 0
        assert metrics[-1]["reward_margin"] > 0


def test_interrupted_training_resumes_to_same_weights(tmp_path, tiny_checkpoint, monkeypatch):
    import alignment.training as training
    config = configuration(tmp_path, tiny_checkpoint)
    original = training.validate
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated disconnect")
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "validate", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        run_training(config)
    monkeypatch.setattr(training, "validate", original)
    resumed = run_training(config, resume=True)
    config["output_dir"] = str(tmp_path / "uninterrupted")
    uninterrupted = run_training(config)
    for p, q in zip(resumed.parameters(), uninterrupted.parameters()):
        torch.testing.assert_close(p, q)


def test_generation_checkpoints_rows_and_rejects_changed_settings(tmp_path, tiny_checkpoint, monkeypatch):
    import sys
    from scripts import generate
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"id": str(i), "instruction": "Q", "dataset": "toy"} for i in range(3)]))
    output = tmp_path / "generations.json"
    arguments = ["generate", "--model", str(tiny_checkpoint), "--prompts", str(prompts),
                 "--output", str(output), "--generator", "tiny", "--device", "cpu", "--max-new-tokens", "6"]
    monkeypatch.setattr(sys, "argv", arguments)
    generate.main()
    original = output.read_text()
    assert len(json.loads(original)) == 3
    generate.main()
    assert output.read_text() == original
    arguments[-1] = "7"
    with pytest.raises(ValueError, match="changed"):
        generate.main()


def test_persona_trl_lora_training_saves_adapter(tmp_path, tiny_checkpoint, monkeypatch):
    import sys
    from scripts import train_persona
    data = tmp_path / "persona.json"
    data.write_text(json.dumps([{"prompt": "Q", "chosen": "a", "rejected": "b"} for _ in range(20)]))
    config = {"model": str(tiny_checkpoint), "data_path": str(data), "output_dir": str(tmp_path / "persona"),
              "seed": 42, "rank": 2, "alpha": 4, "dropout": 0., "target_modules": ["c_attn"],
              "learning_rate": .001, "batch_size": 1, "gradient_accumulation": 8, "epochs": 1, "beta": .1}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["train_persona", "--config", str(path)])
    train_persona.main()
    assert (tmp_path / "persona/best/adapter_model.safetensors").exists()
    complete = json.loads((tmp_path / "persona/complete.json").read_text())
    assert complete["best_checkpoint"] is not None
    assert (tmp_path / "persona/metrics.jsonl").exists()
