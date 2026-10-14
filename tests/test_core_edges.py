import math

import pytest
import torch

from alignment import core
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_eos_as_pad_keeps_real_eos_and_masks_only_padding():
    tokenizer = ToyTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    batch = core.build_sft_batch(tokenizer, ["Q", "Long"], ["a", "bc"], 20)
    assert batch["labels"][0, 2] == tokenizer.eos_token_id
    assert batch["labels"][0, 3:].eq(-100).all()
    assert batch["attention_mask"][0, 3:].eq(0).all()


@pytest.mark.parametrize("max_length, expected", [(2, []), (3, ["a"]), (4, ["a", "b"]), (5, ["a", "b", "<eos>"])])
def test_truncation_appends_eos_before_cutting(max_length, expected):
    tokenizer = ToyTokenizer()
    labels = core.build_sft_batch(tokenizer, ["Q:"], ["ab"], max_length)["labels"][0]
    assert labels[labels.ne(-100)].tolist() == [tokenizer.vocab[c] for c in expected]


def test_logprobs_match_manual_values_and_preserve_row_order():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=2)
    scores = core.compute_response_logprobs(model, tokenizer, ["Q:"] * 3, ["b", "aa", "z"], 20)
    probs = model.logit_bias.log_softmax(0)
    expected = torch.stack([sum(probs[tokenizer.vocab[c]] for c in r) + probs[1] for r in ["b", "aa", "z"]])
    torch.testing.assert_close(scores, expected)


def test_causal_shift_uses_previous_position_logits():
    class PositionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(10 * 80).float().reshape(10, 80) / 100)

        def forward(self, input_ids, attention_mask=None):
            from types import SimpleNamespace
            logits = self.weight[:input_ids.shape[1]].unsqueeze(0).expand(input_ids.shape[0], -1, -1)
            return SimpleNamespace(logits=logits)

    tokenizer, model = ToyTokenizer(), PositionModel()
    actual = core.compute_response_logprobs(model, tokenizer, ["Q:"], ["ab"], 8)[0]
    logps = model.weight.log_softmax(-1)
    expected = logps[1, tokenizer.vocab["a"]] + logps[2, tokenizer.vocab["b"]] + logps[3, 1]
    torch.testing.assert_close(actual, expected)


def test_no_eos_and_all_masked_rows_are_finite_zero():
    tokenizer = ToyTokenizer()
    tokenizer.eos_token_id = None
    model = ToyCausalLM(tokenizer.vocab_size)
    loss = core.compute_sft_loss(model, tokenizer, ["long prompt"], ["a"], 2)
    assert loss.item() == 0
    assert torch.isfinite(loss)


def test_dpo_matches_manual_objective_and_improves_after_step():
    pc = torch.tensor([-1., -4.], requires_grad=True)
    pr = torch.tensor([-3., -3.], requires_grad=True)
    rc = torch.tensor([-2., -3.], requires_grad=True)
    rr = torch.tensor([-2., -4.], requires_grad=True)
    actual = core.compute_dpo_loss_from_logps(pc, pr, rc, rr, .1)
    expected = (-torch.log(torch.sigmoid(torch.tensor([.2, -.2])))).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert rc.grad is None and rr.grad is None
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size)
    reference = ToyCausalLM(tokenizer.vocab_size)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1)
    args = (policy, reference, tokenizer, ["Q:"], ["a"], ["b"], .2, 10)
    before = core.compute_dpo_loss(*args)
    assert math.isclose(before.item(), math.log(2), rel_tol=1e-6)
    before.backward()
    optimizer.step()
    assert core.compute_dpo_loss(*args) < before
    assert reference.logit_bias.grad is None
