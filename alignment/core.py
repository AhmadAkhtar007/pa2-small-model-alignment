"""Response-only causal language-model objectives, implemented with PyTorch."""

import torch
import torch.nn.functional as F


def build_sft_batch(tokenizer, prompts, responses, max_length):
    if len(prompts) != len(responses) or not prompts:
        raise ValueError("Expected equal, nonempty prompt and response lists")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos if eos is not None else 0
    sequences, labels = [], []
    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        if eos is not None:
            response_ids = response_ids + [eos]
        sequences.append((prompt_ids + response_ids)[:max_length])
        labels.append(([-100] * len(prompt_ids) + response_ids)[:max_length])
    width = max(1, max(map(len, sequences)))
    return {
        "input_ids": torch.tensor([s + [pad] * (width - len(s)) for s in sequences]),
        "attention_mask": torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in sequences]),
        "labels": torch.tensor([s + [-100] * (width - len(s)) for s in labels]),
    }


def to_device(batch, model):
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}


def response_scores(model, batch):
    """Return summed log probabilities and scored token counts per example.

    Shift logits against next-token labels. Padding is identified by labels,
    never by token identity, so an EOS used as padding remains trainable.
    """
    batch = to_device(batch, model)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    targets = batch["labels"][:, 1:]
    mask = targets.ne(-100)
    # Cross entropy promotes logits to FP32 for stable AMP scoring without
    # materializing an extra full-vocabulary log-softmax tensor.
    token_nll = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), targets, ignore_index=-100, reduction="none"
    )
    return -(token_nll * mask).sum(dim=1), mask.sum(dim=1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return response_scores(model, build_sft_batch(tokenizer, prompts, responses, max_length))[0]


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    logps, counts = response_scores(model, build_sft_batch(tokenizer, prompts, responses, max_length))
    return -logps.sum() / counts.sum().clamp_min(1)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def compute_dpo_loss_from_logps(
    policy_chosen_logps, policy_rejected_logps,
    reference_chosen_logps, reference_rejected_logps, beta,
):
    reference_gap = (reference_chosen_logps - reference_rejected_logps).detach().to(policy_chosen_logps.device)
    policy_gap = policy_chosen_logps - policy_rejected_logps
    return -F.logsigmoid(beta * (policy_gap - reference_gap)).mean()


def dpo_scores(policy_model, reference_model, batch, beta):
    chosen = response_scores(policy_model, batch["chosen"])[0]
    rejected = response_scores(policy_model, batch["rejected"])[0]
    with torch.no_grad():
        ref_chosen = response_scores(reference_model, batch["chosen"])[0].to(chosen.device)
        ref_rejected = response_scores(reference_model, batch["rejected"])[0].to(chosen.device)
    loss = compute_dpo_loss_from_logps(chosen, rejected, ref_chosen, ref_rejected, beta)
    margin = beta * ((chosen - ref_chosen) - (rejected - ref_rejected))
    return loss, margin.detach()


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return dpo_scores(policy_model, reference_model, batch, beta)[0]
