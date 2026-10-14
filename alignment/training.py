"""Manual SFT and DPO loops; no Trainer or TRL dependency."""

import contextlib
import itertools
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alignment.core import build_dpo_batch, build_sft_batch, dpo_scores, response_scores
from alignment.data import format_prompt
from alignment.io import append_jsonl, file_hash, read_rows, write_json


def prompt_text(tokenizer, instruction, max_prompt_length):
    text = format_prompt(instruction)
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Keep the instruction tail and the complete response delimiter.
    return tokenizer.decode(ids[-max_prompt_length:], skip_special_tokens=False)


def collate_rows(rows, tokenizer, config):
    prompts = [prompt_text(tokenizer, r["prompt"], config["max_prompt_length"]) for r in rows]
    if config["stage"] == "sft":
        return build_sft_batch(tokenizer, prompts, [r["response"] for r in rows], config["max_length"])
    return build_dpo_batch(tokenizer, prompts, [r["chosen"] for r in rows], [r["rejected"] for r in rows], config["max_length"])


def batch_size(batch, stage):
    return (batch if stage == "sft" else batch["chosen"])["input_ids"].shape[0]


def token_count(batch):
    return batch["labels"][:, 1:].ne(-100).sum().item()


def amp_context(device):
    return torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda") else contextlib.nullcontext()


@torch.no_grad()
def validate(model, reference, loader, config):
    model.eval()
    total_loss, total_count, wins, margins = 0., 0, 0, 0.
    with amp_context(config["device"]):
        for batch in loader:
            if config["stage"] == "sft":
                scores, counts = response_scores(model, batch)
                total_loss += -scores.sum().item()
                total_count += counts.sum().item()
            else:
                loss, margin = dpo_scores(model, reference, batch, config["beta"])
                count = len(margin)
                total_loss += loss.item() * count
                total_count += count
                wins += margin.gt(0).sum().item()
                margins += margin.sum().item()
    model.train()
    metrics = {"validation_loss": total_loss / max(total_count, 1)}
    if config["stage"] == "dpo":
        metrics.update(reward_accuracy=wins / total_count, reward_margin=margins / total_count)
    return metrics


def optimizer_step(model, optimizer, scaler, gradient_clip, scheduler=None):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    # An AMP overflow skips the optimizer update; don't advance the schedule.
    updated = scaler.get_scale() >= previous_scale
    if updated and scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return updated


def save_checkpoint(model, tokenizer, path):
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def run_training(config, resume=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

    if config["max_prompt_length"] >= config["max_length"]:
        raise ValueError("Prompt limit must leave space for a response")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "resume" / "training_state.pt"
    manifest = {"config": config, "train_sha256": file_hash(config["train_path"]), "val_sha256": file_hash(config["val_path"])}
    if resume:
        if not state_path.exists():
            raise ValueError("No interrupted-run state found; completed stages have best/final models")
        if read_rows(output / "run.json") != manifest:
            raise ValueError("Resume requires the same configuration and datasets")
    elif (output / "run.json").exists():
        raise ValueError("Run directory already exists; use --resume or a new output_dir")
    write_json(output / "run.json", manifest)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    load_path = str(output / "resume") if resume else config["model"]
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(load_path, torch_dtype=torch.float32).to(config["device"])
    model.config.use_cache = False
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = None
    if config["stage"] == "dpo":
        reference_device = config["reference_device"]
        dtype = torch.float16 if reference_device.startswith("cuda") else torch.float32
        reference = AutoModelForCausalLM.from_pretrained(config["reference_model"], torch_dtype=dtype).to(reference_device)
        reference.requires_grad_(False).eval()
        reference.config.use_cache = False
        for network in (model, reference):
            for module in network.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.
    train_rows, val_rows = read_rows(config["train_path"]), read_rows(config["val_path"])
    if not train_rows or not val_rows:
        raise ValueError("Both training and validation data are required")
    def collate(rows):
        batch = collate_rows(rows, tokenizer, config)
        sides = [batch] if config["stage"] == "sft" else list(batch.values())
        if any(b["labels"][:, 1:].ne(-100).sum(1).eq(0).any() for b in sides):
            raise ValueError("Truncation removed every scored response token; reduce max_prompt_length")
        return batch
    val_loader = DataLoader(val_rows, batch_size=config["batch_size"], collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    batches_per_epoch = math.ceil(len(train_rows) / config["batch_size"])
    total_steps = math.ceil(batches_per_epoch / config["gradient_accumulation"]) * config["epochs"]
    scheduler = get_scheduler(config["scheduler"], optimizer, num_warmup_steps=int(total_steps * config["warmup_ratio"]), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=config["device"].startswith("cuda"))
    epoch_start, batch_start, step = 0, 0, 0
    best = float("inf") if config["stage"] == "sft" else -float("inf")
    if resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        for name, expected in state["model_sha256"].items():
            if file_hash(state_path.parent / name) != expected:
                raise ValueError("Incomplete model checkpoint; its weights do not match the saved optimizer state")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch_start, batch_start, step, best = state["epoch"], state["batch"], state["step"], state["best"]
        torch.set_rng_state(state["rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        del state
    else:
        baseline = validate(model, reference, val_loader, config)
        append_jsonl(output / "metrics.jsonl", {"step": 0, **baseline})
    model.train()
    for epoch in range(epoch_start, config["epochs"]):
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        loader = DataLoader(train_rows, batch_size=config["batch_size"], shuffle=True, generator=generator, collate_fn=collate)
        cursor = batch_start if epoch == epoch_start else 0
        iterator = iter(loader)
        for _ in range(cursor):
            next(iterator)
        while group := list(itertools.islice(iterator, config["gradient_accumulation"])):
            denominator = sum(token_count(b) if config["stage"] == "sft" else batch_size(b, "dpo") for b in group)
            train_loss = 0.
            for batch in group:
                with amp_context(config["device"]):
                    if config["stage"] == "sft":
                        logps, _ = response_scores(model, batch)
                        weighted_loss = -logps.sum() / denominator
                    else:
                        loss, _ = dpo_scores(model, reference, batch, config["beta"])
                        weighted_loss = loss * batch_size(batch, "dpo") / denominator
                train_loss += weighted_loss.item()
                scaler.scale(weighted_loss).backward()
            updated = optimizer_step(model, optimizer, scaler, config["gradient_clip"], scheduler)
            cursor += len(group)
            step += int(updated)
            append_jsonl(output / "metrics.jsonl", {"step": step, "epoch": epoch, "batch": cursor,
                                                     "train_loss": train_loss, "learning_rate": optimizer.param_groups[0]["lr"], "amp_update_skipped": not updated})
            if (updated and step % config["eval_steps"] == 0) or cursor == len(loader):
                metrics = validate(model, reference, val_loader, config)
                append_jsonl(output / "metrics.jsonl", {"step": step, "epoch": epoch, **metrics})
                score = metrics["validation_loss"] if config["stage"] == "sft" else metrics["reward_accuracy"]
                improved = score < best if config["stage"] == "sft" else score > best
                if improved:
                    best = score
                    save_checkpoint(model, tokenizer, output / "best")
                    write_json(output / "best_metrics.json", {"step": step, **metrics})
                save_checkpoint(model, tokenizer, output / "resume")
                state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                         "epoch": epoch, "batch": cursor, "step": step, "best": best, "rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()}
                state["model_sha256"] = {p.name: file_hash(p) for p in state_path.parent.glob("*.safetensors")}
                temporary = state_path.with_suffix(".tmp")
                torch.save(state, temporary)
                temporary.replace(state_path)
                print({"step": step, **metrics}, flush=True)
    # Reuse the last resumable model as final rather than storing a third model.
    state_path.unlink()
    (output / "resume").rename(output / "final")
    write_json(output / "complete.json", {"optimizer_steps": step, "best_score": best})
    return model
