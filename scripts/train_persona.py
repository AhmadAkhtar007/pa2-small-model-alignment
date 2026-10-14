"""The assignment's only TRL/PEFT training track."""

import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

from alignment.io import append_jsonl, file_hash, read_rows, write_json


class DurableMetrics(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        append_jsonl(Path(args.output_dir) / "metrics.jsonl", {"step": state.global_step, **(logs or {})})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/persona.json")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    config = read_rows(args.config)
    rows = read_rows(config["data_path"])
    random.Random(config["seed"]).shuffle(rows)
    val_size = max(1, round(len(rows) * .1))
    validation, training = rows[:val_size], rows[val_size:]
    def convert(row):
        return {"prompt": [{"role": "user", "content": row["prompt"]}],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}]}
    output = Path(config["output_dir"])
    if (output / "run.json").exists() and not args.resume_from_checkpoint:
        raise ValueError("Run directory already exists; resume or choose a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "dataset_sha256": file_hash(config["data_path"]),
                "train_size": len(training), "validation_size": len(validation),
                "reference": "Frozen base model obtained by disabling the trainable LoRA adapter"}
    if args.resume_from_checkpoint and read_rows(output / "run.json") != manifest:
        raise ValueError("Resume requires the same persona configuration and dataset")
    write_json(output / "train_subset.json", training)
    write_json(output / "validation_subset.json", validation)
    write_json(output / "run.json", manifest)
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], torch_dtype=torch.float32)
    model.config.use_cache = False
    lora = LoraConfig(r=config["rank"], lora_alpha=config["alpha"], lora_dropout=config["dropout"],
                      target_modules=config["target_modules"], bias="none", task_type="CAUSAL_LM")
    training_args = DPOConfig(
        output_dir=str(output), learning_rate=config["learning_rate"], beta=config["beta"],
        per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation"], num_train_epochs=config["epochs"],
        max_length=512, max_prompt_length=256, max_completion_length=256,
        fp16=torch.cuda.is_available(), bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, seed=config["seed"],
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_rewards/accuracies", greater_is_better=True,
        logging_steps=1, report_to="none", warmup_ratio=0.03, lr_scheduler_type="cosine",
        max_grad_norm=1.0, weight_decay=0.0, optim="adamw_torch", remove_unused_columns=False,
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=training_args, processing_class=tokenizer,
                         train_dataset=Dataset.from_list([convert(r) for r in training]),
                         eval_dataset=Dataset.from_list([convert(r) for r in validation]),
                         peft_config=lora, callbacks=[DurableMetrics()])
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # load_best_model_at_end may have selected an earlier adapter; save it as best.
    trainer.save_model(str(output / "best"))
    tokenizer.save_pretrained(output / "best")
    # The highest-step Trainer checkpoint contains the final epoch adapter.
    checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    write_json(output / "complete.json", {"best_adapter": str(output / "best"), "final_adapter": str(checkpoints[-1]),
                                         "best_checkpoint": trainer.state.best_model_checkpoint})


if __name__ == "__main__":
    main()
