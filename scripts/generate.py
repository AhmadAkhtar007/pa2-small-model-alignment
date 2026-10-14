"""Greedy generation with identical settings and provenance-checked resume."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.io import file_hash, read_rows, write_json
from alignment.training import prompt_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat-template", action="store_true", help="Use only for SmolLM2 persona comparisons")
    parser.add_argument("--limit", type=int, help="Use 2-5 for a separate sanity-check output")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rows = read_rows(args.prompts)
    if args.limit:
        rows = rows[:args.limit]
    ids = [str(r["id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Prompt IDs must be unique")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True
    output = Path(args.output)
    metadata = output.with_suffix(".meta.json")
    settings = {**vars(args), "do_sample": False, "num_beams": 1,
                "temperature": 1.0, "top_p": 1.0, "length_penalty": 1.0,
                "prompt_sha256": file_hash(args.prompts), "eos_token_id": tokenizer.eos_token_id,
                "model_revision": getattr(model.config, "_commit_hash", None)}
    weights = [p for directory in [args.model, args.adapter] if directory and Path(directory).is_dir()
               for p in Path(directory).glob("*.safetensors")]
    settings["local_weight_sha256"] = {str(p): file_hash(p) for p in weights}
    if metadata.exists() and read_rows(metadata) != settings:
        raise ValueError("Generation settings, weights, or prompts changed; use a new output path")
    if output.exists() and not metadata.exists():
        raise ValueError("Existing outputs have no provenance metadata")
    write_json(metadata, settings)
    completed = read_rows(output) if output.exists() else []
    seen = {str(r["id"]) for r in completed}
    if len(seen) != len(completed) or not seen <= set(ids):
        raise ValueError("Existing outputs contain duplicate or unexpected IDs")
    for row in rows:
        if str(row["id"]) in seen:
            continue
        if args.chat_template:
            text = tokenizer.apply_chat_template([{"role": "user", "content": row["instruction"]}], tokenize=False, add_generation_prompt=True)
        else:
            text = prompt_text(tokenizer, row["instruction"], args.max_prompt_length)
        input_ids = tokenizer.encode(text, add_special_tokens=False)[-args.max_prompt_length:]
        inputs = torch.tensor([input_ids], device=args.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False, num_beams=1,
                max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
        completed.append({**row, "id": str(row["id"]), "output": response, "generator": args.generator})
        write_json(output, completed)
        print(f"{len(completed)}/{len(rows)}: {row['id']}", flush=True)


if __name__ == "__main__":
    main()
