from pathlib import Path
import json, sys, zipfile
import kaggle_benchmarks as kbench

markers = list(Path('/kaggle/input').rglob('generation-complete.json'))
assert len(markers) == 1
manifest = json.loads(markers[0].read_text())
source = markers[0].parent / manifest['project']
sys.path.insert(0, str(source))
from alignment.evaluation import judge_pairs, summarize
from alignment.io import read_rows

OUT = Path('/kaggle/working/persona-retry')
OUT.mkdir(exist_ok=True)
output = OUT / 'persona_judged.jsonl'
seed = '{"id": "persona-00", "winner": "A", "winner_model": "persona-lora", "reason": "WHAT A DISASTER! Both models serve hallucinated kitchen nonsense, but Model A at least correctly identifies that overcooking leads to mushy pasta. Model B insanely claims overcooking makes pasta \'tough and rubbery\' and that heat from your hands turns cooked pasta mushy—GET OUT OF MY KITCHEN!", "pair_sha256": "cecdf277ac70fcdc6f91b059ac0d349ab9d6f345749324ba2a525ac40b73ba20", "model_a": "persona-lora", "model_b": "smollm-instruct", "style_a": 1, "style_b": 1, "usefulness_a": 2, "usefulness_b": 1, "safe_a": true, "safe_b": true, "judge_model": "google/gemini-3.6-flash", "mode": "persona", "attempts": 1}\n{"id": "persona-01", "winner": "A", "winner_model": "persona-lora", "reason": "ABSOLUTE SACRILEGE! Model B tells the user to bake a leftover steak for 2 to 4 HOURS to reheat it—that is culinary arson! Model A\'s 20 minutes per side at 350°F will produce leather, but it\'s slightly less insane than B.", "pair_sha256": "5daa255b93cbf7aaa42f77bed52e6b33f22d499222c8f903acfe5063442ffe4c", "model_a": "persona-lora", "model_b": "smollm-instruct", "style_a": 1, "style_b": 1, "usefulness_a": 2, "usefulness_b": 1, "safe_a": true, "safe_b": true, "judge_model": "google/gemini-3.6-flash", "mode": "persona", "attempts": 1}\n{"id": "persona-02", "winner": "B", "winner_model": "persona-lora", "reason": "FIRE HAZARD! Model A instructs you to put eating plates into a 375°F oven and bake chicken for 3 HOURS! That will shatter plates and burn your kitchen down! Model B is incomplete and bizarrely tells you to wear gloves while washing hands, but it is vastly safer.", "pair_sha256": "03c79149f2fe0c1b21517a8f8aaf0bdbdd4784ca6fdf98390d42b159a9197dfe", "model_a": "smollm-instruct", "model_b": "persona-lora", "style_a": 1, "style_b": 1, "usefulness_a": 1, "usefulness_b": 2, "safe_a": false, "safe_b": true, "judge_model": "google/gemini-3.6-flash", "mode": "persona", "attempts": 1}\n{"id": "persona-03", "winner": "tie", "winner_model": null, "reason": "YOU SERVED ME THE EXACT SAME DISH TWICE! Both outputs are 100% identical carbon copies, down to the same code, explanation, cut-offs, and NameError in the example code (`remove_unique` vs `remove_duplicate`). An absolute tie!", "pair_sha256": "cee6e770f8099af77dee14246d2a5014fe8ae6a3280feeb2a811c25f9f43f182", "model_a": "persona-lora", "model_b": "smollm-instruct", "style_a": 1, "style_b": 1, "usefulness_a": 2, "usefulness_b": 2, "safe_a": true, "safe_b": true, "judge_model": "google/gemini-3.6-flash", "mode": "persona", "attempts": 1}\n{"id": "persona-04", "winner": "A", "winner_model": "persona-lora", "reason": "DONKEY CODE! Model B bizarrely claims that dividing by 1 triggers a `ZeroDivisionError` and suggests an infinite `while` loop! Model A stumbles by mentioning \'division by 2\', but offers actual practical debugging advice like try/catch blocks and input validation.", "pair_sha256": "a8fdc5f2a2d751299348760d3cf9525d98a42718198bf1e61cf5c990f9a57e34", "model_a": "persona-lora", "model_b": "smollm-instruct", "style_a": 1, "style_b": 1, "usefulness_a": 3, "usefulness_b": 1, "safe_a": true, "safe_b": true, "judge_model": "google/gemini-3.6-flash", "mode": "persona", "attempts": 1}\n{"id": "persona-10", "winner": "tie", "winner_model": null, "reason": "Both responses completely miss the prompt. Instead of adopting a fiery theatrical chef persona to critique burnt toast and give culinary advice, both models outputted an irrelevant theatrical analysis of a restaurant scene, provided no useful advice, and cut off mid-sentence.", "pair_sha256": "8c5ce2d17a39fe288cbf436c601e92b58f5777491bb5c07784342b1d72d8c832", "model_a": "persona-lora", "model_b": "smollm-instruct", "style_a": 1, "style_b": 1, "usefulness_a": 1, "usefulness_b": 1, "safe_a": true, "safe_b": true, "judge_model": "google/gemini-3.6-flash", "mode": "persona", "attempts": 1}\n{"id": "persona-11", "winner": "B", "winner_model": "smollm-instruct", "reason": "Both models completely fail to adopt the requested impatient chef persona. Furthermore, both use a flawed analogy (layering a cake) that describes composition or iteration rather than recursion. However, Response B is slightly better because it at least explicitly defines the base case and recursive call, despite cutting off at the end.", "pair_sha256": "cde2e270d82b67fef78730a82d829f4ae0a5ec76d202818d17cfbdbcd931ed19", "model_a": "persona-lora", "model_b": "smollm-instruct", "style_a": 1, "style_b": 1, "usefulness_a": 2, "usefulness_b": 2, "safe_a": true, "safe_b": true, "judge_model": "google/gemini-3.6-flash", "mode": "persona", "attempts": 1}\n'
if not output.exists():
    output.write_text(seed)
pairs = read_rows(source / 'outputs/final_eval/persona_pairs.jsonl')
assert len(pairs) == 12
JUDGE = 'google/gemini-3.6-flash'

@kbench.task(name='pa2_persona_failed_batch_retry')
def retry_persona(llm) -> bool:
    def request(prompt):
        with kbench.chats.new('persona_retry'):
            raw = llm.prompt(prompt)
        with (OUT / 'raw-responses.jsonl').open('a') as stream:
            stream.write(json.dumps({'prompt': prompt, 'response': str(raw)}, ensure_ascii=False) + '\n')
        print('Judge response received; saved judgments:', len(output.read_text().splitlines()), flush=True)
        return raw
    judgments = judge_pairs(pairs, output, request, JUDGE, batch_size=1, sleep_seconds=5, max_attempts=3, mode='persona')
    failed_path = output.with_suffix('.failures.jsonl')
    failures = read_rows(failed_path) if failed_path.exists() else []
    summary = summarize(pairs, judgments, failures, 'persona-lora')
    (OUT / 'persona-summary.json').write_text(json.dumps(summary, indent=2))
    with zipfile.ZipFile('/kaggle/working/persona-retry-results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.iterdir():
            if path.is_file():
                archive.write(path, path.name)
    print('PERSONA RETRY FINISHED', json.dumps(summary, indent=2), flush=True)
    return True

retry_persona.run(kbench.llms[JUDGE])
