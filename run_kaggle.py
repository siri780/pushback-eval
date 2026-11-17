"""Run the pushback evaluation on a free Kaggle GPU (Settings -> Accelerator -> GPU T4 x2).

    !git clone https://github.com/siri780/pushback-eval && cd pushback-eval
    !python run_kaggle.py --model Qwen/Qwen2.5-7B-Instruct

Writes results/<model>.jsonl (every answer) and results/<model>.md (the table).
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pushback import load_questions, report, run


def make_ask(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                 device_map="auto")

    def ask(messages):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)   # greedy: repeatable
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return ask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    questions = load_questions()
    ask = make_ask(args.model)
    rows = run(questions, ask, "baseline") + run(questions, ask, "steady")
    out = Path("results")
    out.mkdir(exist_ok=True)
    name = args.model.split("/")[-1]
    (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    table = report(rows)
    (out / f"{name}.md").write_text(f"# {args.model}\n\n{table}\n")
    print(table)
