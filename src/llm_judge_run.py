"""Send each judge item to an LLM and store its JSON verdict.

The verdicts reported in the paper (results/llm_judge/judgments.jsonl) were
produced by Claude Opus 5.5, reading exactly the system and user text in
items.jsonl, with no access to items_key.csv. This script reruns the same
protocol through the Anthropic API, so anyone with a key can repeat or extend
the evaluation (e.g. with another model, or several runs to measure
self-consistency).

Usage:
    export ANTHROPIC_API_KEY=...
    python src/llm_judge_run.py --model claude-opus-5-5 --out results/llm_judge/judgments_api.jsonl
"""
import argparse
import json
import os
import re
import time


def main(items, model, out):
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['id'] for l in open(out)}
    with open(out, 'a') as f:
        for line in open(items):
            it = json.loads(line)
            if it['id'] in done:
                continue
            msg = client.messages.create(model=model, max_tokens=300, temperature=0,
                                         system=it['system'],
                                         messages=[{'role': 'user', 'content': it['user']}])
            text = msg.content[0].text
            verdict = json.loads(re.search(r'\{.*\}', text, re.S).group(0))
            verdict['id'] = it['id']
            f.write(json.dumps(verdict) + '\n')
            f.flush()
            time.sleep(0.5)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', default='results/llm_judge/items.jsonl')
    ap.add_argument('--model', default='claude-opus-5-5')
    ap.add_argument('--out', default='results/llm_judge/judgments_api.jsonl')
    a = ap.parse_args()
    main(a.items, a.model, a.out)
