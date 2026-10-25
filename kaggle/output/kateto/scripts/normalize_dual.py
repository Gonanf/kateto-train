"""Normaliza filas viejas sin envelope dual openai/rwkv al schema v2.
Parsea bloques think/tool_call/tool_response del answer y arma messages.
Uso: python3 scripts/normalize_dual.py (reescribe data/toolcalling_v2/multi_tool.jsonl y zero_tool.jsonl)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

VDIR = Path(__file__).resolve().parent.parent / "data" / "toolcalling_v2"
SYS = "Sos Kateto, rioplatense seco, con opinion propia. Cerras tajante, nunca con pregunta."


def parse(answer):
    think = re.search(r"<think>(.*?)</think>", answer, re.S)
    calls = re.findall(r"<tool_call>(.*?)</tool_call>", answer, re.S)
    resps = re.findall(r"<tool_response>(.*?)</tool_response>", answer, re.S)
    tail = answer.split("</tool_response>")[-1].strip() if "</tool_response>" in answer else answer.strip()
    if think:
        tail = answer.split("</think>", 1)[-1]
        for block in re.findall(r"<tool_call>.*?</tool_call>\s*<tool_response>.*?</tool_response>",
                                tail, re.S):
            tail = tail.replace(block, "")
        tail = tail.strip()
    parsed_calls = [json.loads(c) for c in calls]
    parsed_resps = [json.loads(r) for r in resps]
    return (think.group(1).strip() if think else ""), parsed_calls, parsed_resps, tail


def main():
    for fname in ["multi_tool.jsonl", "zero_tool.jsonl"]:
        p = VDIR / fname
        rows = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
        for r in rows:
            if "rwkv" in r and "openai" in r:
                continue
            think, calls, resps, final = parse(r["answer"])
            r["response"] = final
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": r["question"]}]
            for i, (c, resp) in enumerate(zip(calls, resps)):
                cid = f"call_{i}"
                msgs.append({"role": "assistant", "content": json.dumps(c, ensure_ascii=False),
                             "tool_calls": [{"id": cid, "type": "function",
                                             "function": {"name": c["name"],
                                                          "arguments": json.dumps(c["arguments"],
                                                                                  ensure_ascii=False)}}]})
                msgs.append({"role": "tool", "tool_call_id": cid,
                             "content": json.dumps(resp, ensure_ascii=False)})
            msgs.append({"role": "assistant", "content": final})
            r["openai"] = {"messages": msgs}
            if calls:
                r["tool"] = calls[0]["name"]
                r["arguments"] = calls[0]["arguments"]
                r["rwkv"] = {"think": think,
                             "tool_call": calls if len(calls) > 1 else calls[0],
                             "tool_response": resps if len(resps) > 1 else resps[0],
                             "final": final}
            else:
                r["tool"] = None
                r["arguments"] = {}
                r["rwkv"] = {"think": think, "final": final}
        p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                     encoding="utf-8")
        print(f"{fname}: {len(rows)} normalizadas")


if __name__ == "__main__":
    main()
