"""Probe for rwkv_pipeline.data_utils.build_labels (todo 8). Stdlib only.

Encodes one ChatML row via build_labels, asserts prompt positions are
-100 and response positions equal the token id; asserts a row missing
<|im_end|> stays fully -100; asserts legacy User:/Assistant: spans.
Exit 0 on success -> out/labels_check.log
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rwkv_pipeline.data_utils import build_labels  # noqa: E402

IM_USER = [11, 12]
IM_START = [13, 14]
IM_END = [15, 16]
VOICE = 77
PROMPT = [21, 22, 23]
RESP = [31, 32, 33]
CHATML = {"im_user": IM_USER, "im_start": IM_START, "im_end": IM_END}

USER = [41, 42]
ASST = [43, 44]

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(PROJECT, "out", "labels_check.log")


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" | " + detail if detail else ""))
    return cond


def main():
    ok = True
    lines = []

    # Case 1: well-formed ChatML row.
    row = IM_USER + PROMPT + IM_START + [VOICE] + RESP + IM_END
    lab = build_labels(row, delims=CHATML)
    n_pre = len(IM_USER) + len(PROMPT)
    n_voice = n_pre + len(IM_START)
    ok &= check("chatml prompt masked",
                all(v == -100 for v in lab[:n_pre]), str(lab[:n_pre]))
    ok &= check("chatml delims+voice masked",
                all(v == -100 for v in lab[n_pre:n_voice + 1]),
                str(lab[n_pre:n_voice + 1]))
    ok &= check("chatml response kept",
                lab[n_voice + 1:n_voice + 1 + len(RESP)] == RESP,
                str(lab[n_voice + 1:n_voice + 1 + len(RESP)]))
    ok &= check("chatml im_end masked", all(v == -100 for v in lab[-len(IM_END):]))
    lines.append("ChatML row: ids   =" + str(row))
    lines.append("ChatML row: labels=" + str(lab))
    lines.append("prompt positions -100: %s | response positions equal token id: %s"
                 % (all(v == -100 for v in lab[:n_pre]),
                    lab[n_voice + 1:n_voice + 1 + len(RESP)] == RESP))

    # Case 2 (QA failure): missing <|im_end|> -> whole span stays -100.
    row2 = IM_USER + PROMPT + IM_START + [VOICE] + RESP
    lab2 = build_labels(row2, delims=CHATML)
    ok &= check("missing im_end all -100",
                all(v == -100 for v in lab2), str(lab2))
    lines.append("missing-im_end row: labels=" + str(lab2))

    # Case 3: legacy User:/Assistant: spans (todo 1 decision).
    row3 = USER + PROMPT + ASST + RESP
    lab3 = build_labels(row3, mode="legacy",
                        delims={"user": USER, "asst": ASST})
    ok &= check("legacy prompt masked",
                all(v == -100 for v in lab3[:len(USER) + len(PROMPT)]),
                str(lab3))
    ok &= check("legacy response kept",
                lab3[len(USER) + len(PROMPT) + len(ASST):] == RESP,
                str(lab3))
    lines.append("legacy row: labels=" + str(lab3))

    # Case 4: pad forced to -100 even inside an open span.
    row4 = IM_USER + PROMPT + IM_START + [VOICE] + RESP + [0] + IM_END
    lab4 = build_labels(row4, delims=CHATML, pad_id=0)
    ok &= check("pad masked in span", lab4[n_voice + 1 + len(RESP)] == -100,
                str(lab4))

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote " + LOG)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
