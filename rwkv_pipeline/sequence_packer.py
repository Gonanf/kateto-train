"""Sequence packing with per-document state reset flags (SPEC §2.1).

Contract: input is a list of examples, each ``{"input_ids": [...], "labels": [...]}``
where ``labels`` already carries -100 on user/system/delimiter/voice-name
positions (todo 8 ``build_labels`` output shape). This module only passes
labels through and appends -100 pad; it never computes a loss mask itself.

OUTPUT FORBIDDEN for training while kernel gate (todo 14) is FAIL —
see docs/verify-wave0-kernel.md. Packed batches must not reach a
DataLoader or train script until the gate is PASS (todo 14 raises
``PackedBatchesForbidden`` otherwise); this module is code-complete but
QUARANTINED.
"""

import warnings


class SequencePacker:
    """Pack multiple examples into fixed-length lists with a reset mask."""

    def __init__(self, max_len: int = 512):
        self.max_len: int = max_len

    def pack(self, examples: list[dict[str, list[int]]]) -> dict[str, list[int] | list[bool]]:
        """Pack examples without splitting any across the boundary.

        Returns ``input_ids`` (padded with 0), ``labels`` (-100 pad
        passthrough) and ``reset_mask`` (False on each example's first
        token, True elsewhere including pad). Output length is always
        exactly ``max_len``. A single example longer than ``max_len``
        is skipped with a warning.
        """
        input_ids, labels, reset_mask = [], [], []
        for ex in examples:
            ids, lab = ex["input_ids"], ex["labels"]
            if len(ids) > self.max_len:
                warnings.warn(
                    "SequencePacker: example of len %d > max_len %d, skipped"
                    % (len(ids), self.max_len)
                )
                continue
            if len(input_ids) + len(ids) > self.max_len:
                break  # never split an example across the boundary
            reset_flag = [True] * len(ids)
            reset_flag[0] = False  # reset on document boundary
            input_ids.extend(ids)
            labels.extend(lab)
            reset_mask.extend(reset_flag)
        pad = self.max_len - len(input_ids)
        input_ids += [0] * pad
        labels += [-100] * pad
        reset_mask += [True] * pad
        return {"input_ids": input_ids, "labels": labels, "reset_mask": reset_mask}


if __name__ == "__main__":
    import warnings as _w

    packer = SequencePacker(max_len=16)
    examples = [
        {"input_ids": [10, 11, 12], "labels": [-100, 11, 12]},
        {"input_ids": [20, 21], "labels": [-100, 21]},
        {"input_ids": [30, 31, 32, 33], "labels": [-100, 31, 32, 33]},
    ]
    out = packer.pack(examples)
    used = sum(len(e["input_ids"]) for e in examples)
    print("lengths: %d (max_len 16, used %d, pad %d)" % (len(out["input_ids"]), used, 16 - used))
    print("exact max_len: %s" % (len(out["input_ids"]) == len(out["labels"]) == len(out["reset_mask"]) == 16))
    print("first-token flags False at starts [0,3,5]: %s" % ([out["reset_mask"][i] for i in (0, 3, 5)] == [False] * 3))
    print("pad labels all -100: %s" % all(v == -100 for v in out["labels"][used:]))
    print("pad reset_mask all True: %s" % all(out["reset_mask"][used:]))
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        long_out = packer.pack([{"input_ids": list(range(20)), "labels": [-100] * 20}])
    print("over-long skipped with warning: %s" % (len(caught) == 1 and "skipped" in str(caught[0].message)))
    print("over-long output still exact max_len, all pad: %s"
          % (len(long_out["input_ids"]) == 16 and long_out["input_ids"] == [0] * 16
             and long_out["labels"] == [-100] * 16))
