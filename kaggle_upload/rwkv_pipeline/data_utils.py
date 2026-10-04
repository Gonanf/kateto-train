"""Loss-mask label builder for Kateto RWKV training (SPEC 2.4, corrected).

Corrected 2-state machine (CLOSED/OPEN). The SPEC sketch is intent-only:
its flag is never set True, so this module implements the fixed version:

- chatml_text (default): ChatML delimiters arrive as plain-text
  multi-token sequences (todo 1 verdict: single-token ids MISSING from
  ``rwkv_vocab_v20230424``), so callers pass pre-encoded id sequences.
  State CLOSES on ``<|im_user|>`` / ``<|im_end|>`` (+ optional extras);
  it OPENS only AFTER the single voice-name token following
  ``<|im_start|>``. Labels keep response tokens, -100 everywhere else
  (prompt, delimiters, voice-name, pad). A span closed by anything other
  than ``<|im_end|>`` (next ``<|im_user|>``, EOS) is reverted to all
  -100: a missing ``<|im_end|>`` must leak zero gradient.
- legacy: ``User:``/``Assistant:`` spans, mirroring the (commented-out)
  reference ``generate_mask`` in ``RWKV-PEFT/rwkvt/dataset/mask.py``:
  ``User:`` span CLOSES (masked), ``Assistant:`` span OPENS *inclusively*
  (span tokens kept, as upstream sets mask=1 on them).

Stdlib only.
"""
from typing import Any

IGNORE = -100


def _match(ids: list[int], pos: int, seq: list[int]) -> bool:
    """True if ``seq`` (non-empty) matches ``ids`` starting at ``pos``."""
    if not seq or pos + len(seq) > len(ids):
        return False
    return ids[pos:pos + len(seq)] == list(seq)


def _match_any(ids: list[int], pos: int, seqs: list[list[int]]) -> list[int] | None:
    for s in sorted(seqs, key=len, reverse=True):
        if _match(ids, pos, s):
            return s
    return None


def chatml_seqs(tokenizer):
    """Encode the three ChatML delimiters with a real tokenizer.

    Returns ``{"im_user": [...], "im_start": [...], "im_end": [...]}``
    for use as ``build_labels(..., delims=chatml_seqs(tok))``.
    """
    return {
        "im_user": list(tokenizer.encode("<|im_user|>")),
        "im_start": list(tokenizer.encode("<|im_start|>")),
        "im_end": list(tokenizer.encode("<|im_end|>")),
    }


def build_labels(input_ids: list[int], mode: str = "chatml_text",
                 delims: dict[str, Any] | None = None, pad_id: int = 0) -> list[int]:
    """Build loss-mask labels: response token ids kept, -100 elsewhere.

    Args:
        input_ids: list of token ids for one row.
        mode: "chatml_text" (default) or "legacy".
        delims: chatml_text -> {"im_user", "im_start", "im_end",
            optional "extra_close": [seq, ...]}; legacy ->
            {"user": [...], "asst": [...]}. All seqs are id lists.
        pad_id: id forced to -100 even inside an open span.
    """
    ids = list(input_ids)
    labels = [IGNORE] * len(ids)
    if not ids:
        return labels
    delims = delims or {}

    if mode == "legacy":
        user_spans = [delims["user"]] if delims.get("user") else []
        asst_spans = [delims["asst"]] if delims.get("asst") else []
        is_open = False
        i = 0
        while i < len(ids):
            hit = _match_any(ids, i, user_spans)
            if hit is not None:
                is_open = False
                i += len(hit)
                continue
            hit = _match_any(ids, i, asst_spans)
            if hit is not None:
                is_open = True
                for j in range(i, i + len(hit)):  # upstream mask=1 on span
                    labels[j] = ids[j] if ids[j] != pad_id else IGNORE
                i += len(hit)
                continue
            labels[i] = ids[i] if (is_open and ids[i] != pad_id) else IGNORE
            i += 1
        return labels

    if mode != "chatml_text":
        raise ValueError("mode must be 'chatml_text' or 'legacy'")

    im_user = delims.get("im_user", [])
    im_start = delims.get("im_start", [])
    im_end = delims.get("im_end", [])
    close_seqs = [s for s in [im_user, im_end] + list(
        delims.get("extra_close", [])) if s]
    is_open = False
    pending = []  # positions labelled while OPEN, committed only by im_end
    i = 0
    while i < len(ids):
        # A close delimiter ends the span: commit only on im_end,
        # otherwise revert (missing-im_end span stays fully -100).
        hit = _match_any(ids, i, close_seqs)
        if hit is not None:
            if is_open:
                if _match(ids, i, im_end):
                    pending = []
                else:
                    for j in pending:
                        labels[j] = IGNORE
                    pending = []
                is_open = False
            i += len(hit)
            continue
        if _match(ids, i, im_start):
            if is_open:  # malformed nesting: drop the unclosed span
                for j in pending:
                    labels[j] = IGNORE
                pending = []
                is_open = False
            i += len(im_start)
            # Next single token is the voice name -> ignore; but if a
            # close delimiter follows immediately there is no voice name.
            if i < len(ids) and _match_any(ids, i, close_seqs) is None:
                i += 1
            is_open = True
            continue
        if ids[i] == pad_id:
            labels[i] = IGNORE
        elif is_open:
            labels[i] = ids[i]
            pending.append(i)
        i += 1
    if is_open:  # EOS with no im_end: no gradient leak
        for j in pending:
            labels[j] = IGNORE
    return labels
