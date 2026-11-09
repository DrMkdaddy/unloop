"""Reference implementation of DRY and DRY-C.

DRY-C is DRY plus the count term from

    Yichuan Chen and Xin Wang, "When Do Repetition Loops Lock? Truncation
    Sampling, Bounded Penalties, and a Blind Spot in DRY", arXiv:2610.24715v2.

The count term is the part that sequence breakers cannot cap: it penalizes a
token by how many times it followed an earlier occurrence of the current
suffix, not by how long the matching run is.

This module is deliberately small and dependency-free. It is the code that the
tests and the simulation run; the C++ implementation in ``patches/`` mirrors it
inside llama.cpp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Token = int


@dataclass(frozen=True)
class DRYParams:
    """Sampler parameters, with the same meaning as llama.cpp's --dry-* flags."""

    multiplier: float = 1.0          # lambda
    base: float = 1.75               # b   (length term)
    allowed_length: int = 2          # a   (length term)
    penalty_last_n: int = 4096       # W
    sequence_breakers: Tuple[Tuple[Token, ...], ...] = ()

    # DRY-C; count_min_length == 0 disables the count term
    count_min_length: int = 24       # a'
    count_allowed: int = 2           # c0
    count_base: float = 3.0          # b_c

    @property
    def count_enabled(self) -> bool:
        return self.count_min_length > 0 and self.count_base > 1.0


def _window(history: Sequence[Token], params: DRYParams) -> List[Token]:
    if params.penalty_last_n <= 0:
        return []
    return list(history[-params.penalty_last_n:])


def _breaker_limit(h: Sequence[Token], params: DRYParams) -> int:
    """Index (counted from the end) where the most recent breaker starts.

    Mirrors llama.cpp: scan backwards for a breaker head, then match its tail
    against the tokens before the head.  Returns len(h) when no breaker is
    present.
    """
    n = len(h)
    for i in range(n):
        pos = n - 1 - i
        for seq in params.sequence_breakers:
            if not seq or h[pos] != seq[0]:
                continue
            tail = seq[1:]
            if len(tail) <= i and all(h[pos - 1 - off] == tail[off] for off in range(len(tail))):
                return i - len(tail)
    return n


def _single_token_breakers(params: DRYParams) -> set:
    return {seq[0] for seq in params.sequence_breakers if len(seq) == 1}


def length_penalties(history: Sequence[Token], params: DRYParams) -> Dict[Token, float]:
    """DRY's original length term: lambda * base ** (match - allowed)."""
    if params.multiplier == 0.0 or params.base < 1.0 or params.penalty_last_n == 0:
        return {}

    h = _window(history, params)
    n = len(h)
    if n <= params.allowed_length:
        return {}

    rep_limit = _breaker_limit(h, params)
    if rep_limit < params.allowed_length:
        return {}

    breakers = _single_token_breakers(params)
    best: Dict[Token, int] = {}

    # For every position q that ends an earlier occurrence of the current
    # suffix, measure how far the two agree. This is the naive form of the
    # reversed Z-algorithm llama.cpp uses.
    for q in range(n - 1):
        m = 0
        while q - m >= 0 and h[q - m] == h[n - 1 - m]:
            m += 1
        m = min(m, rep_limit)
        if m < params.allowed_length:
            continue
        nxt = h[q + 1]
        if nxt in breakers:
            continue
        if m > best.get(nxt, 0):
            best[nxt] = m

    return {
        tok: params.multiplier * params.base ** (m - params.allowed_length)
        for tok, m in best.items()
    }


def count_penalties(history: Sequence[Token], params: DRYParams) -> Dict[Token, float]:
    """DRY-C's count term: lambda * b_c ** (c(v) - c0).

    c(v) is the number of earlier positions in the window that are preceded by
    the current a'-token suffix and followed by v. Breakers do not apply.
    """
    if not params.count_enabled or params.penalty_last_n == 0:
        return {}

    h = _window(history, params)
    a = params.count_min_length
    n = len(h)
    if n <= a + 1:
        return {}

    suffix = h[n - a:]
    counts: Dict[Token, int] = {}
    # every earlier position whose a-token suffix matches, overlaps included
    for j in range(a - 1, n - 1):
        if h[j - a + 1:j + 1] == suffix:
            nxt = h[j + 1]
            counts[nxt] = counts.get(nxt, 0) + 1

    return {
        tok: params.multiplier * params.count_base ** (c - params.count_allowed)
        for tok, c in counts.items()
        if c >= params.count_allowed
    }


def penalties(history: Sequence[Token], params: DRYParams) -> Dict[Token, float]:
    """max(length term, count term), as in the paper."""
    out = length_penalties(history, params)
    for tok, p in count_penalties(history, params).items():
        out[tok] = max(out.get(tok, 0.0), p)
    return out


def apply(logits: Sequence[float], history: Sequence[Token], params: DRYParams) -> List[float]:
    """Subtract the DRY-C penalty from logits and return a new list."""
    pen = penalties(history, params)
    return [x - pen.get(i, 0.0) for i, x in enumerate(logits)]


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    z = sum(e)
    return [x / z for x in e]


def sample_dist(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    """The probabilities after the sampler, i.e. what the tests compare."""
    if temperature == 0.0:
        out = [0.0] * len(logits)
        out[max(range(len(logits)), key=lambda i: logits[i])] = 1.0
        return out
    return softmax([x / temperature for x in logits])


def find_loop(tokens: Sequence[Token], min_unit: int = 16, max_unit: int = 400,
              min_repeats: int = 3) -> Optional[Tuple[int, int]]:
    """Return (unit_length, repetitions) for the longest verbatim loop.

    A loop is a unit of ``min_unit``..``max_unit`` tokens repeated at least
    ``min_repeats`` times consecutively. This is the text-side analogue of the
    paper's token-level loop detector.
    """
    n = len(tokens)
    best: Optional[Tuple[int, int]] = None
    for unit in range(min_unit, min(max_unit, n // min_repeats) + 1):
        for start in range(0, n - min_repeats * unit + 1):
            block = tokens[start:start + unit]
            reps = 1
            pos = start + unit
            while pos + unit <= n and tokens[pos:pos + unit] == block:
                reps += 1
                pos += unit
            if reps >= min_repeats:
                if best is None or reps * unit > best[0] * best[1]:
                    best = (unit, reps)
    return best


def find_text_loop(text: str, min_unit: int = 40, max_unit: int = 1600,
                   min_repeats: int = 3) -> Optional[Tuple[int, int]]:
    """Same detector over characters, for generated text."""
    return find_loop(list(text), min_unit=min_unit, max_unit=max_unit, min_repeats=min_repeats)
