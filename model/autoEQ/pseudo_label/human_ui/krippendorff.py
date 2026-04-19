"""Krippendorff's alpha for interval metric (V/A 연속값용).

Reference: https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

반환: alpha in [-1, 1]. 1.0 = 완벽 일치, 0 = chance, 음수 = systematic disagreement.

입력 형식: dict[window_id, list[value]] — 각 window에 대한 평가자별 값(list 길이 가변, NaN 허용).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import math


def krippendorff_alpha_interval(ratings: Mapping[str, Sequence[float]]) -> float:
    """Interval metric Krippendorff's alpha."""
    # collect (unit_id, value) pairs
    units = []  # list of list of values per unit (drop NaN)
    all_values = []
    for uid, vals in ratings.items():
        clean = [float(v) for v in vals if v is not None and not _isnan(v)]
        if len(clean) >= 2:
            units.append(clean)
            all_values.extend(clean)

    if len(units) < 1 or len(all_values) < 2:
        return float("nan")

    # observed disagreement Do
    num_do = 0.0
    denom_do = 0.0
    for vals in units:
        n = len(vals)
        if n < 2:
            continue
        pairs = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pairs += (vals[i] - vals[j]) ** 2
        num_do += pairs / (n - 1)
        denom_do += n

    if denom_do == 0:
        return float("nan")
    Do = num_do / denom_do

    # expected disagreement De
    N = len(all_values)
    num_de = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            num_de += (all_values[i] - all_values[j]) ** 2
    De = num_de / (N * (N - 1)) if N > 1 else 0.0

    if De == 0:
        return float("nan")
    return 1.0 - Do / De


def _isnan(x) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False
