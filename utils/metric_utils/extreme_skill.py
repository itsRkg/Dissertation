"""
extreme_skill.py — categorical (dichotomous) verification scores for heavy-rainfall detection.

NEW, additive module. Does NOT modify metrics.py. Created for track_c
(see docs/track_3_rl_reuse/session_plan.md §4).

PURPOSE / FRAMING (read before using):
    The dissertation does NOT claim extreme-event skill (the models under-disperse;
    see session_plan.md §0). These scores are a PRIVATE DIAGNOSTIC + limitation
    evidence — they characterise *where* the model's heavy-rain detection breaks
    down, for decision calls and the limitations section. They are not a headline
    metric and never gate the track_c success criterion (RMSE + train-val gap).

METHOD:
    Given real-space (mm/day) observations `obs` and predictions `pred` and a
    threshold `tau`, build the 2x2 contingency table at that threshold:

        a = hits          (obs >= tau  AND  pred >= tau)
        b = false alarms  (obs <  tau  AND  pred >= tau)
        c = misses        (obs >= tau  AND  pred <  tau)
        d = correct neg   (obs <  tau  AND  pred <  tau)

    and compute standard WMO/CAWCR forecast-verification scores:

        POD  (prob. of detection / hit rate) = a / (a + c)              best 1
        FAR  (false alarm RATIO)             = b / (a + b)              best 0
        SR   (success ratio)                 = a / (a + b) = 1 - FAR    best 1
        CSI  (critical success index / TS)   = a / (a + b + c)          best 1
        frequency_bias                       = (a + b) / (a + c)        best 1
        HSS  (Heidke skill score, 2x2)       = 2(ad - bc)
                                               / [(a+c)(c+d) + (a+b)(b+d)]  best 1, 0 = chance

    NOTE on FAR: this is the false alarm *ratio* b/(a+b), NOT the false alarm
    *rate* (a.k.a. POFD = b/(b+d)). The two are routinely confused; we use the
    ratio because it is the standard precipitation-verification convention.

    frequency_bias < 1  => model under-forecasts the event (our expected regime,
    due to MSE mean-reversion); > 1 => over-forecasts.

IMD daily rainfall categories (mm/day) used as default thresholds:
    rather heavy >= 35.6, heavy >= 64.5, very heavy >= 124.5.
    Very-heavy events are rare -> scores are statistically noisy. ALWAYS read
    `n_events` alongside the scores; lead any narrative with the lower thresholds.

Relationship to the existing utils/metric_utils/metrics.py::extreme_metrics():
    that helper returns precision/recall/F1 at a threshold. Here, recall == POD
    and precision == success_ratio == (1 - FAR). This module adds the
    meteorology-standard names plus CSI, frequency bias and HSS, and a
    multi-threshold table.

Sources for the score definitions (verified 2026-05-30 via web search):
    - CAWCR Forecast Verification: methods across time/space scales
      (https://www.cawcr.gov.au/projects/verification/)
    - EUMETRAIN, Frequency Bias
      (https://resources.eumetrain.org/data/4/451/english/msg/ver_categ_forec/uos2/uos2_ko1.htm)
    - NWS/OWP Glossary of Forecast Verification Metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = (35.6, 64.5, 124.5)  # IMD rather-heavy / heavy / very-heavy (mm/day)


def _clean(obs, pred):
    """Flatten to 1-D float arrays and drop pairs where either side is NaN/inf."""
    obs = np.asarray(obs, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if obs.shape != pred.shape:
        raise ValueError(
            f"obs and pred must have the same number of elements after ravel; "
            f"got {obs.shape} vs {pred.shape}"
        )
    mask = np.isfinite(obs) & np.isfinite(pred)
    return obs[mask], pred[mask]


def contingency(obs, pred, tau):
    """Return the 2x2 contingency counts (a=hits, b=false_alarms, c=misses, d=correct_neg)."""
    obs, pred = _clean(obs, pred)
    o = obs >= tau
    p = pred >= tau
    a = int(np.count_nonzero(o & p))
    b = int(np.count_nonzero(~o & p))
    c = int(np.count_nonzero(o & ~p))
    d = int(np.count_nonzero(~o & ~p))
    return a, b, c, d


def _safe_div(num, den):
    """Division that returns NaN (not an error) when the denominator is zero."""
    den = float(den)
    return float(num) / den if den != 0.0 else np.nan


def pod(a, b, c, d):
    return _safe_div(a, a + c)


def far(a, b, c, d):
    return _safe_div(b, a + b)


def success_ratio(a, b, c, d):
    return _safe_div(a, a + b)


def csi(a, b, c, d):
    return _safe_div(a, a + b + c)


def frequency_bias(a, b, c, d):
    return _safe_div(a + b, a + c)


def hss(a, b, c, d):
    denom = (a + c) * (c + d) + (a + b) * (b + d)
    return _safe_div(2.0 * (a * d - b * c), denom)


def extreme_skill_row(obs, pred, tau):
    """All scores + counts at a single threshold, as a dict."""
    a, b, c, d = contingency(obs, pred, tau)
    return {
        "threshold_mm": float(tau),
        "n_events": a + c,        # number of OBSERVED events >= tau
        "n_forecast": a + b,      # number of FORECAST events >= tau
        "hits": a,
        "false_alarms": b,
        "misses": c,
        "correct_neg": d,
        "POD": pod(a, b, c, d),
        "FAR": far(a, b, c, d),
        "success_ratio": success_ratio(a, b, c, d),
        "CSI": csi(a, b, c, d),
        "frequency_bias": frequency_bias(a, b, c, d),
        "HSS": hss(a, b, c, d),
    }


def extreme_skill_table(obs, pred, thresholds=DEFAULT_THRESHOLDS):
    """
    Multi-threshold table of extreme-event detection scores.

    Parameters
    ----------
    obs, pred : array-like (any shape; flattened internally), real-space mm/day.
    thresholds : iterable of float, mm/day. Defaults to IMD 35.6/64.5/124.5.

    Returns
    -------
    pandas.DataFrame, one row per threshold. Columns include n_events (read this!),
    POD, FAR, CSI, frequency_bias, HSS, and the raw contingency counts.
    """
    rows = [extreme_skill_row(obs, pred, t) for t in thresholds]
    return pd.DataFrame(rows)


def extreme_skill_by_group(obs, pred, group, thresholds=DEFAULT_THRESHOLDS):
    """
    Optional: per-group (e.g. per-station or per-season) tables, concatenated.
    `group` is an array-like of labels aligned with the flattened obs/pred.
    Useful for the limitation section; not required for the pooled headline table.
    """
    obs = np.asarray(obs, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    group = np.asarray(group).ravel()
    if not (obs.shape == pred.shape == group.shape):
        raise ValueError("obs, pred and group must have the same number of elements.")
    out = []
    for g in pd.unique(group):
        sel = group == g
        tbl = extreme_skill_table(obs[sel], pred[sel], thresholds)
        tbl.insert(0, "group", g)
        out.append(tbl)
    return pd.concat(out, ignore_index=True)


def _selftest():
    """Hand-built contingency cases with known answers. Run: python extreme_skill.py"""
    # ---- Case 1: known table a=5, b=2, c=3, d=90 at tau=10 ----
    obs = np.array([20] * 5 + [1] * 2 + [20] * 3 + [1] * 90, dtype=float)
    pred = np.array([20] * 5 + [20] * 2 + [1] * 3 + [1] * 90, dtype=float)
    a, b, c, d = contingency(obs, pred, 10.0)
    assert (a, b, c, d) == (5, 2, 3, 90), (a, b, c, d)
    assert np.isclose(pod(a, b, c, d), 5 / 8)
    assert np.isclose(far(a, b, c, d), 2 / 7)
    assert np.isclose(success_ratio(a, b, c, d), 5 / 7)
    assert np.isclose(csi(a, b, c, d), 5 / 10)
    assert np.isclose(frequency_bias(a, b, c, d), 7 / 8)
    assert np.isclose(hss(a, b, c, d), 888.0 / 1388.0)  # = 0.639769...

    # ---- Case 2: perfect forecast ----
    o2 = np.array([0, 50, 100, 5, 80], dtype=float)
    a, b, c, d = contingency(o2, o2.copy(), 64.5)
    assert b == 0 and c == 0
    assert np.isclose(pod(a, b, c, d), 1.0)
    assert np.isclose(far(a, b, c, d), 0.0)
    assert np.isclose(csi(a, b, c, d), 1.0)
    assert np.isclose(frequency_bias(a, b, c, d), 1.0)
    assert np.isclose(hss(a, b, c, d), 1.0)

    # ---- Case 3: no observed events above threshold -> undefined scores are NaN ----
    a, b, c, d = contingency(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 1000.0)
    assert (a, b, c, d) == (0, 0, 0, 3)
    assert np.isnan(pod(a, b, c, d))
    assert np.isnan(far(a, b, c, d))
    assert np.isnan(csi(a, b, c, d))
    assert np.isnan(frequency_bias(a, b, c, d))
    assert np.isnan(hss(a, b, c, d))

    # ---- Case 4: NaN pairs are dropped, not counted ----
    a, b, c, d = contingency(
        np.array([20.0, np.nan, 20.0]), np.array([20.0, 20.0, np.nan]), 10.0
    )
    assert (a, b, c, d) == (1, 0, 0, 0), (a, b, c, d)

    # ---- Case 5: table shape + columns ----
    tbl = extreme_skill_table(obs, pred)
    assert list(tbl["threshold_mm"]) == list(DEFAULT_THRESHOLDS)
    assert {"n_events", "POD", "FAR", "CSI", "frequency_bias", "HSS"}.issubset(tbl.columns)

    print("extreme_skill self-test: ALL PASS")
    print(extreme_skill_table(obs, pred).to_string(index=False))


if __name__ == "__main__":
    _selftest()
