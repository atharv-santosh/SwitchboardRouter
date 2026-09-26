"""Fitting the learned escalation policy on recorded shadow runs, and scoring it honestly."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .events import iter_runs
from .policies import FEATURE_NAMES


@dataclass
class Row:
    features: dict[str, float]
    label: int
    run_id: str
    task_id: str


def load_shadow_rows(directory: str | Path) -> list[Row]:
    """Pull every labelled cheap-model step out of recorded shadow trajectories."""
    rows: list[Row] = []
    for run_id, events in iter_runs(directory):
        task_id = ""
        for event in events:
            if event.type == "run_start":
                task_id = str(event.payload.get("task_id", ""))
            elif event.type == "decision" and event.payload.get("kind") == "shadow_label":
                rows.append(
                    Row(
                        features={k: float(v) for k, v in event.payload["features"].items()},
                        label=int(event.payload["cheap_was_wrong"]),
                        run_id=run_id,
                        task_id=task_id,
                    )
                )
    return rows


def split_rows(rows: list[Row], holdout_fraction: float = 0.3) -> tuple[list[Row], list[Row]]:
    """Split by task id, not by row, so the same task cannot appear on both sides.

    Splitting rows at random would put steps from one task into train and holdout at once.
    Adjacent steps of one run share almost everything, so that split leaks and the holdout
    number it produces is not a holdout number.
    """
    task_ids = sorted({r.task_id for r in rows})
    if len(task_ids) < 2:
        return rows, []
    n_holdout = max(1, int(round(len(task_ids) * holdout_fraction)))
    # Stride rather than take a contiguous block, so the holdout is spread across the suite.
    stride = max(1, len(task_ids) // n_holdout)
    holdout = {task_ids[i] for i in range(0, len(task_ids), stride)[:n_holdout]}
    train = [r for r in rows if r.task_id not in holdout]
    test = [r for r in rows if r.task_id in holdout]
    return train, test


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


@dataclass
class Fit:
    weights: dict[str, float]
    threshold: float
    n_train: int
    n_holdout: int
    train_metrics: dict[str, float] = field(default_factory=dict)
    holdout_metrics: dict[str, float] = field(default_factory=dict)
    base_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {k: round(v, 5) for k, v in self.weights.items()},
            "threshold": round(self.threshold, 4),
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "base_rate_cheap_wrong": round(self.base_rate, 4),
            "train": self.train_metrics,
            "holdout": self.holdout_metrics,
        }


def fit_logistic(
    rows: list[Row],
    *,
    epochs: int = 900,
    lr: float = 0.35,
    l2: float = 1e-3,
) -> dict[str, float]:
    """Plain batch gradient descent. Nine features and a few thousand rows do not need more."""
    weights = {name: 0.0 for name in FEATURE_NAMES}
    if not rows:
        return weights
    n = len(rows)
    for _ in range(epochs):
        grads = {name: 0.0 for name in FEATURE_NAMES}
        for row in rows:
            z = sum(weights[k] * row.features.get(k, 0.0) for k in weights)
            err = sigmoid(z) - row.label
            for k in weights:
                grads[k] += err * row.features.get(k, 0.0)
        for k in weights:
            penalty = 0.0 if k == "bias" else l2 * weights[k]
            weights[k] -= lr * (grads[k] / n + penalty)
    return weights


def predict(weights: dict[str, float], features: dict[str, float]) -> float:
    return sigmoid(sum(weights.get(k, 0.0) * v for k, v in features.items()))


def metrics_at(weights: dict[str, float], rows: list[Row], threshold: float) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        flagged = predict(weights, row.features) >= threshold
        if row.label == 1:
            tp += int(flagged)
            fn += int(not flagged)
        else:
            fp += int(flagged)
            tn += int(not flagged)
    total = max(1, tp + fp + tn + fn)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round((tp + tn) / total, 4),
        "recall_on_wrong": round(tpr, 4),
        "false_escalation_rate": round(fpr, 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "youden_j": round(tpr - fpr, 4),
        "auc": round(auc(weights, rows), 4),
    }


def auc(weights: dict[str, float], rows: list[Row]) -> float:
    """Rank-based AUC; ties count a half. Small enough here to do the O(n^2) way."""
    pos = [predict(weights, r.features) for r in rows if r.label == 1]
    neg = [predict(weights, r.features) for r in rows if r.label == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def choose_threshold(weights: dict[str, float], rows: list[Row]) -> float:
    """Pick the threshold maximising Youden's J on the training rows.

    J weighs a caught mistake and an unnecessary escalation equally, which is not the real
    economics: an escalation costs a known number of cents and a missed mistake costs a
    whole task. The threshold is therefore a default, not an optimum, and the frontier is
    traced by sweeping it instead of arguing about it.
    """
    best, best_j = 0.5, -2.0
    for i in range(5, 96):
        t = i / 100.0
        j = metrics_at(weights, rows, t)["youden_j"]
        if j > best_j:
            best, best_j = t, j
    return best


def fit(rows: list[Row], holdout_fraction: float = 0.3) -> Fit:
    train, test = split_rows(rows, holdout_fraction)
    weights = fit_logistic(train)
    threshold = choose_threshold(weights, train)
    return Fit(
        weights=weights,
        threshold=threshold,
        n_train=len(train),
        n_holdout=len(test),
        train_metrics=metrics_at(weights, train, threshold),
        holdout_metrics=metrics_at(weights, test, threshold),
        base_rate=sum(r.label for r in rows) / len(rows) if rows else 0.0,
    )


def feature_importance(weights: dict[str, float], rows: Iterable[Row]) -> list[tuple[str, float]]:
    """Weight times feature standard deviation, so the ordering is not a units artefact."""
    rows = list(rows)
    out: list[tuple[str, float]] = []
    for name, weight in weights.items():
        if name == "bias" or not rows:
            continue
        values = [r.features.get(name, 0.0) for r in rows]
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        out.append((name, round(weight * sd, 4)))
    return sorted(out, key=lambda kv: -abs(kv[1]))
