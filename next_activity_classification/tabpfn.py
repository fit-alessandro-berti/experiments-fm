#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py
from pm4py.util import xes_constants
from sklearn.metrics import accuracy_score
from tabpfn_client import TabPFNClassifier, init

try:
    from .feature_extraction import build_next_activity_dataset, fit_feature_space
except ImportError:
    from feature_extraction import build_next_activity_dataset, fit_feature_space


def is_xes_file(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".xes") or path.name.endswith(".xes.gz"))


def log_name_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".xes.gz"):
        return name[: -len(".xes.gz")]
    if name.endswith(".xes"):
        return name[: -len(".xes")]
    return path.stem


def cap_rows(
    x: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[indices], y[indices]


def keep_top_classes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if top_k < 2:
        raise ValueError("top_k must be at least 2.")

    classes, counts = np.unique(y_train, return_counts=True)
    if classes.size == 0:
        raise ValueError("Training labels are empty.")

    sorted_indices = np.argsort(-counts)
    top_classes = classes[sorted_indices[: min(top_k, classes.size)]]
    top_classes_set = set(int(c) for c in top_classes.tolist())

    train_mask = np.isin(y_train, top_classes)
    test_mask = np.isin(y_test, top_classes)

    return (
        x_train[train_mask],
        y_train[train_mask],
        x_test[test_mask],
        y_test[test_mask],
        sorted(top_classes_set),
    )


def evaluate_log(
    train_log_path: Path,
    test_log_path: Path,
    seed: int,
    top_classes: int,
    max_rows: int,
    method_name: str,
    activity_key: str,
    timestamp_key: str,
) -> dict:
    train_log = pm4py.read_xes(str(train_log_path), return_legacy_log_object=True)
    test_log = pm4py.read_xes(str(test_log_path), return_legacy_log_object=True)

    feature_space = fit_feature_space(train_log, activity_key=activity_key)
    x_train_rows, y_train_raw = build_next_activity_dataset(
        train_log,
        feature_space=feature_space,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
    )
    x_test_rows, y_test_raw = build_next_activity_dataset(
        test_log,
        feature_space=feature_space,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
    )

    train_fea_df = pd.DataFrame(x_train_rows)
    test_fea_df = pd.DataFrame(x_test_rows)
    x_train = train_fea_df.to_numpy(dtype=np.float64)
    x_test = test_fea_df.to_numpy(dtype=np.float64)
    y_train = np.asarray(y_train_raw, dtype=np.int64)
    y_test = np.asarray(y_test_raw, dtype=np.int64)
    if not x_train_rows:
        raise ValueError("No train prefixes extracted.")
    if not x_test_rows:
        raise ValueError("No test prefixes extracted.")

    x_train_raw_rows = int(x_train.shape[0])
    x_test_raw_rows = int(x_test.shape[0])
    x_train, y_train, x_test, y_test, kept_classes = keep_top_classes(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        top_k=top_classes,
    )
    if x_train.shape[0] == 0:
        raise ValueError("No training rows left after top-class filtering.")
    if x_test.shape[0] == 0:
        raise ValueError("No test rows left after top-class filtering.")

    x_train, y_train = cap_rows(x_train, y_train, max_rows=max_rows, seed=seed)
    x_test, y_test = cap_rows(x_test, y_test, max_rows=max_rows, seed=seed + 1)

    train_classes = int(np.unique(y_train).shape[0])
    test_classes = int(np.unique(y_test).shape[0])
    if train_classes < 2:
        raise ValueError("Training data has fewer than 2 classes after filtering/capping.")

    model = TabPFNClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    if np.isnan(accuracy):
        raise ValueError("Accuracy is NaN and cannot be written.")

    return {
        "method": method_name,
        "train_log": str(train_log_path),
        "test_log": str(test_log_path),
        "train_prefixes_raw": x_train_raw_rows,
        "test_prefixes_raw": x_test_raw_rows,
        "train_prefixes": int(x_train.shape[0]),
        "test_prefixes": int(x_test.shape[0]),
        "train_classes": train_classes,
        "test_classes": test_classes,
        "kept_train_classes_top_k": kept_classes,
        "many_class_strategy": f"top_{top_classes}_train_frequency_filter",
        "accuracy": accuracy,
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate a TabPFN classifier for next-activity prediction using "
            "logs/training/<percentage> and logs/test."
        )
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "logs" / "training",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "logs" / "test",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--method-name", default="tabpfn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10000,
        help="Maximum number of rows kept per split (train/test) before TabPFN.",
    )
    parser.add_argument(
        "--top-classes",
        type=int,
        default=10,
        help="Keep only the top-K most frequent training classes in both training and test.",
    )
    parser.add_argument("--activity-key", default=xes_constants.DEFAULT_NAME_KEY)
    parser.add_argument("--timestamp-key", default=xes_constants.DEFAULT_TIMESTAMP_KEY)
    parser.add_argument(
        "--api-key",
        default=os.getenv("TABPFN_API_KEY"),
        help="TabPFN API key (defaults to TABPFN_API_KEY env var).",
    )
    args = parser.parse_args()

    if args.api_key:
        init(api_key=args.api_key)
    else:
        init()

    if not args.training_root.exists():
        raise SystemExit(f"Missing training directory: {args.training_root}")
    if not args.test_root.exists():
        raise SystemExit(f"Missing test directory: {args.test_root}")

    percentage_dirs = sorted([path for path in args.training_root.iterdir() if path.is_dir()])
    for percentage_dir in percentage_dirs:
        percentage = percentage_dir.name
        for train_log_path in sorted([path for path in percentage_dir.iterdir() if is_xes_file(path)]):
            test_log_path = args.test_root / train_log_path.name
            if not test_log_path.exists():
                print(f"Skipping {train_log_path.name}: missing test log {test_log_path}")
                continue

            log_name = log_name_from_path(train_log_path)
            result = evaluate_log(
                train_log_path=train_log_path,
                test_log_path=test_log_path,
                seed=args.seed,
                top_classes=args.top_classes,
                max_rows=args.max_rows,
                method_name=args.method_name,
                activity_key=args.activity_key,
                timestamp_key=args.timestamp_key,
            )
            if result.get("accuracy") is None:
                raise RuntimeError(
                    f"Cannot write accuracy for {log_name} at {percentage}: metric is missing."
                )

            output_dir = args.results_dir / percentage / log_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{args.method_name}.json"
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
