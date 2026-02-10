#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pm4py
from pm4py.util import xes_constants
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from .feature_extraction import build_remaining_time_dataset, fit_feature_space
except ImportError:
    from feature_extraction import build_remaining_time_dataset, fit_feature_space


def is_xes_file(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".xes") or path.name.endswith(".xes.gz"))


def log_name_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".xes.gz"):
        return name[: -len(".xes.gz")]
    if name.endswith(".xes"):
        return name[: -len(".xes")]
    return path.stem


def evaluate_log(
    train_log_path: Path,
    test_log_path: Path,
    seed: int,
    method_name: str,
    activity_key: str,
    timestamp_key: str,
) -> dict:
    train_log = pm4py.read_xes(str(train_log_path), return_legacy_log_object=True)
    test_log = pm4py.read_xes(str(test_log_path), return_legacy_log_object=True)

    feature_space = fit_feature_space(train_log, activity_key=activity_key)
    x_train, y_train = build_remaining_time_dataset(
        train_log,
        feature_space=feature_space,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
    )
    x_test, y_test = build_remaining_time_dataset(
        test_log,
        feature_space=feature_space,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
    )

    result = {
        "method": method_name,
        "train_log": str(train_log_path),
        "test_log": str(test_log_path),
        "train_prefixes": len(x_train),
        "test_prefixes": len(x_test),
        "mae": None,
        "r2": None,
        "status": "ok",
    }

    if not x_train:
        result["status"] = "skipped"
        result["reason"] = "No train prefixes extracted."
        return result
    if not x_test:
        result["status"] = "skipped"
        result["reason"] = "No test prefixes extracted."
        return result

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    result["mae"] = float(mean_absolute_error(y_test, y_pred))
    if len(y_test) >= 2:
        result["r2"] = float(r2_score(y_test, y_pred))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate a Random Forest regressor for remaining time using "
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
    parser.add_argument("--method-name", default="random_forest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activity-key", default=xes_constants.DEFAULT_NAME_KEY)
    parser.add_argument("--timestamp-key", default=xes_constants.DEFAULT_TIMESTAMP_KEY)
    args = parser.parse_args()

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
                method_name=args.method_name,
                activity_key=args.activity_key,
                timestamp_key=args.timestamp_key,
            )

            output_dir = args.results_dir / percentage / log_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{args.method_name}.json"
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
