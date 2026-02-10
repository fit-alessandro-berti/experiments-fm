#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path

import pm4py

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
BASE_LOGS_DIR = PROJECT_ROOT / "base_logs"
OUTPUT_DIR = PROJECT_ROOT / "logs"
TRAINING_DIR = OUTPUT_DIR / "training"
TEST_DIR = OUTPUT_DIR / "test"
SEED = 42

# (folder_name, percentage)
SAMPLING_LEVELS = [
    ("0005", 0.5),
    ("0010", 1),
    ("0030", 3),
    ("0050", 5),
    ("0100", 10),
    ("0200", 20),
    ("0500", 50),
    ("1000", 100),
]


def output_file_name(log_path: Path) -> str:
    name = log_path.name
    if name.endswith(".xes.gz"):
        stem = name[: -len(".xes.gz")]
    elif name.endswith(".xes"):
        stem = name[: -len(".xes")]
    else:
        stem = log_path.stem
    return f"{stem}.xes.gz"


def discover_logs(base_logs_dir: Path) -> list[Path]:
    log_files = []
    for path in sorted(base_logs_dir.iterdir()):
        if path.is_file() and (path.name.endswith(".xes") or path.name.endswith(".xes.gz")):
            log_files.append(path)
    return log_files


def main() -> None:
    random.seed(SEED)

    if not BASE_LOGS_DIR.exists():
        raise FileNotFoundError(f"Missing input directory: {BASE_LOGS_DIR}")

    log_files = discover_logs(BASE_LOGS_DIR)
    if not log_files:
        print(f"No .xes/.xes.gz files found in {BASE_LOGS_DIR}")
        return

    for folder_name, _ in SAMPLING_LEVELS:
        (TRAINING_DIR / folder_name).mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    for log_path in log_files:
        print(f"Reading: {log_path}")
        log = pm4py.read_xes(str(log_path))
        total_cases = log["case:concept:name"].nunique()
        output_name = output_file_name(log_path)

        if total_cases == 0:
            print(f"Skipping empty log (0 cases): {log_path}")
            continue

        if total_cases == 1:
            train_cases = 1
        else:
            train_cases = int(round(total_cases * 0.8))
            train_cases = min(max(train_cases, 1), total_cases - 1)

        random.seed(f"{SEED}:{log_path.name}:split")
        training_log = pm4py.sample_cases(log, num_cases=train_cases)
        training_case_ids = set(training_log["case:concept:name"].unique())
        test_log = log[~log["case:concept:name"].isin(training_case_ids)]

        test_output_path = TEST_DIR / output_name
        if test_output_path.exists():
            print(f"Skipping existing file: {test_output_path}")
        else:
            test_cases = test_log["case:concept:name"].nunique()
            print(
                f"Writing test split ({test_cases}/{total_cases} cases): {test_output_path}"
            )
            pm4py.write_xes(test_log, str(test_output_path))

        training_total_cases = training_log["case:concept:name"].nunique()

        for folder_name, percentage in SAMPLING_LEVELS:
            output_path = TRAINING_DIR / folder_name / output_name

            if output_path.exists():
                print(f"Skipping existing file: {output_path}")
                continue

            if percentage == 100:
                sampled_log = training_log
                num_cases = training_total_cases
            else:
                num_cases = max(1, round(training_total_cases * (percentage / 100)))
                random.seed(f"{SEED}:{log_path.name}:train:{percentage}")
                sampled_log = pm4py.sample_cases(training_log, num_cases=num_cases)

            print(
                "Writing training "
                f"{percentage}% sample ({num_cases}/{training_total_cases} cases): "
                f"{output_path}"
            )
            pm4py.write_xes(sampled_log, str(output_path))


if __name__ == "__main__":
    main()
