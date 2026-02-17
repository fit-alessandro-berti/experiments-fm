#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any


def strip_xes_suffix(name: str) -> str:
    if name.endswith(".xes.gz"):
        return name[: -len(".xes.gz")]
    if name.endswith(".xes"):
        return name[: -len(".xes")]
    return Path(name).stem


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for src, dst in replacements.items():
        escaped = escaped.replace(src, dst)
    return escaped


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def discover_logs(base_logs_dir: Path) -> list[str]:
    if not base_logs_dir.exists():
        return []
    logs = []
    for path in sorted(base_logs_dir.iterdir()):
        if path.is_file() and (path.name.endswith(".xes") or path.name.endswith(".xes.gz")):
            logs.append(strip_xes_suffix(path.name))
    return logs


def load_results(results_dir: Path) -> tuple[set[str], dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    methods: set[str] = set()
    data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    if not results_dir.exists():
        return methods, data

    for percentage_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        percentage = percentage_dir.name
        for log_dir in sorted([p for p in percentage_dir.iterdir() if p.is_dir()]):
            log_name = log_dir.name
            for json_path in sorted([p for p in log_dir.iterdir() if p.is_file() and p.suffix == ".json"]):
                payload = read_json(json_path)
                if payload is None:
                    continue
                method_name = str(payload.get("method") or json_path.stem)
                methods.add(method_name)
                data.setdefault(method_name, {}).setdefault(log_name, {})[percentage] = payload

    return methods, data


def method_sort_key(method_name: str) -> tuple[int, str]:
    priority = {
        "random_forest": 0,
        "knn": 1,
        "svm": 2,
        "svr": 3,
        "xgboost": 4,
        "lightgbm": 5,
        "tabpfn": 6,
    }
    return priority.get(method_name, 100), method_name


def get_numeric_metric(payload: dict[str, Any] | None, candidate_keys: list[str]) -> float | None:
    if not payload:
        return None
    if payload.get("status") not in (None, "ok"):
        return None
    for key in candidate_keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return None


def format_metric(value: float | None, decimals: int) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def style_metric(value_text: str, rank: int | None) -> str:
    if rank == 0:
        return rf"\textcolor{{green}}{{\textbf{{{value_text}}}}}"
    if rank == 1:
        return rf"\textcolor{{violet}}{{\textit{{{value_text}}}}}"
    if rank == 2:
        return rf"\textcolor{{orange}}{{{value_text}}}"
    return value_text


def render_table(
    title: str,
    label: str,
    methods: list[str],
    logs: list[str],
    percentages: list[str],
    data: dict[str, dict[str, dict[str, dict[str, Any]]]],
    metric_keys: list[str],
    decimals: int,
    transform: Callable[[float], float] | None = None,
    higher_is_better: bool = True,
) -> str:
    column_count = len(logs) * len(percentages)
    col_spec = "l" + ("c" * column_count)
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    first_header_cells = [r"\textbf{Method}"]
    for log_name in logs:
        first_header_cells.append(
            rf"\multicolumn{{{len(percentages)}}}{{c}}{{\textbf{{{latex_escape(log_name)}}}}}"
        )
    lines.append(" & ".join(first_header_cells) + r" \\")

    second_header_cells = [""]
    for _log_name in logs:
        second_header_cells.extend([latex_escape(percentage) for percentage in percentages])
    lines.append(" & ".join(second_header_cells) + r" \\")
    lines.append(r"\hline")

    values_by_method: list[list[float | None]] = []
    for method in methods:
        row_values: list[float | None] = []
        for log_name in logs:
            for percentage in percentages:
                payload = data.get(method, {}).get(log_name, {}).get(percentage)
                value = get_numeric_metric(payload, metric_keys)
                if value is not None and transform is not None:
                    value = transform(value)
                row_values.append(value)
        values_by_method.append(row_values)

    ranks_by_value_per_column: list[dict[float, int]] = []
    for column_idx in range(column_count):
        present_values = [
            method_values[column_idx]
            for method_values in values_by_method
            if method_values[column_idx] is not None
        ]
        unique_values = sorted(set(present_values), reverse=higher_is_better)
        top_values = unique_values[:3]
        ranks_by_value_per_column.append({value: rank for rank, value in enumerate(top_values)})

    for method_idx, method in enumerate(methods):
        row_cells = [latex_escape(method)]
        for column_idx, value in enumerate(values_by_method[method_idx]):
            value_text = format_metric(value, decimals)
            if value is None:
                row_cells.append(value_text)
                continue
            rank = ranks_by_value_per_column[column_idx].get(value)
            row_cells.append(style_metric(value_text, rank))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    output_path = repo_root / "preliminary_output.tex"

    parser = argparse.ArgumentParser(
        description="Build LaTeX summary tables from classification/regression experiment JSON results."
    )
    parser.add_argument(
        "--classification-results",
        type=Path,
        default=repo_root / "next_activity_classification" / "results",
    )
    parser.add_argument(
        "--regression-results",
        type=Path,
        default=repo_root / "remaining_time_prediction" / "results",
    )
    parser.add_argument(
        "--base-logs-dir",
        type=Path,
        default=repo_root / "base_logs",
    )
    parser.add_argument(
        "--percentages",
        nargs="+",
        default=["0050", "0200", "1000"],
        help="Percentage-folder codes to include as columns.",
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        help="Optional explicit list of log names (without extension).",
    )
    args = parser.parse_args()

    discovered_logs = discover_logs(args.base_logs_dir)
    logs = args.logs if args.logs else discovered_logs
    if not logs:
        raise SystemExit("No logs available. Provide --logs or ensure base_logs contains .xes/.xes.gz files.")

    cls_methods, cls_data = load_results(args.classification_results)
    reg_methods, reg_data = load_results(args.regression_results)

    all_methods = sorted(cls_methods.union(reg_methods), key=method_sort_key)

    table_accuracy = render_table(
        title="Classification accuracy.",
        label="tab:classification-accuracy",
        methods=all_methods,
        logs=logs,
        percentages=args.percentages,
        data=cls_data,
        metric_keys=["accuracy"],
        decimals=3,
    )

    table_mae = render_table(
        title="Regression MAE.",
        label="tab:regression-mae",
        methods=all_methods,
        logs=logs,
        percentages=args.percentages,
        data=reg_data,
        metric_keys=["mae", "mae_seconds"],
        decimals=2,
        transform=lambda value: value / 3600.0,
        higher_is_better=False,
    )

    table_r2 = render_table(
        title=r"Regression $R^2$.",
        label="tab:regression-r2",
        methods=all_methods,
        logs=logs,
        percentages=args.percentages,
        data=reg_data,
        metric_keys=["r2"],
        decimals=3,
    )

    tex_content = "\n\n".join([table_accuracy, table_mae, table_r2]) + "\n"
    output_path.write_text(tex_content, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
