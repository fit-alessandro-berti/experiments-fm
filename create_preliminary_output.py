#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

TARGET_PERCENTAGE_CODES = ["0005", "0010", "0030", "0050", "0100", "0200", "0500", "1000"]
TARGET_GAP_METHODS = ["tabpfn", "our_fm", "our_fm_knn"]


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
        "linear_regression": 0,
        "random_forest": 1,
        "knn": 2,
        "svm": 3,
        "gaussian_nb": 4,
        "svr": 5,
        "xgboost": 6,
        "lightgbm": 7,
        "tabpfn": 8,
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


def format_percentage(percentage_code: str) -> str:
    try:
        percentage_value = int(percentage_code) / 10.0
    except ValueError:
        return latex_escape(percentage_code)
    if percentage_value.is_integer():
        return f"{int(percentage_value)}\\%"
    return f"{percentage_value:.1f}\\%"


def style_metric(value_text: str, rank: int | None) -> str:
    if rank == 0:
        return rf"\textcolor{{green}}{{\textbf{{{value_text}}}}}"
    if rank == 1:
        return rf"\textcolor{{violet}}{{\textit{{{value_text}}}}}"
    if rank == 2:
        return rf"\textcolor{{orange}}{{{value_text}}}"
    return value_text


def percentage_sort_key(percentage_code: str) -> tuple[int, str]:
    try:
        return int(percentage_code), percentage_code
    except ValueError:
        return 10**9, percentage_code


def ensure_required_percentages(percentages: list[str], required_codes: list[str]) -> list[str]:
    merged = set(percentages)
    merged.update(required_codes)
    return sorted(merged, key=percentage_sort_key)


def compute_gap_to_best(
    model_value: float | None,
    best_value: float | None,
    higher_is_better: bool,
) -> float | None:
    if model_value is None or best_value is None:
        return None

    denominator = abs(best_value)
    if denominator <= 1e-12:
        return 0.0 if abs(model_value - best_value) <= 1e-12 else None

    if higher_is_better:
        difference = ((best_value - model_value) / denominator) * 100.0
    else:
        difference = ((model_value - best_value) / denominator) * 100.0

    if difference < 0 and difference > -1e-9:
        return 0.0
    return difference


def compute_average_percentage_gap(
    methods: list[str],
    logs: list[str],
    percentages: list[str],
    data: dict[str, dict[str, dict[str, dict[str, Any]]]],
    metric_keys: list[str],
    selected_methods: list[str],
    transform: Callable[[float], float] | None = None,
    higher_is_better: bool = True,
) -> dict[str, dict[str, float | None]]:
    averages: dict[str, dict[str, float | None]] = {}

    for percentage in percentages:
        method_to_differences: dict[str, list[float]] = {method: [] for method in selected_methods}
        for log_name in logs:
            row_values: dict[str, float] = {}
            for method in methods:
                payload = data.get(method, {}).get(log_name, {}).get(percentage)
                value = get_numeric_metric(payload, metric_keys)
                if value is not None and transform is not None:
                    value = transform(value)
                if value is not None:
                    row_values[method] = value

            if not row_values:
                continue

            best_value = max(row_values.values()) if higher_is_better else min(row_values.values())
            for method in selected_methods:
                model_value = row_values.get(method)
                difference = compute_gap_to_best(
                    model_value=model_value,
                    best_value=best_value,
                    higher_is_better=higher_is_better,
                )
                if difference is None:
                    continue
                method_to_differences[method].append(difference)

        averages[percentage] = {
            method: (
                sum(values) / len(values)
                if values
                else None
            )
            for method, values in method_to_differences.items()
        }

    return averages


def render_average_percentage_gap_table(
    title: str,
    label: str,
    percentages: list[str],
    selected_methods: list[str],
    averages: dict[str, dict[str, float | None]],
    decimals: int = 2,
) -> str:
    col_spec = "l" + ("c" * len(selected_methods))
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{0.75\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    header_cells = [r"\textbf{Data Fraction}"]
    header_cells.extend([rf"\textbf{{{latex_escape(method)}}}" for method in selected_methods])
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\hline")

    for percentage in percentages:
        row_cells = [format_percentage(percentage)]
        for method in selected_methods:
            value = averages.get(percentage, {}).get(method)
            row_cells.append("" if value is None else f"{value:.{decimals}f}\\%")
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


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
    gap_methods: list[str] | None = None,
    gap_decimals: int = 2,
) -> str:
    color_legend = (
        r"\textcolor{green}{\textbf{green}} = best, "
        r"\textcolor{violet}{\textit{violet}} = second best, "
        r"\textcolor{orange}{orange} = third best."
    )
    gap_methods = gap_methods or []
    col_spec = "ll" + ("c" * len(methods))
    if gap_methods:
        col_spec += "|" + ("c" * len(gap_methods)) + "|"
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{title} {color_legend}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{0.75\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    header_cells = [r"\textbf{Event Log}", r"\textbf{Data Fraction}"]
    header_cells.extend([rf"\textbf{{{latex_escape(method)}}}" for method in methods])
    for method in gap_methods:
        header_cells.append(rf"\textbf{{{latex_escape(method)} gap}}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\hline")

    for log_name in logs:
        for percentage_idx, percentage in enumerate(percentages):
            value_by_method: dict[str, float | None] = {}
            row_values: list[float | None] = []
            for method in methods:
                payload = data.get(method, {}).get(log_name, {}).get(percentage)
                value = get_numeric_metric(payload, metric_keys)
                if value is not None and transform is not None:
                    value = transform(value)
                value_by_method[method] = value
                row_values.append(value)

            present_values = [value for value in row_values if value is not None]
            unique_values = sorted(set(present_values), reverse=higher_is_better)
            top_values = unique_values[:3]
            row_ranks = {value: rank for rank, value in enumerate(top_values)}
            best_value = unique_values[0] if unique_values else None

            row_cells: list[str] = []
            if percentage_idx == 0:
                row_cells.append(
                    rf"\multirow{{{len(percentages)}}}{{*}}{{\textbf{{{latex_escape(log_name)}}}}}"
                )
            else:
                row_cells.append("")
            row_cells.append(format_percentage(percentage))

            for value in row_values:
                value_text = format_metric(value, decimals)
                if value is None:
                    row_cells.append(value_text)
                    continue
                row_cells.append(style_metric(value_text, row_ranks.get(value)))

            for method in gap_methods:
                gap_value = compute_gap_to_best(
                    model_value=value_by_method.get(method),
                    best_value=best_value,
                    higher_is_better=higher_is_better,
                )
                row_cells.append("" if gap_value is None else f"{gap_value:.{gap_decimals}f}\\%")

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
        default=["0005", "0010", "0030", "0050", "0100", "0200", "0500", "1000"],
        help="Percentage-folder codes to include as rows.",
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        help="Optional explicit list of log names (without extension).",
    )
    args = parser.parse_args()

    table_percentages = ensure_required_percentages(args.percentages, required_codes=["0100"])
    discovered_logs = discover_logs(args.base_logs_dir)
    logs = args.logs if args.logs else discovered_logs
    if not logs:
        raise SystemExit("No logs available. Provide --logs or ensure base_logs contains .xes/.xes.gz files.")

    cls_methods, cls_data = load_results(args.classification_results)
    reg_methods, reg_data = load_results(args.regression_results)

    classification_methods = sorted(cls_methods, key=method_sort_key)
    regression_methods = sorted(reg_methods, key=method_sort_key)

    table_accuracy = render_table(
        title="Classification accuracy.",
        label="tab:classification-accuracy",
        methods=classification_methods,
        logs=logs,
        percentages=table_percentages,
        data=cls_data,
        metric_keys=["accuracy"],
        decimals=3,
        gap_methods=["our_fm", "our_fm_knn"],
    )

    table_mae = render_table(
        title="Regression MAE.",
        label="tab:regression-mae",
        methods=regression_methods,
        logs=logs,
        percentages=table_percentages,
        data=reg_data,
        metric_keys=["mae", "mae_seconds"],
        decimals=2,
        transform=lambda value: value / 3600.0,
        higher_is_better=False,
        gap_methods=["our_fm", "our_fm_knn"],
    )

    table_r2 = render_table(
        title=r"Regression $R^2$.",
        label="tab:regression-r2",
        methods=regression_methods,
        logs=logs,
        percentages=table_percentages,
        data=reg_data,
        metric_keys=["r2"],
        decimals=3,
    )

    gap_percentages = TARGET_PERCENTAGE_CODES
    classification_gap_averages = compute_average_percentage_gap(
        methods=classification_methods,
        logs=logs,
        percentages=gap_percentages,
        data=cls_data,
        metric_keys=["accuracy"],
        selected_methods=TARGET_GAP_METHODS,
        higher_is_better=True,
    )
    table_accuracy_gap = render_average_percentage_gap_table(
        title="Average percentage gap to the best classification model (lower is better).",
        label="tab:classification-gap-to-best",
        percentages=gap_percentages,
        selected_methods=TARGET_GAP_METHODS,
        averages=classification_gap_averages,
    )

    mae_gap_averages = compute_average_percentage_gap(
        methods=regression_methods,
        logs=logs,
        percentages=gap_percentages,
        data=reg_data,
        metric_keys=["mae", "mae_seconds"],
        selected_methods=TARGET_GAP_METHODS,
        transform=lambda value: value / 3600.0,
        higher_is_better=False,
    )
    table_mae_gap = render_average_percentage_gap_table(
        title="Average percentage gap to the best MAE model (lower is better).",
        label="tab:mae-gap-to-best",
        percentages=gap_percentages,
        selected_methods=TARGET_GAP_METHODS,
        averages=mae_gap_averages,
    )

    header = "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage{graphicx} % Required for inserting images",
            r"\usepackage{xcolor}",
            r"\usepackage{multirow}",
            "",
            r"\title{Trial123}",
            r"\author{a.berti }",
            r"\date{February 2026}",
            "",
            r"\begin{document}",
            "",
            r"\maketitle",
            "",
            r"\section{Introduction}",
            "",
        ]
    )

    sections = [
        table_accuracy + "\n\n" + table_accuracy_gap,
        table_mae + "\n\n" + table_mae_gap,
        table_r2,
    ]
    body = "\n\n\\clearpage\\newpage\n\n".join(sections)
    tex_content = header + body + "\n\n\\end{document}\n"
    output_path.write_text(tex_content, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
