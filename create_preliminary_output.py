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
    color_legend = (
        r"\textcolor{green}{\textbf{green}} = best, "
        r"\textcolor{violet}{\textit{violet}} = second best, "
        r"\textcolor{orange}{orange} = third best."
    )
    col_spec = "ll" + ("c" * len(methods))
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{title} {color_legend}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    header_cells = [r"\textbf{Event Log}", r"\textbf{Data Fraction}"]
    header_cells.extend([rf"\textbf{{{latex_escape(method)}}}" for method in methods])
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\hline")

    for log_name in logs:
        for percentage_idx, percentage in enumerate(percentages):
            row_values: list[float | None] = []
            for method in methods:
                payload = data.get(method, {}).get(log_name, {}).get(percentage)
                value = get_numeric_metric(payload, metric_keys)
                if value is not None and transform is not None:
                    value = transform(value)
                row_values.append(value)

            present_values = [value for value in row_values if value is not None]
            unique_values = sorted(set(present_values), reverse=higher_is_better)
            top_values = unique_values[:3]
            row_ranks = {value: rank for rank, value in enumerate(top_values)}

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
        percentages=args.percentages,
        data=cls_data,
        metric_keys=["accuracy"],
        decimals=3,
    )

    table_mae = render_table(
        title="Regression MAE.",
        label="tab:regression-mae",
        methods=regression_methods,
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
        methods=regression_methods,
        logs=logs,
        percentages=args.percentages,
        data=reg_data,
        metric_keys=["r2"],
        decimals=3,
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

    body = "\n\n\\clearpage\\newpage\n\n".join([table_accuracy, table_mae, table_r2])
    tex_content = header + body + "\n\n\\end{document}\n"
    output_path.write_text(tex_content, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
