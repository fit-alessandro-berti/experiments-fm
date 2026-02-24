#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

TARGET_PERCENTAGE_CODES = ["0005", "0010", "0030", "0050", "0100", "0200", "0500", "1000"]
TABLE_PERCENTAGE_CODES = ["0005", "0030", "0050", "0100", "0200", "1000"]
TARGET_CLASSIFICATION_BUCKET_LOGS = ["receipt", "roadtraffic", "sepsis"]
TARGET_GAP_METHODS = ["tabpfn", "knn", "our_fm", "our_fm_knn"]
GAP_METHOD_LABELS = {
    "our_fm": "fm",
    "our_fm_knn": "fm_knn",
}
PLOT_METHOD_COLORS = {
    "knn": "teal",
    "tabpfn": "blue",
    "our_fm": "red",
    "our_fm_knn": "orange",
}
THICK_LINE_METHODS = {"our_fm", "our_fm_knn"}
DISTANCE_CORR_COLUMN = "Distance correlation(activity_block, time_block)"
MI_MEAN_COLUMN = "MI(activity_indicators, time_features) mean"
MI_PREV_WEIGHTED_COLUMN = "MI(activity_indicators, time_features) prevalence-weighted mean"
ACT_TIME_CORR_COLUMNS = [DISTANCE_CORR_COLUMN, MI_MEAN_COLUMN, MI_PREV_WEIGHTED_COLUMN]
MODEL_COLUMN_SPEC = r">{\centering\arraybackslash}p{1cm}"
THIN_MODEL_VRULE = r"!{\vrule width 0.25pt}"
THICK_GAP_VRULE = r"!{\vrule width 0.9pt}"


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


def format_method_header_label(method_name: str) -> str:
    return latex_escape(method_name.replace("_", " "))


def with_vertical_rules(column_specs: list[str]) -> str:
    return "|" + "|".join(column_specs) + "|"


def build_ml_table_col_spec(prefix_columns: list[str], model_count: int, gap_count: int = 0) -> str:
    spec = "|" + "|".join(prefix_columns)

    if model_count > 0:
        spec += THIN_MODEL_VRULE + MODEL_COLUMN_SPEC
        for _ in range(1, model_count):
            spec += THIN_MODEL_VRULE + MODEL_COLUMN_SPEC

    if gap_count > 0:
        spec += THICK_GAP_VRULE + MODEL_COLUMN_SPEC
        for _ in range(1, gap_count):
            spec += THICK_GAP_VRULE + MODEL_COLUMN_SPEC
        spec += THICK_GAP_VRULE
    else:
        spec += "|"

    return spec


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def parse_numeric(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value.strip())
    except (TypeError, ValueError, AttributeError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_confidence_range_median(range_text: str | None) -> float | None:
    if not range_text:
        return None
    stripped = range_text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return None
    body = stripped[1:-1]
    parts = [part.strip() for part in body.split(",")]
    if len(parts) != 2:
        return None
    low = parse_numeric(parts[0])
    high = parse_numeric(parts[1])
    if low is None or high is None:
        return None
    return (low + high) / 2.0


def load_bucket_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        return {}

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}

    header = lines[0].split(";")
    if header and header[-1] == "":
        header = header[:-1]

    buckets_by_log: dict[str, list[dict[str, str]]] = {}
    for line in lines[1:]:
        fields = line.split(";")
        if fields and fields[-1] == "":
            fields = fields[:-1]
        if len(fields) < len(header):
            fields = fields + [""] * (len(header) - len(fields))
        row = {key: fields[idx] for idx, key in enumerate(header)}
        log_name = row.get("log", "").strip()
        if not log_name:
            continue
        buckets_by_log.setdefault(log_name, []).append(row)

    for log_name, rows in buckets_by_log.items():
        rows.sort(key=lambda row: int(row.get("bucket", "0")) if row.get("bucket", "").isdigit() else 0)
        buckets_by_log[log_name] = rows

    return buckets_by_log


def load_semicolon_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []

    header = lines[0].split(";")
    if header and header[-1] == "":
        header = header[:-1]

    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        fields = line.split(";")
        if fields and fields[-1] == "":
            fields = fields[:-1]
        if len(fields) < len(header):
            fields = fields + [""] * (len(header) - len(fields))
        rows.append({key: fields[idx] for idx, key in enumerate(header)})

    return rows


def load_comma_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []

    header = lines[0].split(",")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) < len(header):
            fields = fields + [""] * (len(header) - len(fields))
        rows.append({key: fields[idx] for idx, key in enumerate(header)})

    return rows


def load_act_time_corr_metrics(path: Path) -> dict[str, dict[str, float]]:
    rows = load_semicolon_table(path)
    metrics_by_log: dict[str, dict[str, float]] = {}
    for row in rows:
        log_name = row.get("log", "").strip()
        if not log_name:
            continue

        parsed_row: dict[str, float] = {}
        for column in ACT_TIME_CORR_COLUMNS:
            value = parse_numeric(row.get(column))
            if value is not None:
                parsed_row[column] = value
        metrics_by_log[log_name] = parsed_row

    return metrics_by_log


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


def method_line_thickness(method_name: str) -> str:
    return "line width=2.6pt" if method_name in THICK_LINE_METHODS else "thick"


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


def percentage_code_to_fraction(percentage_code: str) -> float | None:
    try:
        return int(percentage_code) / 10.0
    except ValueError:
        return None


def style_metric(value_text: str, rank: int | None) -> str:
    if rank == 0:
        return rf"\textcolor{{green}}{{\textbf{{{value_text}}}}}"
    if rank == 1:
        return rf"\textcolor{{violet}}{{\textit{{{value_text}}}}}"
    if rank == 2:
        return rf"\textcolor{{orange}}{{{value_text}}}"
    return value_text


def join_for_caption(items: list[str]) -> str:
    escaped_items = [latex_escape(item) for item in items]
    if not escaped_items:
        return ""
    if len(escaped_items) == 1:
        return escaped_items[0]
    if len(escaped_items) == 2:
        return f"{escaped_items[0]} and {escaped_items[1]}"
    return ", ".join(escaped_items[:-1]) + f", and {escaped_items[-1]}"


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


def compute_average_percentage_gap_by_log(
    methods: list[str],
    logs: list[str],
    percentages: list[str],
    data: dict[str, dict[str, dict[str, dict[str, Any]]]],
    metric_keys: list[str],
    selected_methods: list[str],
    transform: Callable[[float], float] | None = None,
    higher_is_better: bool = True,
) -> dict[str, float | None]:
    averages_by_log: dict[str, float | None] = {}

    for log_name in logs:
        differences: list[float] = []
        for percentage in percentages:
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
                if difference is not None:
                    differences.append(difference)

        averages_by_log[log_name] = (sum(differences) / len(differences)) if differences else None

    return averages_by_log


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None

    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)

    centered_x = [value - mean_x for value in x_values]
    centered_y = [value - mean_y for value in y_values]

    variance_x = sum(value * value for value in centered_x)
    variance_y = sum(value * value for value in centered_y)
    denominator = math.sqrt(variance_x * variance_y)
    if denominator <= 1e-12:
        return None

    covariance = sum(x_val * y_val for x_val, y_val in zip(centered_x, centered_y))
    return covariance / denominator


def compute_metric_gap_correlations(
    metrics_by_log: dict[str, dict[str, float]],
    gap_by_log: dict[str, float | None],
    metric_columns: list[str],
) -> dict[str, tuple[float | None, int]]:
    results: dict[str, tuple[float | None, int]] = {}

    for metric_name in metric_columns:
        x_values: list[float] = []
        y_values: list[float] = []
        for log_name, metric_values in metrics_by_log.items():
            metric_value = metric_values.get(metric_name)
            gap_value = gap_by_log.get(log_name)
            if metric_value is None or gap_value is None:
                continue
            x_values.append(metric_value)
            y_values.append(gap_value)

        correlation = pearson_correlation(x_values, y_values)
        results[metric_name] = (correlation, len(x_values))

    return results


def render_average_percentage_gap_table(
    title: str,
    label: str,
    percentages: list[str],
    selected_methods: list[str],
    averages: dict[str, dict[str, float | None]],
    decimals: int = 2,
) -> str:
    col_spec = build_ml_table_col_spec(prefix_columns=["l"], model_count=len(selected_methods))
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{0.5\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    header_cells = [r"\textbf{\%}"]
    header_cells.extend([rf"\textbf{{{format_method_header_label(method)}}}" for method in selected_methods])
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


def render_metric_gap_correlation_table(
    title: str,
    label: str,
    correlations: dict[str, tuple[float | None, int]],
    metric_columns: list[str],
    decimals: int = 3,
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{0.85\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{|p{0.60\textwidth}|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Metric} & \textbf{Pearson $r$} & \textbf{Logs used} \\")
    lines.append(r"\hline")

    for metric_name in metric_columns:
        correlation, sample_size = correlations.get(metric_name, (None, 0))
        corr_text = "" if correlation is None else f"{correlation:.{decimals}f}"
        lines.append(
            rf"{latex_escape(metric_name)} & {corr_text} & {sample_size} \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_csv_table(
    title: str,
    label: str,
    rows: list[dict[str, str]],
    column_order: list[str],
    column_labels: dict[str, str],
) -> str:
    col_spec = with_vertical_rules(["l"] + ["c"] * (len(column_order) - 1))
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    header_cells = [rf"\textbf{{{latex_escape(column_labels.get(col, col))}}}" for col in column_order]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\hline")

    for row in rows:
        row_cells = [latex_escape(row.get(col, "")) for col in column_order]
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_average_percentage_gap_tikz_plot(
    label: str,
    percentages: list[str],
    selected_methods: list[str],
    averages: dict[str, dict[str, float | None]],
    decimals: int = 2,
    title: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{figure}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")

    tick_label_by_position: dict[float, str] = {}
    for percentage in percentages:
        fraction = percentage_code_to_fraction(percentage)
        if fraction is not None:
            tick_label_by_position[math.log1p(fraction)] = f"{fraction:g}"

    unique_xtick_positions = sorted(tick_label_by_position.keys())
    xtick_text = ",".join(f"{tick:.6f}" for tick in unique_xtick_positions)
    xtick_labels_text = ",".join(tick_label_by_position[tick] for tick in unique_xtick_positions)

    method_to_coordinates: dict[str, list[str]] = {}
    min_gap: float | None = None
    for method in selected_methods:
        coordinates: list[str] = []
        for percentage in percentages:
            fraction = percentage_code_to_fraction(percentage)
            if fraction is None:
                continue
            gap_value = averages.get(percentage, {}).get(method)
            if gap_value is None:
                continue
            x_value = math.log1p(fraction)
            min_gap = gap_value if min_gap is None else min(min_gap, gap_value)
            coordinates.append(f"({x_value:.6f},{gap_value:.{decimals}f})")
        method_to_coordinates[method] = coordinates

    axis_options = [
        r"width=0.8\textwidth",
        r"height=0.45\textwidth",
        r"xlabel={log(1 + Data Fraction (\%))}",
        r"ylabel={Average Gap to Best (\%)}",
        r"grid=major",
        r"legend style={at={(0.5,-0.22)}, anchor=north}",
        r"legend columns=2",
        f"ymin={(min_gap if min_gap is not None else 0.0):.{decimals}f}",
    ]
    if xtick_text:
        axis_options.append(rf"xtick={{{xtick_text}}}")
    if xtick_labels_text:
        axis_options.append(rf"xticklabels={{{xtick_labels_text}}}")

    lines.append(r"\begin{axis}[")
    lines.append(",\n".join(axis_options))
    lines.append(r"]")

    for method in selected_methods:
        coordinates = method_to_coordinates.get(method, [])
        if not coordinates:
            continue

        color = PLOT_METHOD_COLORS.get(method, "black")
        line_thickness = method_line_thickness(method)
        lines.append(
            rf"\addplot+[smooth, {line_thickness}, color={color}, mark=*] coordinates {{ {' '.join(coordinates)} }};"
        )
        lines.append(rf"\addlegendentry{{{latex_escape(method)}}}")

    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    if title is None:
        method_list = join_for_caption(selected_methods)
        auto_title = (
            "Average percentage gap to the best classification model by data fraction "
            f"(methods: {method_list}; lower is better)."
        )
    else:
        auto_title = title
    lines.append(rf"\caption{{{auto_title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def render_classification_bucket_dual_tikz_plot(
    log_name: str,
    bucket_rows: list[dict[str, str]],
    label: str,
    title: str | None = None,
    decimals: int = 2,
) -> str:
    fm_fixed_coords: list[str] = []
    rf_fixed_coords: list[str] = []
    fm_range_coords: list[str] = []
    rf_range_coords: list[str] = []

    for row in bucket_rows:
        bucket_value = parse_numeric(row.get("bucket"))
        if bucket_value is None:
            continue

        fm_accuracy = parse_numeric(row.get("fm_classification_accuracy"))
        rf_accuracy = parse_numeric(row.get("classic_rf_accuracy"))

        if fm_accuracy is not None:
            fm_fixed_coords.append(f"({bucket_value:.0f},{(fm_accuracy * 100.0):.{decimals}f})")
        if rf_accuracy is not None:
            rf_fixed_coords.append(f"({bucket_value:.0f},{(rf_accuracy * 100.0):.{decimals}f})")

        fm_median = parse_confidence_range_median(row.get("fm_classification_bucket_range"))
        rf_median = parse_confidence_range_median(row.get("classic_rf_classification_bucket_range"))

        if fm_accuracy is not None and fm_median is not None:
            fm_range_coords.append(f"({math.log1p(fm_median):.6f},{(fm_accuracy * 100.0):.{decimals}f})")
        if rf_accuracy is not None and rf_median is not None:
            rf_range_coords.append(f"({math.log1p(rf_median):.6f},{(rf_accuracy * 100.0):.{decimals}f})")

    if title is None:
        auto_title = (
            f"Classification bucket-wise accuracy comparison for the {latex_escape(log_name)} event log. "
            "Left: fixed bucket IDs. Right: median confidence positions with log(1+X)."
        )
    else:
        auto_title = title

    lines: list[str] = []
    lines.append(r"\begin{figure}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{minipage}[t]{0.49\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(",\n".join(
        [
            r"width=\textwidth",
            r"height=0.65\textwidth",
            r"title={Fixed Buckets}",
            r"xlabel={Bucket}",
            r"ylabel={Accuracy (\%)}",
            r"grid=major",
            r"ymin=0",
            r"ymax=100",
            r"xtick={1,2,3,4,5}",
            r"legend style={at={(0.5,-0.25)}, anchor=north}",
            r"legend columns=2",
        ]
    ))
    lines.append(r"]")
    if fm_fixed_coords:
        lines.append(
            rf"\addplot+[smooth, line width=2.6pt, color=red, mark=*] coordinates {{ {' '.join(fm_fixed_coords)} }};"
        )
        lines.append(r"\addlegendentry{fm}")
    if rf_fixed_coords:
        lines.append(
            rf"\addplot+[smooth, thick, color=blue, mark=*] coordinates {{ {' '.join(rf_fixed_coords)} }};"
        )
        lines.append(r"\addlegendentry{classic\_rf}")
    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{minipage}")
    lines.append(r"\hfill")
    lines.append(r"\begin{minipage}[t]{0.49\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(",\n".join(
        [
            r"width=\textwidth",
            r"height=0.65\textwidth",
            r"title={Confidence-Range Medians}",
            r"xlabel={log(1 + median confidence)}",
            r"ylabel={Accuracy (\%)}",
            r"grid=major",
            r"ymin=0",
            r"ymax=100",
        ]
    ))
    lines.append(r"]")
    if fm_range_coords:
        lines.append(
            rf"\addplot+[smooth, line width=2.6pt, color=red, mark=*] coordinates {{ {' '.join(fm_range_coords)} }};"
        )
    if rf_range_coords:
        lines.append(
            rf"\addplot+[smooth, thick, color=blue, mark=*] coordinates {{ {' '.join(rf_range_coords)} }};"
        )
    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{minipage}")
    lines.append(rf"\caption{{{auto_title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def collect_classification_bucket_coordinates(
    bucket_rows: list[dict[str, str]],
    decimals: int = 2,
) -> tuple[list[str], list[str], list[str], list[str]]:
    fm_fixed_coords: list[str] = []
    rf_fixed_coords: list[str] = []
    fm_range_coords: list[str] = []
    rf_range_coords: list[str] = []

    for row in bucket_rows:
        bucket_value = parse_numeric(row.get("bucket"))
        if bucket_value is None:
            continue

        fm_accuracy = parse_numeric(row.get("fm_classification_accuracy"))
        rf_accuracy = parse_numeric(row.get("classic_rf_accuracy"))

        if fm_accuracy is not None:
            fm_fixed_coords.append(f"({bucket_value:.0f},{(fm_accuracy * 100.0):.{decimals}f})")
        if rf_accuracy is not None:
            rf_fixed_coords.append(f"({bucket_value:.0f},{(rf_accuracy * 100.0):.{decimals}f})")

        fm_median = parse_confidence_range_median(row.get("fm_classification_bucket_range"))
        rf_median = parse_confidence_range_median(row.get("classic_rf_classification_bucket_range"))
        if fm_accuracy is not None and fm_median is not None:
            fm_range_coords.append(f"({math.log1p(fm_median):.6f},{(fm_accuracy * 100.0):.{decimals}f})")
        if rf_accuracy is not None and rf_median is not None:
            rf_range_coords.append(f"({math.log1p(rf_median):.6f},{(rf_accuracy * 100.0):.{decimals}f})")

    return fm_fixed_coords, rf_fixed_coords, fm_range_coords, rf_range_coords


def render_classification_bucket_compact_figure(
    bucket_rows_by_log: dict[str, list[dict[str, str]]],
    selected_logs: list[str],
    label: str,
    decimals: int = 2,
) -> str | None:
    available_logs = [log_name for log_name in selected_logs if log_name in bucket_rows_by_log]
    if not available_logs:
        available_logs = sorted(bucket_rows_by_log.keys())
    if not available_logs:
        return None

    lines: list[str] = []
    lines.append(r"\begin{figure}[ht]")
    lines.append(r"\centering")

    lines.append(r"\begin{minipage}[t]{\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\textbf{Fixed Buckets}\\[2pt]")
    for idx, log_name in enumerate(available_logs):
        fm_fixed, rf_fixed, _, _ = collect_classification_bucket_coordinates(
            bucket_rows=bucket_rows_by_log[log_name],
            decimals=decimals,
        )
        lines.append(r"\begin{minipage}[t]{0.30\textwidth}")
        lines.append(r"\centering")
        axis_options = [
            r"width=\textwidth",
            r"height=0.62\textwidth",
            rf"title={{{latex_escape(log_name)}}}",
            r"xlabel={Bucket}",
            r"grid=major",
            r"scale only axis",
            r"ymin=0",
            r"ymax=100",
            r"xtick={1,2,3,4,5}",
            r"tick label style={font=\tiny}",
            r"label style={font=\scriptsize}",
            r"title style={font=\scriptsize}",
            r"legend style={at={(0.5,-0.30)}, anchor=north, font=\tiny, draw=none, fill=white}",
            r"legend columns=1",
        ]
        if idx == 0:
            axis_options.append(r"ylabel={Accuracy (\%)}")

        lines.append(r"\begin{tikzpicture}")
        lines.append(r"\begin{axis}[")
        lines.append(",\n".join(axis_options))
        lines.append(r"]")
        if fm_fixed:
            lines.append(
                rf"\addplot+[smooth, line width=2.6pt, color=red, mark=*] coordinates {{ {' '.join(fm_fixed)} }};"
            )
            lines.append(r"\addlegendentry{fm}")
        if rf_fixed:
            lines.append(
                rf"\addplot+[smooth, thick, color=blue, mark=*] coordinates {{ {' '.join(rf_fixed)} }};"
            )
            lines.append(r"\addlegendentry{classic\_rf}")
        lines.append(r"\end{axis}")
        lines.append(r"\end{tikzpicture}")
        lines.append(r"\end{minipage}")
        if idx < len(available_logs) - 1:
            lines.append(r"\hspace{0.02\textwidth}")
    lines.append(r"\end{minipage}")

    lines.append(r"\vspace{16pt}")
    lines.append(r"\begin{minipage}[t]{\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\textbf{Confidence-Range Medians}\\[2pt]")
    for idx, log_name in enumerate(available_logs):
        _, _, fm_range, rf_range = collect_classification_bucket_coordinates(
            bucket_rows=bucket_rows_by_log[log_name],
            decimals=decimals,
        )
        lines.append(r"\begin{minipage}[t]{0.30\textwidth}")
        lines.append(r"\centering")
        axis_options = [
            r"width=\textwidth",
            r"height=0.62\textwidth",
            rf"title={{{latex_escape(log_name)}}}",
            r"xlabel={log(1+median conf.)}",
            r"grid=major",
            r"scale only axis",
            r"ymin=0",
            r"ymax=100",
            r"tick label style={font=\tiny}",
            r"label style={font=\scriptsize}",
            r"title style={font=\scriptsize}",
            r"legend style={at={(0.5,-0.30)}, anchor=north, font=\tiny, draw=none, fill=white}",
            r"legend columns=1",
        ]
        if idx == 0:
            axis_options.append(r"ylabel={Accuracy (\%)}")

        lines.append(r"\begin{tikzpicture}")
        lines.append(r"\begin{axis}[")
        lines.append(",\n".join(axis_options))
        lines.append(r"]")
        if fm_range:
            lines.append(
                rf"\addplot+[smooth, line width=2.6pt, color=red, mark=*] coordinates {{ {' '.join(fm_range)} }};"
            )
            lines.append(r"\addlegendentry{fm}")
        if rf_range:
            lines.append(
                rf"\addplot+[smooth, thick, color=blue, mark=*] coordinates {{ {' '.join(rf_range)} }};"
            )
            lines.append(r"\addlegendentry{classic\_rf}")
        lines.append(r"\end{axis}")
        lines.append(r"\end{tikzpicture}")
        lines.append(r"\end{minipage}")
        if idx < len(available_logs) - 1:
            lines.append(r"\hspace{0.02\textwidth}")
    lines.append(r"\end{minipage}")

    lines.append(
        r"\caption{Compact classification confidence-bucket comparison for receipt, roadtraffic, and sepsis. Left minipage: fixed buckets. Right minipage: confidence-range medians.}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def render_regression_mae_bucket_dual_tikz_plot(
    log_name: str,
    bucket_rows: list[dict[str, str]],
    label: str,
    title: str | None = None,
    decimals: int = 2,
) -> str:
    fm_fixed_coords: list[str] = []
    rf_fixed_coords: list[str] = []
    fm_range_coords: list[str] = []
    rf_range_coords: list[str] = []

    for row in bucket_rows:
        bucket_value = parse_numeric(row.get("bucket"))
        if bucket_value is None:
            continue

        fm_mae = parse_numeric(row.get("fm_mae"))
        rf_mae = parse_numeric(row.get("classic_rf_mae"))

        if fm_mae is not None:
            fm_fixed_coords.append(f"({bucket_value:.0f},{fm_mae:.{decimals}f})")
        if rf_mae is not None:
            rf_fixed_coords.append(f"({bucket_value:.0f},{rf_mae:.{decimals}f})")

        fm_median = parse_confidence_range_median(row.get("fm_regression_bucket_range"))
        rf_median = parse_confidence_range_median(row.get("classic_rf_regression_bucket_range"))

        if fm_mae is not None and fm_median is not None:
            fm_range_coords.append(f"({math.log1p(fm_median):.6f},{fm_mae:.{decimals}f})")
        if rf_mae is not None and rf_median is not None:
            rf_range_coords.append(f"({math.log1p(rf_median):.6f},{rf_mae:.{decimals}f})")

    if title is None:
        auto_title = (
            f"Regression MAE bucket-wise comparison for the {latex_escape(log_name)} event log. "
            "Left: fixed bucket IDs. Right: median confidence positions with log(1+X)."
        )
    else:
        auto_title = title

    lines: list[str] = []
    lines.append(r"\begin{figure}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{minipage}[t]{0.49\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(",\n".join(
        [
            r"width=\textwidth",
            r"height=0.65\textwidth",
            r"title={Fixed Buckets}",
            r"xlabel={Bucket}",
            r"ylabel={MAE}",
            r"grid=major",
            r"xtick={1,2,3,4,5}",
            r"legend style={at={(0.5,-0.25)}, anchor=north}",
            r"legend columns=2",
        ]
    ))
    lines.append(r"]")
    if fm_fixed_coords:
        lines.append(
            rf"\addplot+[smooth, line width=2.6pt, color=red, mark=*] coordinates {{ {' '.join(fm_fixed_coords)} }};"
        )
        lines.append(r"\addlegendentry{fm}")
    if rf_fixed_coords:
        lines.append(
            rf"\addplot+[smooth, thick, color=blue, mark=*] coordinates {{ {' '.join(rf_fixed_coords)} }};"
        )
        lines.append(r"\addlegendentry{classic\_rf}")
    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{minipage}")
    lines.append(r"\hfill")
    lines.append(r"\begin{minipage}[t]{0.49\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(",\n".join(
        [
            r"width=\textwidth",
            r"height=0.65\textwidth",
            r"title={Confidence-Range Medians}",
            r"xlabel={log(1 + median confidence)}",
            r"ylabel={MAE}",
            r"grid=major",
        ]
    ))
    lines.append(r"]")
    if fm_range_coords:
        lines.append(
            rf"\addplot+[smooth, line width=2.6pt, color=red, mark=*] coordinates {{ {' '.join(fm_range_coords)} }};"
        )
    if rf_range_coords:
        lines.append(
            rf"\addplot+[smooth, thick, color=blue, mark=*] coordinates {{ {' '.join(rf_range_coords)} }};"
        )
    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{minipage}")
    lines.append(rf"\caption{{{auto_title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def collect_regression_bucket_coordinates(
    bucket_rows: list[dict[str, str]],
    decimals: int = 2,
) -> tuple[list[str], list[str], list[str], list[str]]:
    fm_fixed_coords: list[str] = []
    rf_fixed_coords: list[str] = []
    fm_range_coords: list[str] = []
    rf_range_coords: list[str] = []

    for row in bucket_rows:
        bucket_value = parse_numeric(row.get("bucket"))
        if bucket_value is None:
            continue

        fm_mae = parse_numeric(row.get("fm_mae"))
        rf_mae = parse_numeric(row.get("classic_rf_mae"))
        if fm_mae is not None:
            fm_fixed_coords.append(f"({bucket_value:.0f},{fm_mae:.{decimals}f})")
        if rf_mae is not None:
            rf_fixed_coords.append(f"({bucket_value:.0f},{rf_mae:.{decimals}f})")

        fm_median = parse_confidence_range_median(row.get("fm_regression_bucket_range"))
        rf_median = parse_confidence_range_median(row.get("classic_rf_regression_bucket_range"))
        if fm_mae is not None and fm_median is not None:
            fm_range_coords.append(f"({math.log1p(fm_median):.6f},{fm_mae:.{decimals}f})")
        if rf_mae is not None and rf_median is not None:
            rf_range_coords.append(f"({math.log1p(rf_median):.6f},{rf_mae:.{decimals}f})")

    return fm_fixed_coords, rf_fixed_coords, fm_range_coords, rf_range_coords


def render_regression_mae_bucket_compact_figure(
    bucket_rows_by_log: dict[str, list[dict[str, str]]],
    selected_logs: list[str],
    label: str,
    decimals: int = 2,
) -> str | None:
    available_logs = [log_name for log_name in selected_logs if log_name in bucket_rows_by_log]
    if not available_logs:
        available_logs = sorted(bucket_rows_by_log.keys())
    if not available_logs:
        return None

    lines: list[str] = []
    lines.append(r"\begin{figure}[ht]")
    lines.append(r"\centering")

    lines.append(r"\begin{minipage}[t]{\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\textbf{Fixed Buckets}\\[2pt]")
    for idx, log_name in enumerate(available_logs):
        fm_fixed, rf_fixed, _, _ = collect_regression_bucket_coordinates(
            bucket_rows=bucket_rows_by_log[log_name],
            decimals=decimals,
        )
        lines.append(r"\begin{minipage}[t]{0.30\textwidth}")
        lines.append(r"\centering")
        axis_options = [
            r"width=\textwidth",
            r"height=0.62\textwidth",
            rf"title={{{latex_escape(log_name)}}}",
            r"xlabel={Bucket}",
            r"grid=major",
            r"scale only axis",
            r"tick label style={font=\tiny}",
            r"label style={font=\scriptsize}",
            r"title style={font=\scriptsize}",
            r"legend style={at={(0.5,-0.30)}, anchor=north, font=\tiny, draw=none, fill=white}",
            r"legend columns=1",
        ]
        if idx == 0:
            axis_options.append(r"ylabel={MAE}")

        lines.append(r"\begin{tikzpicture}")
        lines.append(r"\begin{axis}[")
        lines.append(",\n".join(axis_options))
        lines.append(r"]")
        if fm_fixed:
            lines.append(
                rf"\addplot+[smooth, line width=2.6pt, color=red, mark=*] coordinates {{ {' '.join(fm_fixed)} }};"
            )
            lines.append(r"\addlegendentry{fm}")
        if rf_fixed:
            lines.append(
                rf"\addplot+[smooth, thick, color=blue, mark=*] coordinates {{ {' '.join(rf_fixed)} }};"
            )
            lines.append(r"\addlegendentry{classic\_rf}")
        lines.append(r"\end{axis}")
        lines.append(r"\end{tikzpicture}")
        lines.append(r"\end{minipage}")
        if idx < len(available_logs) - 1:
            lines.append(r"\hspace{0.02\textwidth}")
    lines.append(r"\end{minipage}")

    lines.append(r"\vspace{16pt}")
    lines.append(r"\begin{minipage}[t]{\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\textbf{Confidence-Range Medians}\\[2pt]")
    for idx, log_name in enumerate(available_logs):
        _, _, fm_range, rf_range = collect_regression_bucket_coordinates(
            bucket_rows=bucket_rows_by_log[log_name],
            decimals=decimals,
        )
        lines.append(r"\begin{minipage}[t]{0.30\textwidth}")
        lines.append(r"\centering")
        axis_options = [
            r"width=\textwidth",
            r"height=0.62\textwidth",
            rf"title={{{latex_escape(log_name)}}}",
            r"xlabel={log(1+median conf.)}",
            r"grid=major",
            r"scale only axis",
            r"tick label style={font=\tiny}",
            r"label style={font=\scriptsize}",
            r"title style={font=\scriptsize}",
            r"legend style={at={(0.5,-0.30)}, anchor=north, font=\tiny, draw=none, fill=white}",
            r"legend columns=1",
        ]
        if idx == 0:
            axis_options.append(r"ylabel={MAE}")

        lines.append(r"\begin{tikzpicture}")
        lines.append(r"\begin{axis}[")
        lines.append(",\n".join(axis_options))
        lines.append(r"]")
        if fm_range:
            lines.append(
                rf"\addplot+[smooth, line width=2.6pt, color=red, mark=*] coordinates {{ {' '.join(fm_range)} }};"
            )
            lines.append(r"\addlegendentry{fm}")
        if rf_range:
            lines.append(
                rf"\addplot+[smooth, thick, color=blue, mark=*] coordinates {{ {' '.join(rf_range)} }};"
            )
            lines.append(r"\addlegendentry{classic\_rf}")
        lines.append(r"\end{axis}")
        lines.append(r"\end{tikzpicture}")
        lines.append(r"\end{minipage}")
        if idx < len(available_logs) - 1:
            lines.append(r"\hspace{0.02\textwidth}")
    lines.append(r"\end{minipage}")

    lines.append(
        r"\caption{Compact regression MAE confidence-bucket comparison for receipt, roadtraffic, and sepsis. Left minipage: fixed buckets. Right minipage: confidence-range medians.}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def render_regression_mae_gap_tikz_plot(
    label: str,
    percentages: list[str],
    selected_methods: list[str],
    averages: dict[str, dict[str, float | None]],
    decimals: int = 2,
    title: str | None = None,
) -> str:
    if title is None:
        method_list = join_for_caption(selected_methods)
        auto_title = (
            "Average percentage gap to the best MAE model by data fraction "
            f"(methods: {method_list}; lower is better)."
        )
    else:
        auto_title = title

    return render_average_percentage_gap_tikz_plot(
        label=label,
        percentages=percentages,
        selected_methods=selected_methods,
        averages=averages,
        decimals=decimals,
        title=auto_title,
    )


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
    col_spec = build_ml_table_col_spec(
        prefix_columns=["l", "l"],
        model_count=len(methods),
        gap_count=len(gap_methods),
    )
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{title} {color_legend}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{0.75\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    header_cells = [r"\textbf{Event Log}", r"\textbf{\%}"]
    header_cells.extend([rf"\textbf{{{format_method_header_label(method)}}}" for method in methods])
    for method in gap_methods:
        gap_label = GAP_METHOD_LABELS.get(method, method)
        header_cells.append(rf"\textbf{{{format_method_header_label(gap_label)} gap}}")
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


def render_accuracy_tikz_plot(
    label: str,
    log_name: str,
    methods: list[str],
    percentages: list[str],
    data: dict[str, dict[str, dict[str, dict[str, Any]]]],
    decimals: int = 2,
    title: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{figure}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")

    tick_label_by_position: dict[float, str] = {}
    for percentage in percentages:
        fraction = percentage_code_to_fraction(percentage)
        if fraction is not None:
            tick_label_by_position[math.log1p(fraction)] = f"{fraction:g}"

    unique_xtick_positions = sorted(tick_label_by_position.keys())
    xtick_text = ",".join(f"{tick:.6f}" for tick in unique_xtick_positions)
    xtick_labels_text = ",".join(tick_label_by_position[tick] for tick in unique_xtick_positions)

    method_to_coordinates: dict[str, list[str]] = {}
    min_accuracy: float | None = None
    for method in methods:
        coordinates: list[str] = []
        for percentage in percentages:
            fraction = percentage_code_to_fraction(percentage)
            if fraction is None:
                continue
            payload = data.get(method, {}).get(log_name, {}).get(percentage)
            accuracy = get_numeric_metric(payload, ["accuracy"])
            if accuracy is None:
                continue
            x_value = math.log1p(fraction)
            y_value = accuracy * 100.0
            min_accuracy = y_value if min_accuracy is None else min(min_accuracy, y_value)
            coordinates.append(f"({x_value:.6f},{y_value:.{decimals}f})")
        method_to_coordinates[method] = coordinates

    axis_options = [
        r"width=0.8\textwidth",
        r"height=0.45\textwidth",
        r"xlabel={log(1 + Data Fraction (\%))}",
        r"ylabel={Accuracy (\%)}",
        r"grid=major",
        r"legend style={at={(0.5,-0.22)}, anchor=north}",
        r"legend columns=2",
        f"ymin={(min_accuracy if min_accuracy is not None else 0.0):.{decimals}f}",
        r"ymax=100",
    ]
    if xtick_text:
        axis_options.append(rf"xtick={{{xtick_text}}}")
    if xtick_labels_text:
        axis_options.append(rf"xticklabels={{{xtick_labels_text}}}")

    lines.append(r"\begin{axis}[")
    lines.append(",\n".join(axis_options))
    lines.append(r"]")

    for method in methods:
        coordinates = method_to_coordinates.get(method, [])
        if not coordinates:
            continue

        color = PLOT_METHOD_COLORS.get(method, "black")
        line_thickness = method_line_thickness(method)
        lines.append(
            rf"\addplot+[smooth, {line_thickness}, color={color}, mark=*] coordinates {{ {' '.join(coordinates)} }};"
        )
        lines.append(rf"\addlegendentry{{{latex_escape(method)}}}")

    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    if title is None:
        method_list = join_for_caption(methods)
        auto_title = (
            f"Classification accuracy percentages by data fraction for the {latex_escape(log_name)} event log "
            f"(methods: {method_list})."
        )
    else:
        auto_title = title
    lines.append(rf"\caption{{{auto_title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def render_regression_mae_tikz_plot(
    label: str,
    log_name: str,
    methods: list[str],
    percentages: list[str],
    data: dict[str, dict[str, dict[str, dict[str, Any]]]],
    decimals: int = 2,
    title: str | None = None,
    metric_keys: list[str] | None = None,
    transform: Callable[[float], float] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{figure}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")

    tick_label_by_position: dict[float, str] = {}
    for percentage in percentages:
        fraction = percentage_code_to_fraction(percentage)
        if fraction is not None:
            tick_label_by_position[math.log1p(fraction)] = f"{fraction:g}"

    unique_xtick_positions = sorted(tick_label_by_position.keys())
    xtick_text = ",".join(f"{tick:.6f}" for tick in unique_xtick_positions)
    xtick_labels_text = ",".join(tick_label_by_position[tick] for tick in unique_xtick_positions)

    candidate_metric_keys = metric_keys if metric_keys is not None else ["mae", "mae_seconds"]
    value_transform = transform if transform is not None else (lambda value: value / 3600.0)

    method_to_coordinates: dict[str, list[str]] = {}
    min_mae: float | None = None
    for method in methods:
        coordinates: list[str] = []
        for percentage in percentages:
            fraction = percentage_code_to_fraction(percentage)
            if fraction is None:
                continue
            payload = data.get(method, {}).get(log_name, {}).get(percentage)
            mae_value = get_numeric_metric(payload, candidate_metric_keys)
            if mae_value is None:
                continue
            mae_value = value_transform(mae_value)
            x_value = math.log1p(fraction)
            min_mae = mae_value if min_mae is None else min(min_mae, mae_value)
            coordinates.append(f"({x_value:.6f},{mae_value:.{decimals}f})")
        method_to_coordinates[method] = coordinates

    axis_options = [
        r"width=0.8\textwidth",
        r"height=0.45\textwidth",
        r"xlabel={log(1 + Data Fraction (\%))}",
        r"ylabel={MAE (hours)}",
        r"grid=major",
        r"legend style={at={(0.5,-0.22)}, anchor=north}",
        r"legend columns=2",
        f"ymin={(min_mae if min_mae is not None else 0.0):.{decimals}f}",
    ]
    if xtick_text:
        axis_options.append(rf"xtick={{{xtick_text}}}")
    if xtick_labels_text:
        axis_options.append(rf"xticklabels={{{xtick_labels_text}}}")

    lines.append(r"\begin{axis}[")
    lines.append(",\n".join(axis_options))
    lines.append(r"]")

    for method in methods:
        coordinates = method_to_coordinates.get(method, [])
        if not coordinates:
            continue

        color = PLOT_METHOD_COLORS.get(method, "black")
        line_thickness = method_line_thickness(method)
        lines.append(
            rf"\addplot+[smooth, {line_thickness}, color={color}, mark=*] coordinates {{ {' '.join(coordinates)} }};"
        )
        lines.append(rf"\addlegendentry{{{latex_escape(method)}}}")

    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    if title is None:
        method_list = join_for_caption(methods)
        auto_title = (
            f"Regression MAE by data fraction for the {latex_escape(log_name)} event log "
            f"(methods: {method_list})."
        )
    else:
        auto_title = title
    lines.append(rf"\caption{{{auto_title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def extract_resizebox_block(latex_content: str) -> str:
    lines = latex_content.splitlines()
    start_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.startswith(r"\resizebox{"):
            start_idx = idx
            break
    if start_idx is None:
        return latex_content

    end_idx: int | None = None
    for idx in range(start_idx + 1, len(lines)):
        if lines[idx] == r"}":
            end_idx = idx
            break
    if end_idx is None:
        return "\n".join(lines[start_idx:])
    return "\n".join(lines[start_idx : end_idx + 1])


def extract_tikzpicture_block(latex_content: str) -> str:
    lines = latex_content.splitlines()
    start_idx: int | None = None
    for idx, line in enumerate(lines):
        if line == r"\begin{tikzpicture}":
            start_idx = idx
            break
    if start_idx is None:
        return latex_content

    end_idx: int | None = None
    for idx in range(start_idx, len(lines)):
        if lines[idx] == r"\end{tikzpicture}":
            end_idx = idx
            break
    if end_idx is None:
        return "\n".join(lines[start_idx:])
    return "\n".join(lines[start_idx : end_idx + 1])


def set_resizebox_width(resizebox_block: str, width_spec: str = r"\textwidth") -> str:
    lines = resizebox_block.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith(r"\resizebox{"):
            lines[idx] = rf"\resizebox{{{width_spec}}}{{!}}{{%"
            break
    return "\n".join(lines)


def adapt_gap_tikz_for_small_minipage(tikz_block: str) -> str:
    adjusted_lines: list[str] = []
    for line in tikz_block.splitlines():
        stripped = line.strip()

        if stripped.startswith("width="):
            adjusted_lines.append(r"width=0.92\textwidth,")
            continue
        if stripped.startswith("height="):
            adjusted_lines.append(r"height=1.00\textwidth,")
            continue
        if stripped.startswith("legend style="):
            adjusted_lines.append(
                r"legend style={at={(0.5,-0.30)}, anchor=north, font=\tiny, /tikz/every even column/.append style={column sep=4pt}},"
            )
            continue
        if stripped.startswith("legend columns="):
            adjusted_lines.append(r"legend columns=2,")
            continue
        if stripped == "grid=major,":
            adjusted_lines.append(line)
            adjusted_lines.append(r"scale only axis,")
            adjusted_lines.append(r"tick label style={font=\tiny},")
            adjusted_lines.append(r"label style={font=\scriptsize},")
            continue

        adjusted_lines.append(line)

    return "\n".join(adjusted_lines)


def render_classification_minipage_row(
    classification_table_latex: str,
    classification_gap_plot_latex: str,
) -> str:
    classification_table_block = set_resizebox_width(extract_resizebox_block(classification_table_latex))
    classification_gap_plot_block = adapt_gap_tikz_for_small_minipage(
        extract_tikzpicture_block(classification_gap_plot_latex)
    )

    lines: list[str] = []
    lines.append(r"\begin{figure}[ht]")
    lines.append(r"\centering")
    lines.append(r"\makebox[\textwidth][c]{%")
    lines.append(r"\begin{minipage}[t]{0.6\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\textbf{Overall Classification Accuracy}\\[2pt]")
    lines.append(classification_table_block)
    lines.append(r"\end{minipage}%")
    lines.append(r"\hspace{0.01\textwidth}%")
    lines.append(r"\begin{minipage}[t]{0.38\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\textbf{Classification Gap Graph}\\[2pt]")
    lines.append(classification_gap_plot_block)
    lines.append(r"\end{minipage}%")
    lines.append(r"}")
    lines.append(
        r"\caption{Side-by-side classification overview: overall accuracy table and classification gap graph.}"
    )
    lines.append(r"\label{fig:classification-overview-minipage}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def render_mae_minipage_row(
    mae_table_latex: str,
    mae_gap_plot_latex: str,
) -> str:
    mae_table_block = set_resizebox_width(extract_resizebox_block(mae_table_latex))
    mae_gap_plot_block = adapt_gap_tikz_for_small_minipage(
        extract_tikzpicture_block(mae_gap_plot_latex)
    )

    lines: list[str] = []
    lines.append(r"\begin{figure}[ht]")
    lines.append(r"\centering")
    lines.append(r"\makebox[\textwidth][c]{%")
    lines.append(r"\begin{minipage}[t]{0.6\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\textbf{Overall Regression MAE}\\[2pt]")
    lines.append(mae_table_block)
    lines.append(r"\end{minipage}%")
    lines.append(r"\hspace{0.01\textwidth}%")
    lines.append(r"\begin{minipage}[t]{0.38\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\textbf{MAE Gap Graph}\\[2pt]")
    lines.append(mae_gap_plot_block)
    lines.append(r"\end{minipage}%")
    lines.append(r"}")
    lines.append(
        r"\caption{Side-by-side regression MAE overview: overall MAE table and MAE gap graph.}"
    )
    lines.append(r"\label{fig:mae-overview-minipage}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def render_dual_tikz_minipage_figure(
    left_figure_latex: str,
    right_figure_latex: str,
    left_title: str,
    right_title: str,
    caption: str,
    label: str,
) -> str:
    def adapt_tikz_for_dual_minipage(tikz_block: str) -> str:
        adjusted_lines: list[str] = []
        for line in tikz_block.splitlines():
            stripped = line.strip()
            if stripped.startswith("width="):
                adjusted_lines.append(r"width=0.92\textwidth,")
                continue
            if stripped.startswith("height="):
                adjusted_lines.append(r"height=0.60\textwidth,")
                continue
            if stripped.startswith("legend style="):
                adjusted_lines.append(
                    r"legend style={at={(0.5,-0.52)}, anchor=north, font=\tiny, /tikz/every even column/.append style={column sep=4pt}},"
                )
                continue
            adjusted_lines.append(line)
        return "\n".join(adjusted_lines)

    left_tikz_block = adapt_tikz_for_dual_minipage(extract_tikzpicture_block(left_figure_latex))
    right_tikz_block = adapt_tikz_for_dual_minipage(extract_tikzpicture_block(right_figure_latex))

    lines: list[str] = []
    lines.append(r"\begin{figure}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{minipage}[t]{0.49\textwidth}")
    lines.append(r"\centering")
    lines.append(rf"\textbf{{{latex_escape(left_title)}}}\\[2pt]")
    lines.append(left_tikz_block)
    lines.append(r"\end{minipage}")
    lines.append(r"\hfill")
    lines.append(r"\begin{minipage}[t]{0.49\textwidth}")
    lines.append(r"\centering")
    lines.append(rf"\textbf{{{latex_escape(right_title)}}}\\[2pt]")
    lines.append(right_tikz_block)
    lines.append(r"\end{minipage}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    output_path = repo_root / "preliminary_output.tex"
    buckets_path = repo_root / "other_data" / "buckets.txt"
    act_time_corr_path = repo_root / "other_data" / "act_time_corr_metrics.txt"
    intra_expert_metrics_path = repo_root / "other_data" / "intra_expert_similarity_metrics.csv"
    inter_expert_metrics_path = repo_root / "other_data" / "inter_expert_metrics.csv"

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

    requested_percentages = ensure_required_percentages(args.percentages, required_codes=["0100"])
    table_percentages = [code for code in requested_percentages if code in TABLE_PERCENTAGE_CODES]
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
        percentages=table_percentages,
        selected_methods=TARGET_GAP_METHODS,
        averages=classification_gap_averages,
    )
    figure_classification_gap = render_average_percentage_gap_tikz_plot(
        label="fig:classification-average-gap-to-best-curves",
        percentages=gap_percentages,
        selected_methods=["knn", "tabpfn", "our_fm", "our_fm_knn"],
        averages=classification_gap_averages,
    )
    figure_billing_accuracy = render_accuracy_tikz_plot(
        label="fig:billing-classification-accuracy-curves",
        log_name="billing",
        methods=["knn", "tabpfn", "our_fm", "our_fm_knn"],
        percentages=TARGET_PERCENTAGE_CODES,
        data=cls_data,
    )
    figure_helpdesk_accuracy = render_accuracy_tikz_plot(
        label="fig:helpdesk-classification-accuracy-curves",
        log_name="helpdesk",
        methods=["knn", "tabpfn", "our_fm", "our_fm_knn"],
        percentages=TARGET_PERCENTAGE_CODES,
        data=cls_data,
    )
    figure_classification_overview_row = render_classification_minipage_row(
        classification_table_latex=table_accuracy,
        classification_gap_plot_latex=figure_classification_gap,
    )
    figure_classification_accuracy_pair = render_dual_tikz_minipage_figure(
        left_figure_latex=figure_billing_accuracy,
        right_figure_latex=figure_helpdesk_accuracy,
        left_title="billing",
        right_title="helpdesk",
        caption=(
            "Classification accuracy percentages by data fraction shown side by side "
            "for billing and helpdesk."
        ),
        label="fig:classification-accuracy-curves-billing-helpdesk-side-by-side",
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
        percentages=table_percentages,
        selected_methods=TARGET_GAP_METHODS,
        averages=mae_gap_averages,
    )
    figure_mae_gap = render_regression_mae_gap_tikz_plot(
        label="fig:mae-average-gap-to-best-curves",
        percentages=gap_percentages,
        selected_methods=["knn", "tabpfn", "our_fm", "our_fm_knn"],
        averages=mae_gap_averages,
    )
    figure_mae_overview_row = render_mae_minipage_row(
        mae_table_latex=table_mae,
        mae_gap_plot_latex=figure_mae_gap,
    )
    figure_billing_mae = render_regression_mae_tikz_plot(
        label="fig:billing-regression-mae-curves",
        log_name="billing",
        methods=["knn", "tabpfn", "our_fm", "our_fm_knn"],
        percentages=TARGET_PERCENTAGE_CODES,
        data=reg_data,
    )
    figure_sepsis_mae = render_regression_mae_tikz_plot(
        label="fig:sepsis-regression-mae-curves",
        log_name="sepsis",
        methods=["knn", "tabpfn", "our_fm", "our_fm_knn"],
        percentages=TARGET_PERCENTAGE_CODES,
        data=reg_data,
    )
    figure_regression_mae_pair = render_dual_tikz_minipage_figure(
        left_figure_latex=figure_billing_mae,
        right_figure_latex=figure_sepsis_mae,
        left_title="billing",
        right_title="sepsis",
        caption=(
            "Regression MAE by data fraction shown side by side "
            "for billing and sepsis."
        ),
        label="fig:regression-mae-curves-billing-sepsis-side-by-side",
    )

    act_time_metrics_by_log = load_act_time_corr_metrics(act_time_corr_path)
    metric_logs = sorted(act_time_metrics_by_log.keys())
    classification_corr_table: str | None = None
    mae_corr_table: str | None = None
    if metric_logs:
        classification_gap_by_log = compute_average_percentage_gap_by_log(
            methods=classification_methods,
            logs=metric_logs,
            percentages=gap_percentages,
            data=cls_data,
            metric_keys=["accuracy"],
            selected_methods=TARGET_GAP_METHODS,
            higher_is_better=True,
        )
        mae_gap_by_log = compute_average_percentage_gap_by_log(
            methods=regression_methods,
            logs=metric_logs,
            percentages=gap_percentages,
            data=reg_data,
            metric_keys=["mae", "mae_seconds"],
            selected_methods=TARGET_GAP_METHODS,
            transform=lambda value: value / 3600.0,
            higher_is_better=False,
        )

        classification_metric_correlations = compute_metric_gap_correlations(
            metrics_by_log=act_time_metrics_by_log,
            gap_by_log=classification_gap_by_log,
            metric_columns=ACT_TIME_CORR_COLUMNS,
        )
        mae_metric_correlations = compute_metric_gap_correlations(
            metrics_by_log=act_time_metrics_by_log,
            gap_by_log=mae_gap_by_log,
            metric_columns=ACT_TIME_CORR_COLUMNS,
        )

        classification_corr_table = render_metric_gap_correlation_table(
            title=(
                "Pearson correlation between activity-time metrics and event-log average "
                "classification gap to best (lower is better)."
            ),
            label="tab:activity-time-vs-classification-gap-correlation",
            correlations=classification_metric_correlations,
            metric_columns=ACT_TIME_CORR_COLUMNS,
        )
        mae_corr_table = render_metric_gap_correlation_table(
            title=(
                "Pearson correlation between activity-time metrics and event-log average "
                "MAE gap to best (lower is better)."
            ),
            label="tab:activity-time-vs-mae-gap-correlation",
            correlations=mae_metric_correlations,
            metric_columns=ACT_TIME_CORR_COLUMNS,
        )

    intra_expert_table_rows = load_comma_table(intra_expert_metrics_path)
    intra_expert_table_latex: str | None = None
    if intra_expert_table_rows:
        intra_expert_columns = [
            "log",
            "expert",
            "intra_centroid_cos",
            "inter_centroid_cos",
            "centroid_margin",
            "knn_purity@5",
            "knn_purity@10",
        ]
        intra_expert_labels = {
            "log": "Event Log",
            "expert": "Expert",
            "intra_centroid_cos": "Intra Centroid Cos",
            "inter_centroid_cos": "Inter Centroid Cos",
            "centroid_margin": "Centroid Margin",
            "knn_purity@5": "kNN Purity@5",
            "knn_purity@10": "kNN Purity@10",
        }
        intra_expert_table_latex = render_csv_table(
            title="Intra-expert similarity metrics.",
            label="tab:intra-expert-similarity-metrics",
            rows=intra_expert_table_rows,
            column_order=intra_expert_columns,
            column_labels=intra_expert_labels,
        )

    inter_expert_table_rows = load_comma_table(inter_expert_metrics_path)
    inter_expert_table_latex: str | None = None
    if inter_expert_table_rows:
        inter_expert_columns = [
            "log",
            "expert_a",
            "expert_b",
            "classification_mean_cos",
            "classification_mean_l2",
            "classification_centroid_cos_mean",
            "regression_mean_cos",
            "regression_mean_l2",
        ]
        inter_expert_labels = {
            "log": "Event Log",
            "expert_a": "Expert A",
            "expert_b": "Expert B",
            "classification_mean_cos": "Cls Mean Cos",
            "classification_mean_l2": "Cls Mean L2",
            "classification_centroid_cos_mean": "Cls Centroid Cos Mean",
            "regression_mean_cos": "Reg Mean Cos",
            "regression_mean_l2": "Reg Mean L2",
        }
        inter_expert_table_latex = render_csv_table(
            title="Inter-expert metrics.",
            label="tab:inter-expert-metrics",
            rows=inter_expert_table_rows,
            column_order=inter_expert_columns,
            column_labels=inter_expert_labels,
        )

    bucket_rows_by_log = load_bucket_rows(buckets_path)
    classification_bucket_compact_figure: str | None = None
    regression_mae_bucket_compact_figure: str | None = None
    if bucket_rows_by_log:
        classification_bucket_compact_figure = render_classification_bucket_compact_figure(
            bucket_rows_by_log=bucket_rows_by_log,
            selected_logs=TARGET_CLASSIFICATION_BUCKET_LOGS,
            label="fig:classification-buckets-compact-comparison",
        )
        regression_mae_bucket_compact_figure = render_regression_mae_bucket_compact_figure(
            bucket_rows_by_log=bucket_rows_by_log,
            selected_logs=TARGET_CLASSIFICATION_BUCKET_LOGS,
            label="fig:regression-mae-buckets-compact-comparison",
        )

    header = "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage{graphicx} % Required for inserting images",
            r"\usepackage{array}",
            r"\usepackage{xcolor}",
            r"\usepackage{multirow}",
            r"\usepackage{tikz}",
            r"\usepackage{pgfplots}",
            r"\pgfplotsset{compat=1.18}",
            r"\renewcommand{\arraystretch}{0.92}",
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
        figure_classification_overview_row
        + "\n\n"
        + figure_classification_accuracy_pair,
        figure_mae_overview_row
        + "\n\n"
        + figure_regression_mae_pair,
        table_r2,
    ]
    if classification_bucket_compact_figure:
        sections.append(
            r"\section{Classification Confidence Buckets}"
            + "\n\n"
            + classification_bucket_compact_figure
        )
    if regression_mae_bucket_compact_figure:
        sections.append(
            r"\section{Regression MAE Confidence Buckets}"
            + "\n\n"
            + regression_mae_bucket_compact_figure
        )
    if classification_corr_table and mae_corr_table:
        sections.append(
            r"\section{Activity-Time Correlation Analysis}"
            + "\n\n"
            + classification_corr_table
            + "\n\n"
            + mae_corr_table
        )
    if intra_expert_table_latex and inter_expert_table_latex:
        sections.append(
            r"\section{Expert Similarity Metrics}"
            + "\n\n"
            + intra_expert_table_latex
            + "\n\n"
            + inter_expert_table_latex
        )
    body = "\n\n\\clearpage\\newpage\n\n".join(sections)
    tex_content = header + body + "\n\n\\end{document}\n"
    output_path.write_text(tex_content, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
