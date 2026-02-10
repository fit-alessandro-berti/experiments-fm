#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

from pm4py.util import xes_constants


@dataclass
class PrefixFeatureSpace:
    activities: list[str]
    paths: list[tuple[str, str]]
    activity_to_index: dict[str, int]
    path_to_index: dict[tuple[str, str], int]


def fit_feature_space(log, activity_key: str = xes_constants.DEFAULT_NAME_KEY) -> PrefixFeatureSpace:
    activities = set()
    paths = set()

    for trace in log:
        events = [event for event in trace if activity_key in event]
        for event in events:
            activities.add(event[activity_key])
        for idx in range(1, len(events)):
            paths.add((events[idx - 1][activity_key], events[idx][activity_key]))

    activity_list = sorted(activities)
    path_list = sorted(paths)
    return PrefixFeatureSpace(
        activities=activity_list,
        paths=path_list,
        activity_to_index={activity: idx for idx, activity in enumerate(activity_list)},
        path_to_index={path: idx for idx, path in enumerate(path_list)},
    )


def build_next_activity_dataset(
    log,
    feature_space: PrefixFeatureSpace,
    activity_key: str = xes_constants.DEFAULT_NAME_KEY,
    timestamp_key: str = xes_constants.DEFAULT_TIMESTAMP_KEY,
) -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    targets: list[int] = []

    n_activities = len(feature_space.activities)
    n_paths = len(feature_space.paths)

    for trace in log:
        events = [event for event in trace if activity_key in event]
        if len(events) < 2:
            continue

        start_time = events[0].get(timestamp_key)
        seen_activities = set()
        first_activity = events[0][activity_key]
        if first_activity in feature_space.activity_to_index:
            seen_activities.add(first_activity)
        seen_paths: set[tuple[str, str]] = set()

        for idx in range(1, len(events)):
            prev_event = events[idx - 1]
            curr_event = events[idx]

            if idx >= 2:
                prior_event = events[idx - 2]
                prior_path = (prior_event[activity_key], prev_event[activity_key])
                if prior_path in feature_space.path_to_index:
                    seen_paths.add(prior_path)

            row = [0.0] * (n_activities + n_paths)
            for seen_activity in seen_activities:
                row[feature_space.activity_to_index[seen_activity]] = 1.0
            for seen_path in seen_paths:
                row[n_activities + feature_space.path_to_index[seen_path]] = 1.0

            prev_to_penultimate = 0.0
            if idx >= 2:
                prev_ts = prev_event.get(timestamp_key)
                prior_ts = events[idx - 2].get(timestamp_key)
                if prev_ts is not None and prior_ts is not None:
                    prev_to_penultimate = (prev_ts - prior_ts).total_seconds()

            start_to_penultimate = 0.0
            if start_time is not None:
                penultimate_ts = prev_event.get(timestamp_key)
                if penultimate_ts is not None:
                    start_to_penultimate = (penultimate_ts - start_time).total_seconds()

            path_time_diff = 0.0
            prev_ts = prev_event.get(timestamp_key)
            curr_ts = curr_event.get(timestamp_key)
            if prev_ts is not None and curr_ts is not None:
                path_time_diff = (curr_ts - prev_ts).total_seconds()

            row.extend([prev_to_penultimate, start_to_penultimate, path_time_diff])

            curr_activity = curr_event[activity_key]
            curr_idx = feature_space.activity_to_index.get(curr_activity)
            if curr_idx is not None:
                features.append(row)
                targets.append(curr_idx)

            if curr_activity in feature_space.activity_to_index:
                seen_activities.add(curr_activity)

    return features, targets

