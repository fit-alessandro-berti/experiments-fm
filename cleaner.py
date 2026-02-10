#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROOTS = [
    Path("next_activity_classification/results"),
    Path("remaining_time_prediction/results"),
]


def should_delete(path: Path) -> tuple[bool, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return True, "invalid_json"

    status = payload.get("status")
    if status == "ok":
        return False, "ok"
    return True, f"status={status!r}"


def cleanup(roots: list[Path], dry_run: bool) -> tuple[int, int]:
    scanned = 0
    deleted = 0

    for root in roots:
        if not root.exists():
            continue
        for json_file in sorted(root.rglob("*.json")):
            scanned += 1
            delete_it, reason = should_delete(json_file)
            if not delete_it:
                continue
            deleted += 1
            action = "Would delete" if dry_run else "Deleted"
            print(f"{action}: {json_file} ({reason})")
            if not dry_run:
                json_file.unlink()

    return scanned, deleted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete experimental result files whose JSON status is not 'ok' "
            "from classification and regression result folders."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=DEFAULT_ROOTS,
        help="Result root folders to clean.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show files that would be deleted without deleting them.",
    )
    args = parser.parse_args()

    scanned, deleted = cleanup(args.roots, dry_run=args.dry_run)
    print(f"Scanned: {scanned} JSON files")
    print(f"{'Would delete' if args.dry_run else 'Deleted'}: {deleted} JSON files")


if __name__ == "__main__":
    main()

