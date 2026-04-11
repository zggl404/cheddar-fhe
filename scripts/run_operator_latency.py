#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    pd = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_OPS = [
    "AddCC",
    "AddCP",
    "MulCC",
    "MulCP",
    "Rescale",
    "Relin",
    "Rotate",
    "Mul+add",
    "Rotate+add",
    "Rotate+mul+add",
]
DEFAULT_REPORT_LEVELS = [3, 6, 10, 15, 20, 30]
WILDCARD_CHARS = set("*?[]")
OPERATOR_LABELS = {
    "addcc": "AddCC",
    "addcp": "AddCP",
    "mulcc": "MulCC",
    "mulcp": "MulCP",
    "rescale": "Rescale",
    "relin": "Relin",
    "rotate": "Rotate",
    "muladd": "Mul+add",
    "multplainadd": "Mul+add",
    "rotateadd": "Rotate+add",
    "roateadd": "Rotate+add",
    "hrotadd": "Rotate+add",
    "rotatemuladd": "Rotate+mul+add",
    "roatemuladd": "Rotate+mul+add",
    "hrotpmadunsafe": "Rotate+mul+add",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Cheddar operator latency benchmarks for one or more JSON "
            "configs and generate CSV tables."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help=(
            "Config file, directory, or glob pattern. "
            "Repeat this flag to pass multiple inputs."
        ),
    )
    parser.add_argument(
        "--ops",
        default=",".join(DEFAULT_OPS),
        help="Comma-separated operator list to benchmark.",
    )
    parser.add_argument(
        "--bench-levels",
        default="all",
        help="Levels to benchmark: 'all', comma list, or ranges like 3-10.",
    )
    parser.add_argument(
        "--report-levels",
        default=",".join(str(level) for level in DEFAULT_REPORT_LEVELS),
        help="Comma-separated levels to include in the compact report tables.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=30, help="Measured iterations.")
    parser.add_argument(
        "--rot-dist",
        type=int,
        default=1234,
        help="Rotation distance used by Rotate-related operators.",
    )
    parser.add_argument(
        "--word-bits",
        default="auto",
        choices=("auto", "32", "64"),
        help="Force the benchmark word size or auto-detect it.",
    )
    parser.add_argument(
        "--build-dir",
        default="build/unittest",
        help="Directory containing the operator_bench executable.",
    )
    parser.add_argument(
        "--executable",
        default="operator_bench",
        help="Benchmark executable name inside --build-dir.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Base directory for outputs. Defaults to results/operator_latency/<timestamp>.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip untimed correctness checks inside the benchmark executable.",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Do not save per-sample raw timing CSVs.",
    )
    return parser.parse_args()


def parse_level_list(spec: str) -> list[int]:
    levels: list[int] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if "-" in token:
            begin_text, end_text = token.split("-", 1)
            begin = int(begin_text)
            end = int(end_text)
            if end < begin:
                raise ValueError(f"Invalid level range: {token}")
            levels.extend(range(begin, end + 1))
        else:
            levels.append(int(token))
    levels = sorted(set(levels))
    if not levels:
        raise ValueError("No report levels selected")
    return levels


def resolve_config_inputs(entries: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    for entry in entries:
        candidate = Path(entry)
        if any(ch in entry for ch in WILDCARD_CHARS):
            matches = sorted(Path().glob(entry))
        elif candidate.is_dir():
            matches = sorted(candidate.glob("*.json"))
        elif candidate.exists():
            matches = [candidate]
        else:
            matches = []

        if not matches:
            raise FileNotFoundError(f"No config files found for input: {entry}")
        for match in matches:
            if match.is_file() and match.suffix == ".json":
                resolved.append(match.resolve())

    unique_configs: list[Path] = []
    seen = set()
    for path in resolved:
        if path not in seen:
            seen.add(path)
            unique_configs.append(path)
    return unique_configs


def default_results_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("results") / "operator_latency" / timestamp


def build_command(
    executable: Path,
    config_path: Path,
    ops: str,
    bench_levels: str,
    warmup: int,
    iters: int,
    rot_dist: int,
    word_bits: str,
    verify: bool,
    summary_out: Path,
    raw_out: Path | None,
) -> list[str]:
    command = [
        str(executable),
        "--config",
        str(config_path),
        "--ops",
        ops,
        "--levels",
        bench_levels,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--rot-dist",
        str(rot_dist),
        "--word-bits",
        word_bits,
        "--summary-out",
        str(summary_out),
    ]
    if raw_out is not None:
        command.extend(["--raw-out", str(raw_out)])
    if not verify:
        command.append("--no-verify")
    return command


def run_benchmark(
    executable: Path,
    config_path: Path,
    args: argparse.Namespace,
    results_dir: Path,
    run_log,
) -> tuple[Path, Path | None]:
    config_dir = results_dir / config_path.stem
    config_dir.mkdir(parents=True, exist_ok=True)

    summary_out = config_dir / "summary.csv"
    raw_out = None if args.skip_raw else config_dir / "raw.csv"

    command = build_command(
        executable=executable,
        config_path=config_path,
        ops=args.ops,
        bench_levels=args.bench_levels,
        warmup=args.warmup,
        iters=args.iters,
        rot_dist=args.rot_dist,
        word_bits=args.word_bits,
        verify=not args.no_verify,
        summary_out=summary_out,
        raw_out=raw_out,
    )

    logging.info("Running benchmark for %s", config_path)
    logging.info("Command: %s", " ".join(command))
    run_log.write(f"$ {' '.join(command)}\n")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        run_log.write(result.stdout)
        if not result.stdout.endswith("\n"):
            run_log.write("\n")
    if result.stderr:
        run_log.write("[stderr]\n")
        run_log.write(result.stderr)
        if not result.stderr.endswith("\n"):
            run_log.write("\n")

    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for {config_path} with exit code {result.returncode}"
        )
    return summary_out, raw_out


def make_level_columns(levels: list[int]) -> list[str]:
    return [f"L={level}" for level in levels]


def operator_order(ops_spec: str) -> list[str]:
    if ops_spec.strip().lower() == "all":
        return DEFAULT_OPS
    ordered_ops: list[str] = []
    for token in (part.strip() for part in ops_spec.split(",")):
        if not token:
            continue
        normalized = "".join(ch for ch in token.lower() if ch.isalnum())
        ordered_ops.append(OPERATOR_LABELS.get(normalized, token))
    return ordered_ops


def write_pivot_table(
    summary_df: pd.DataFrame,
    config_name: str,
    levels: list[int],
    ops: list[str],
    output_path: Path,
) -> pd.DataFrame:
    config_df = summary_df.loc[summary_df["config_name"] == config_name].copy()
    pivot = config_df.pivot_table(
        index="operator",
        columns="level",
        values="mean_us",
        aggfunc="first",
    )
    pivot = pivot.reindex(index=ops)
    pivot = pivot.reindex(columns=levels)
    pivot.columns = make_level_columns(levels)
    pivot.index.name = "Operation"
    pivot.to_csv(output_path)
    return pivot


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (row["config_name"], row["operator"], int(row["level"])),
    )


def sort_raw_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            row["config_name"],
            row["operator"],
            int(row["level"]),
            int(row["sample_index"]),
        ),
    )


def unique_config_names(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    seen = set()
    for row in rows:
        name = row["config_name"]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def available_levels_for_config(rows: list[dict[str, str]], config_name: str) -> list[int]:
    levels = {
        int(row["level"])
        for row in rows
        if row["config_name"] == config_name
    }
    return sorted(levels)


def write_pivot_table_fallback(
    summary_rows: list[dict[str, str]],
    config_name: str,
    levels: list[int],
    ops: list[str],
    output_path: Path,
) -> list[dict[str, str]]:
    config_rows = [row for row in summary_rows if row["config_name"] == config_name]
    value_map: dict[tuple[str, int], str] = {}
    for row in config_rows:
        value_map[(row["operator"], int(row["level"]))] = row["mean_us"]

    fieldnames = ["Operation"] + make_level_columns(levels)
    table_rows: list[dict[str, str]] = []
    for op in ops:
        row = {"Operation": op}
        for level in levels:
            row[f"L={level}"] = value_map.get((op, level), "")
        table_rows.append(row)

    write_csv_rows(output_path, fieldnames, table_rows)
    return table_rows


def format_table_rows(table_rows: list[dict[str, str]], levels: list[int]) -> str:
    headers = ["Operation"] + make_level_columns(levels)
    widths = {header: len(header) for header in headers}

    def display_value(value: str) -> str:
        if value == "":
            return "NA"
        return f"{float(value):.3f}"

    for row in table_rows:
        widths["Operation"] = max(widths["Operation"], len(row["Operation"]))
        for level in levels:
            header = f"L={level}"
            widths[header] = max(widths[header], len(display_value(row[header])))

    def format_line(values: list[str]) -> str:
        return "  ".join(
            value.ljust(widths[header]) for header, value in zip(headers, values)
        )

    lines = [format_line(headers)]
    for row in table_rows:
        values = [row["Operation"]]
        values.extend(display_value(row[f"L={level}"]) for level in levels)
        lines.append(format_line(values))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    configs = resolve_config_inputs(args.config)
    report_levels = parse_level_list(args.report_levels)

    results_dir = Path(args.results_dir) if args.results_dir else default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    executable = (Path(args.build_dir) / args.executable).resolve()
    if not executable.exists():
        logging.error("Benchmark executable not found: %s", executable)
        return 1

    logging.info("Results directory: %s", results_dir)
    logging.info("Resolved %d config(s)", len(configs))

    ops = operator_order(args.ops)
    summary_paths: list[Path] = []
    raw_paths: list[Path] = []

    run_log_path = results_dir / "run.log"
    with run_log_path.open("w", encoding="utf-8") as run_log:
        run_log.write(f"results_dir={results_dir}\n")
        run_log.write(f"configs={[str(path) for path in configs]}\n")
        run_log.write(f"ops={args.ops}\n")
        run_log.write(f"bench_levels={args.bench_levels}\n")
        run_log.write(f"report_levels={report_levels}\n")
        run_log.write(f"warmup={args.warmup} iters={args.iters} rot_dist={args.rot_dist}\n\n")

        for config_path in configs:
            summary_path, raw_path = run_benchmark(
                executable=executable,
                config_path=config_path,
                args=args,
                results_dir=results_dir,
                run_log=run_log,
            )
            summary_paths.append(summary_path)
            if raw_path is not None:
                raw_paths.append(raw_path)

    if pd is not None:
        summary_frames = [pd.read_csv(path) for path in summary_paths]
        summary_df = pd.concat(summary_frames, ignore_index=True)
        summary_df.sort_values(["config_name", "operator", "level"], inplace=True)
        summary_df.to_csv(results_dir / "summary_all_levels.csv", index=False)
        logging.info("Wrote %s", results_dir / "summary_all_levels.csv")

        if raw_paths:
            raw_frames = [pd.read_csv(path) for path in raw_paths]
            raw_df = pd.concat(raw_frames, ignore_index=True)
            raw_df.sort_values(
                ["config_name", "operator", "level", "sample_index"], inplace=True
            )
            raw_df.to_csv(results_dir / "raw_results.csv", index=False)
            logging.info("Wrote %s", results_dir / "raw_results.csv")

        selected_rows = summary_df.loc[summary_df["level"].isin(report_levels)].copy()
        selected_rows.to_csv(results_dir / "summary_selected_levels.csv", index=False)
        logging.info("Wrote %s", results_dir / "summary_selected_levels.csv")

        for config_name in summary_df["config_name"].drop_duplicates().tolist():
            full_levels = sorted(
                summary_df.loc[
                    summary_df["config_name"] == config_name, "level"
                ].unique().tolist()
            )
            config_dir = results_dir / Path(config_name).stem
            full_table_path = config_dir / f"table_full_{Path(config_name).stem}.csv"
            selected_table_path = (
                config_dir / f"table_selected_{Path(config_name).stem}.csv"
            )

            write_pivot_table(summary_df, config_name, full_levels, ops, full_table_path)
            selected_table = write_pivot_table(
                summary_df,
                config_name,
                report_levels,
                ops,
                selected_table_path,
            )

            print("\n" + "=" * 96)
            print(f"Operator latency report for {config_name} (us)")
            print("=" * 96)
            print(selected_table.to_string(float_format=lambda value: f"{value:.3f}"))
            print("=" * 96)
    else:
        logging.warning("pandas is not installed; using CSV/stdlib fallback")
        summary_rows = []
        for path in summary_paths:
            summary_rows.extend(read_csv_rows(path))
        summary_rows = sort_summary_rows(summary_rows)
        if summary_rows:
            write_csv_rows(
                results_dir / "summary_all_levels.csv",
                list(summary_rows[0].keys()),
                summary_rows,
            )
            logging.info("Wrote %s", results_dir / "summary_all_levels.csv")

        if raw_paths:
            raw_rows = []
            for path in raw_paths:
                raw_rows.extend(read_csv_rows(path))
            raw_rows = sort_raw_rows(raw_rows)
            if raw_rows:
                write_csv_rows(
                    results_dir / "raw_results.csv",
                    list(raw_rows[0].keys()),
                    raw_rows,
                )
                logging.info("Wrote %s", results_dir / "raw_results.csv")

        selected_rows = [
            row for row in summary_rows if int(row["level"]) in set(report_levels)
        ]
        if selected_rows:
            write_csv_rows(
                results_dir / "summary_selected_levels.csv",
                list(selected_rows[0].keys()),
                selected_rows,
            )
            logging.info("Wrote %s", results_dir / "summary_selected_levels.csv")

        for config_name in unique_config_names(summary_rows):
            full_levels = available_levels_for_config(summary_rows, config_name)
            config_dir = results_dir / Path(config_name).stem
            full_table_path = config_dir / f"table_full_{Path(config_name).stem}.csv"
            selected_table_path = (
                config_dir / f"table_selected_{Path(config_name).stem}.csv"
            )

            write_pivot_table_fallback(
                summary_rows, config_name, full_levels, ops, full_table_path
            )
            selected_table = write_pivot_table_fallback(
                summary_rows,
                config_name,
                report_levels,
                ops,
                selected_table_path,
            )

            print("\n" + "=" * 96)
            print(f"Operator latency report for {config_name} (us)")
            print("=" * 96)
            print(format_table_rows(selected_table, report_levels))
            print("=" * 96)

    logging.info("All benchmark results saved under %s", results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
