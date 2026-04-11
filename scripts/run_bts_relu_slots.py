#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_CONFIG = "parameters/bootparam_50_11_16.json"
DEFAULT_SLOTS = [32768, 16384, 8192, 4096]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the BTS-With-ReLU slot-count benchmark and export a compact table."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=(
            "Benchmark config JSON. Defaults to the repository's current "
            "rboot_arcsine ReLU config."
        ),
    )
    parser.add_argument(
        "--slots",
        default=",".join(str(slot) for slot in DEFAULT_SLOTS),
        help="Comma-separated slot counts to benchmark.",
    )
    parser.add_argument(
        "--variant",
        default="basic",
        choices=("basic", "min_ks"),
        help="Benchmark either the basic SlimBoot path or the min_ks variant.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=10, help="Measured iterations.")
    parser.add_argument(
        "--word-bits",
        default="auto",
        choices=("auto", "32", "64"),
        help="Force the benchmark word size or auto-detect it.",
    )
    parser.add_argument(
        "--build-dir",
        default="build/unittest",
        help="Directory containing the bts_relu_bench executable.",
    )
    parser.add_argument(
        "--executable",
        default="bts_relu_bench",
        help="Benchmark executable name inside --build-dir.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Base directory for outputs. Defaults to results/bts_relu_slots/<timestamp>.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the benchmark's untimed decode-and-check pass.",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Do not save per-sample raw timing CSVs.",
    )
    return parser.parse_args()


def parse_slot_list(spec: str) -> list[int]:
    slots: list[int] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        slots.append(int(token))
    if not slots:
        raise ValueError("No slot counts selected")
    return slots


def default_results_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("results") / "bts_relu_slots" / timestamp


def build_command(
    executable: Path,
    config_path: Path,
    slots_spec: str,
    variant: str,
    warmup: int,
    iters: int,
    word_bits: str,
    verify: bool,
    summary_out: Path,
    raw_out: Path | None,
) -> list[str]:
    command = [
        str(executable),
        "--config",
        str(config_path),
        "--slots",
        slots_spec,
        "--variant",
        variant,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pretty_variant_label(variant: str) -> str:
    if variant == "min_ks":
        return "BTS With ReLU (MinKS)"
    return "BTS With ReLU"


def build_compact_rows(summary_rows: list[dict[str, str]], slot_order: list[int]) -> list[dict[str, str]]:
    by_slot = {int(row["slot_count"]): row for row in summary_rows}
    compact_rows: list[dict[str, str]] = []
    for slot in slot_order:
        row = {"Slot num": str(slot), "Latency (us)": ""}
        if slot in by_slot:
            row["Latency (us)"] = by_slot[slot]["mean_us"]
        compact_rows.append(row)
    return compact_rows


def format_table(rows: list[dict[str, str]], latency_header: str) -> str:
    headers = ["Slot num", latency_header]
    widths = {header: len(header) for header in headers}

    def display_latency(value: str) -> str:
        if value == "":
            return "NA"
        return f"{float(value):.3f}"

    for row in rows:
        widths["Slot num"] = max(widths["Slot num"], len(row["Slot num"]))
        widths[latency_header] = max(
            widths[latency_header], len(display_latency(row["Latency (us)"]))
        )

    def format_line(values: list[str]) -> str:
        return "  ".join(
            value.ljust(widths[header]) for header, value in zip(headers, values)
        )

    lines = [format_line(headers)]
    for row in rows:
        lines.append(
            format_line([row["Slot num"], display_latency(row["Latency (us)"])])
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        logging.error("Config file not found: %s", config_path)
        return 1

    slot_order = parse_slot_list(args.slots)
    results_dir = Path(args.results_dir) if args.results_dir else default_results_dir()
    config_dir = results_dir / config_path.stem
    config_dir.mkdir(parents=True, exist_ok=True)

    executable = (Path(args.build_dir) / args.executable).resolve()
    if not executable.exists():
        logging.error("Benchmark executable not found: %s", executable)
        return 1

    summary_out = config_dir / "summary.csv"
    raw_out = None if args.skip_raw else config_dir / "raw.csv"
    compact_out = config_dir / "slot_table.csv"
    run_log_path = results_dir / "run.log"

    command = build_command(
        executable=executable,
        config_path=config_path,
        slots_spec=args.slots,
        variant=args.variant,
        warmup=args.warmup,
        iters=args.iters,
        word_bits=args.word_bits,
        verify=not args.no_verify,
        summary_out=summary_out,
        raw_out=raw_out,
    )

    logging.info("Results directory: %s", results_dir)
    logging.info("Command: %s", " ".join(command))

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    with run_log_path.open("w", encoding="utf-8") as run_log:
        run_log.write(f"results_dir={results_dir}\n")
        run_log.write(f"config={config_path}\n")
        run_log.write(f"slots={args.slots}\n")
        run_log.write(
            f"variant={args.variant} warmup={args.warmup} iters={args.iters}\n\n"
        )
        run_log.write(f"$ {' '.join(command)}\n")
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
        logging.error("Benchmark failed. See %s for details.", run_log_path)
        return result.returncode

    summary_rows = read_csv_rows(summary_out)
    summary_rows.sort(key=lambda row: int(row["slot_count"]), reverse=True)

    latency_header = pretty_variant_label(args.variant)
    compact_rows = build_compact_rows(summary_rows, slot_order)
    printable_rows = [
        {"Slot num": row["Slot num"], latency_header: row["Latency (us)"]}
        for row in compact_rows
    ]
    write_csv_rows(compact_out, ["Slot num", latency_header], printable_rows)

    print("\n" + "=" * 64)
    print(f"{latency_header} slot benchmark (us)")
    print("=" * 64)
    print(format_table(compact_rows, latency_header))
    print("=" * 64)

    logging.info("Wrote %s", summary_out)
    if raw_out is not None:
        logging.info("Wrote %s", raw_out)
    logging.info("Wrote %s", compact_out)
    logging.info("Wrote %s", run_log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
