#!/usr/bin/env python3
import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRIMEGEN = ROOT / "primegen32.py"

LOG_SCALE = 50
EVALMOD_LOG_SCALE = 58
NORMAL_LEVEL = 10
EVALMOD_LEVEL = 16
CTS_LEVEL = 4
DNUM = 4
NUM_STC_LEVELS = 3
DEFAULT_EVALMOD_MODE = "legacy"

DEFAULT_OUT = ROOT.parent / "parameters" / f"bootparam_50_{NORMAL_LEVEL}_{EVALMOD_LEVEL}.json"


def parse_int_list(line: str) -> list[int]:
    payload = line.split(":", 1)[1].strip()
    payload = payload.replace("np.int64(", "").replace(")", "")
    return [int(value) for value in ast.literal_eval(payload)]


def parse_level_config(line: str) -> list[list[int]]:
    pairs = re.findall(r"\{(\d+),\s*(\d+)\}", line)
    return [[int(num_main), int(num_terminal)] for num_main, num_terminal in pairs]


def run_primegen() -> str:
    cmd = [
        sys.executable,
        str(PRIMEGEN),
        "--log_scale",
        str(LOG_SCALE),
        "--evalmod_log_scale",
        str(EVALMOD_LOG_SCALE),
        "--normal_level",
        str(NORMAL_LEVEL),
        "--evalmod_level",
        str(EVALMOD_LEVEL),
        "--cts_level",
        str(CTS_LEVEL),
        "--dnum",
        str(DNUM),
    ]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def build_config(primegen_stdout: str, evalmod_mode: str) -> dict:
    q_primes = None
    terminal_primes = None
    auxiliary_primes = None
    level_config = None

    for line in primegen_stdout.splitlines():
        if line.startswith("Q primes:"):
            q_primes = parse_int_list(line)
        elif line.startswith("T primes:"):
            terminal_primes = parse_int_list(line)
        elif line.startswith("P primes:"):
            auxiliary_primes = parse_int_list(line)
        elif line.startswith("State list (num_main + irr_base, num_terminal):"):
            level_config = parse_level_config(line)

    if q_primes is None:
        raise ValueError("Failed to parse Q primes from primegen32.py output")
    if terminal_primes is None:
        raise ValueError("Failed to parse T primes from primegen32.py output")
    if auxiliary_primes is None:
        raise ValueError("Failed to parse P primes from primegen32.py output")
    if level_config is None:
        raise ValueError("Failed to parse level_config from primegen32.py output")

    if level_config[-1] != [len(q_primes), len(terminal_primes)]:
        raise ValueError(
            "Parsed level_config does not match the number of generated primes"
        )

    expected_aux = (len(q_primes) + len(terminal_primes) + DNUM - 1) // DNUM
    if len(auxiliary_primes) != expected_aux:
        raise ValueError("Parsed auxiliary prime count does not match dnum")

    return {
        "log_degree": 16,
        "log_default_scale": LOG_SCALE,
        "boot": True,
        "dense_hamming_weight": 32768,
        "sparse_hamming_weight": 32,
        "num_cts_levels": CTS_LEVEL,
        "num_stc_levels": NUM_STC_LEVELS,
        "evalmod_mode": evalmod_mode,
        "terminal_primes": terminal_primes,
        "main_primes": q_primes,
        "auxiliary_primes": auxiliary_primes,
        "default_encryption_level": NORMAL_LEVEL,
        "level_config": level_config,
        "additional_base": [0, 0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a fixed boot parameter JSON for log_scale=50 testing."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Path to write the generated JSON file",
    )
    parser.add_argument(
        "--evalmod-mode",
        "--evalmod_mode",
        dest="evalmod_mode",
        choices=("legacy", "lattigo_arcsine", "rboot_arcsine"),
        default=DEFAULT_EVALMOD_MODE,
        help="EvalMod mode to encode into the generated JSON file",
    )
    args = parser.parse_args()

    config = build_config(run_primegen(), args.evalmod_mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + "\n", encoding="ascii")

    print(f"Wrote {args.out}")
    print(
        f"evalmod_mode={config['evalmod_mode']}, "
        f"main={len(config['main_primes'])}, "
        f"terminal={len(config['terminal_primes'])}, "
        f"aux={len(config['auxiliary_primes'])}, "
        f"levels={len(config['level_config'])}"
    )


if __name__ == "__main__":
    main()
