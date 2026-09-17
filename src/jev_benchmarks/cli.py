from __future__ import annotations

import argparse

from .config import load_config
from .data import prepare_manifest
from .report import build_report
from .runner import run_backend


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "report"):
        subcommand = commands.add_parser(command)
        subcommand.add_argument("--config", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--backend", required=True, choices=("gliner", "jev"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        print(prepare_manifest(config))
    elif args.command == "run":
        print(run_backend(config, args.backend))
    elif args.command == "report":
        print("\n".join(str(path) for path in build_report(config)))
