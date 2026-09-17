from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import make_config

from jev_benchmarks import cli, data, report, runner


@pytest.mark.parametrize("command", ["prepare", "report"])
def test_parser_requires_config(command: str) -> None:
    args = cli._parser().parse_args([command, "--config", "config.yaml"])
    assert args.command == command


@pytest.mark.parametrize("command", ["prepare", "run", "report"])
def test_main_dispatches(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(data, "prepare_manifest", lambda config: Path("manifest.jsonl"))
    monkeypatch.setattr(runner, "run_backend", lambda config, backend: Path(f"{backend}.jsonl"))
    monkeypatch.setattr(
        report,
        "build_report",
        lambda config: (Path("report.json"), Path("report.md")),
    )
    argv = ["jev-bench", command, "--config", "config.yaml"]
    if command == "run":
        argv.extend(["--backend", "jev"])
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert capsys.readouterr().out.strip()
