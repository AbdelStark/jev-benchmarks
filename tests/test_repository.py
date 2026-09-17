from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_local_links_resolve() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
    local = [target.split("#", 1)[0] for target in targets if "://" not in target]
    missing = [target for target in local if target and not (ROOT / target).exists()]
    assert missing == []


def test_published_result_has_required_contract() -> None:
    path = ROOT / "results" / "reports" / "btzsc-pilot-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["protocol_revision"] == "pilot-v1-preregistered"
    assert set(payload["results"]) == {"gliner", "jev"}
    assert set(payload["results"]["jev"]) == {"agnews", "banking77", "emotiondair"}
    assert all(len(value) == 64 for value in payload["artifacts"].values())


def test_typed_package_marker_exists() -> None:
    assert (ROOT / "src" / "jev_benchmarks" / "py.typed").is_file()
