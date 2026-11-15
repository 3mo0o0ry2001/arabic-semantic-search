"""Fetch the datasets (MIT-licensed SOQAL repository, Mozannar et al., 2019).

ARCD is rebuilt from its train+test files (693 + 702 = 1,395 questions); Arabic-SQuAD
is used only for distractor passages. Files are written to data/.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/husseinmozannar/SOQAL/master/data/"
DATA = Path(__file__).resolve().parents[1] / "data"


def fetch(name: str) -> bytes:
    with urllib.request.urlopen(BASE + name, timeout=120) as r:
        return r.read()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    parts = [json.loads(fetch(n)) for n in ("arcd-train.json", "arcd-test.json")]
    merged = {"data": parts[0]["data"] + parts[1]["data"], "version": parts[0].get("version", "1.1")}
    n = sum(len(p["qas"]) for a in merged["data"] for p in a["paragraphs"])
    assert n == 1395, n
    (DATA / "arcd.json").write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    (DATA / "arabic-squad.json").write_bytes(fetch("Arabic-SQuAD.json"))
    print("wrote data/arcd.json and data/arabic-squad.json")


if __name__ == "__main__":
    main()
