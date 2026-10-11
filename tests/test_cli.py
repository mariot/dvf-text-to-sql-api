"""The eval command: one crashing question must not lose the others."""

import argparse
import datetime as dt
import json
from pathlib import Path

import pytest

from dvf_sql import cli
from dvf_sql.engine import Answer

EVAL = """\
eval_version: "9.9"
eval_today: "2026-10-15"
questions:
  - id: q-ok
    question: "Combien de ventes ?"
  - id: q-crash
    question: "Question qui fait planter le moteur"
"""


class CrashingEngine:
    schema = "dvf.ventes"

    def __init__(self) -> None:
        self.asked: list[str] = []

    def ask(self, question: str, today: dt.date) -> Answer:
        self.asked.append(question)
        if "planter" in question:
            raise AttributeError("'str' object has no attribute 'get'")
        return Answer(question, "answer", message="29 200 ventes")


def run_eval(tmp_path: Path, engine: CrashingEngine, monkeypatch) -> list[dict]:
    questions = tmp_path / "eval.yaml"
    questions.write_text(EVAL)
    output = tmp_path / "run.jsonl"
    monkeypatch.setattr(cli, "build_engine", lambda args: engine)
    args = argparse.Namespace(
        questions=questions,
        output=output,
        ids=None,
        limit=None,
        concurrency=1,
        model="test-model",
    )
    cli.cmd_eval(args)
    return [json.loads(line) for line in output.read_text().splitlines()]


def test_crash_is_recorded_as_error_and_the_run_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    records = run_eval(tmp_path, CrashingEngine(), monkeypatch)

    by_id = {r["id"]: r for r in records}
    assert by_id["q-ok"]["behavior"] == "answer"
    assert by_id["q-crash"]["behavior"] == "error"
    assert by_id["q-crash"]["error"].startswith("AttributeError")
    assert "Traceback" in capsys.readouterr().err  # the cause stays diagnosable


def test_resume_retries_only_the_crashed_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_eval(tmp_path, CrashingEngine(), monkeypatch)
    second = CrashingEngine()
    run_eval(tmp_path, second, monkeypatch)

    assert second.asked == ["Question qui fait planter le moteur"]
