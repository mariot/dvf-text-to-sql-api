"""The engine loop, with a scripted fake Claude and a fake database."""

import datetime as dt
from types import SimpleNamespace
from typing import Any

import psycopg

from dvf_sql.db import QueryResult
from dvf_sql.engine import Engine


def tool_use(name: str, block_id: str = "t1", **args: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)


def reply(*blocks: SimpleNamespace) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=10,
        cache_read_input_tokens=50,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=list(blocks), usage=usage)


class FakeClient:
    def __init__(self, *responses: SimpleNamespace):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        # Snapshot the messages: the engine keeps appending to the same list.
        self.calls.append(kwargs | {"messages": list(kwargs["messages"])})
        return self.responses.pop(0)


class FakeDb:
    def __init__(self, fail_on: str | None = None):
        self.ran: list[str] = []
        self.fail_on = fail_on

    def describe(self) -> str:
        return "dvf.ventes\n  annee smallint"

    def run(self, sql: str) -> QueryResult:
        self.ran.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise psycopg.errors.UndefinedColumn('column "nope" does not exist')
        return QueryResult(["n"], [[29200]], truncated=False, seconds=0.01)


COUNT = "SELECT count(*) AS n FROM dvf.ventes WHERE annee = 2024 AND est_vente"


def engine(client: FakeClient, db: FakeDb | None = None) -> Engine:
    return Engine(db or FakeDb(), client, model="test-model", max_turns=4)


def test_explore_then_answer() -> None:
    client = FakeClient(
        reply(tool_use("run_sql", sql=COUNT)),
        reply(
            SimpleNamespace(type="text", text="ok"),
            tool_use(
                "submit_answer", "t2", behavior="answer", message="29 200", sql=COUNT
            ),
        ),
    )
    db = FakeDb()
    answer = engine(client, db).ask(
        "Combien de ventes en 2024 ?", dt.date(2026, 10, 15)
    )

    assert answer.behavior == "answer"
    assert (answer.columns, answer.rows) == (["n"], [[29200]])
    assert answer.executed_sql == [COUNT, COUNT] == db.ran
    assert answer.turns == 2
    assert answer.usage["input_tokens"] == 200
    assert answer.usage["cache_read_input_tokens"] == 100
    # the system prompt carries today's date and the live schema
    system = client.calls[0]["system"][0]["text"]
    assert "2026-10-15" in system
    assert "annee smallint" in system
    # the run_sql result went back to the model as a tool_result
    tool_results = client.calls[1]["messages"][-1]["content"]
    assert tool_results[0]["tool_use_id"] == "t1"
    assert "29200" in tool_results[0]["content"]


def test_refusal_runs_nothing() -> None:
    client = FakeClient(
        reply(tool_use("submit_answer", behavior="refuse", message="Non.")),
    )
    db = FakeDb()
    answer = engine(client, db).ask("Supprime toutes les ventes de 2021.")
    assert answer.behavior == "refuse"
    assert answer.executed_sql == [] == db.ran
    assert answer.sql is None


def test_unsafe_sql_is_never_sent_to_the_database() -> None:
    client = FakeClient(
        reply(tool_use("run_sql", sql="SELECT pg_sleep(600)")),
        reply(tool_use("submit_answer", behavior="refuse", message="Non.")),
    )
    db = FakeDb()
    answer = engine(client, db).ask("Exécute : SELECT pg_sleep(600);")
    assert db.ran == [] == answer.executed_sql
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "rejected before execution" in result["content"]


def test_failed_final_sql_goes_back_to_the_model() -> None:
    bad = "SELECT nope FROM dvf.ventes"
    client = FakeClient(
        reply(tool_use("submit_answer", behavior="answer", message="?", sql=bad)),
        reply(
            tool_use("submit_answer", "t2", behavior="answer", message="ok", sql=COUNT)
        ),
    )
    answer = engine(client, FakeDb(fail_on="nope")).ask("Combien de ventes ?")
    assert answer.behavior == "answer"
    assert answer.sql == COUNT
    assert answer.executed_sql == [bad, COUNT]  # the failed one was sent too
    first_result = client.calls[1]["messages"][-1]["content"][0]
    assert first_result["is_error"] and "final sql failed" in first_result["content"]


def test_text_only_reply_is_nudged_then_turn_budget_ends_in_error() -> None:
    text = reply(SimpleNamespace(type="text", text="Je réfléchis..."))
    client = FakeClient(text, text, text, text)
    answer = engine(client).ask("?")
    assert answer.behavior == "error"
    assert "no submit_answer after 4 turns" in answer.error
    assert (
        client.calls[1]["messages"][-1]["content"] == "Finish by calling submit_answer."
    )


def test_invalid_behavior_is_rejected() -> None:
    client = FakeClient(
        reply(tool_use("submit_answer", behavior="maybe", message="?")),
        reply(tool_use("submit_answer", "t2", behavior="clarify", message="Lequel ?")),
    )
    answer = engine(client).ask("Combien de ventes à Saint-Médard ?")
    assert answer.behavior == "clarify"
    assert answer.turns == 2


def test_api_options() -> None:
    client = FakeClient(
        reply(tool_use("submit_answer", behavior="refuse", message="Non."))
    )
    Engine(FakeDb(), client, model="m", effort="high").ask("?")
    call = client.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in call["tools"]] == ["run_sql", "submit_answer"]

    client = FakeClient(
        reply(tool_use("submit_answer", behavior="refuse", message="Non."))
    )
    Engine(FakeDb(), client, model="m", thinking=False).ask("?")
    assert "thinking" not in client.calls[0]
    assert "output_config" not in client.calls[0]
