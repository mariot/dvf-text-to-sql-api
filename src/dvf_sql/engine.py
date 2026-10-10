"""Baseline text-to-SQL engine: Claude with two tools, run_sql and submit_answer.

The model explores with run_sql (guarded, read-only, first rows only), then
ends with submit_answer: one of four behaviors. For "answer", the engine runs
the submitted SQL itself and returns its rows; if it fails, the error goes back
to the model, which gets another try within the turn budget.
"""

import datetime as dt
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

import anthropic
import psycopg

from dvf_sql.db import Database, QueryResult
from dvf_sql.guard import UnsafeSQLError, check_sql
from dvf_sql.prompt import build_system_prompt

BEHAVIORS = ("answer", "clarify", "unanswerable", "refuse")
PREVIEW_ROWS = 50  # rows of a run_sql result shown to the model
PARIS = ZoneInfo("Europe/Paris")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_sql",
        "description": (
            "Run ONE read-only PostgreSQL query on the semantic layer and see its "
            f"columns and first {PREVIEW_ROWS} rows. Use it to explore and to test "
            "queries. Never use it when you are going to refuse."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "One query."}},
            "required": ["sql"],
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Finish. behavior is answer, clarify, unanswerable or refuse. For "
            "answer, sql is the final query whose result answers the question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "behavior": {"type": "string", "enum": list(BEHAVIORS)},
                "message": {
                    "type": "string",
                    "description": (
                        "In French: the short answer, the clarifying question, why "
                        "the data cannot answer, or the refusal."
                    ),
                },
                "sql": {
                    "type": ["string", "null"],
                    "description": "Final query. Required for answer, null otherwise.",
                },
                "assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Interpretation choices made to answer.",
                },
            },
            "required": ["behavior", "message"],
        },
    },
]


@dataclass(slots=True)
class Answer:
    question: str
    behavior: str  # one of BEHAVIORS, or "error"
    message: str = ""
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    truncated: bool = False
    assumptions: list[str] = field(default_factory=list)
    executed_sql: list[str] = field(default_factory=list)  # all SQL sent to the DB
    turns: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    latency_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def local_today() -> dt.date:
    """Today in France: relative dates ("l'an dernier") are French dates."""
    return dt.datetime.now(PARIS).date()


def add_usage(total: dict[str, int], usage: Any) -> None:
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        total[key] = total.get(key, 0) + (getattr(usage, key, None) or 0)


def tool_result(tool_use_id: str, content: Any, is_error: bool = False) -> dict:
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text,
        "is_error": is_error,
    }


def preview(result: QueryResult) -> dict[str, Any]:
    return {
        "columns": result.columns,
        "rows": result.rows[:PREVIEW_ROWS],
        "row_count": len(result.rows),
        "more_rows": result.truncated or len(result.rows) > PREVIEW_ROWS,
    }


class Engine:
    def __init__(
        self,
        db: Database,
        client: anthropic.Anthropic | Any,
        model: str,
        max_turns: int = 10,
        max_tokens: int = 16_000,
        thinking: bool = True,
        effort: str | None = None,
    ):
        self.db, self.client, self.model = db, client, model
        self.max_turns, self.max_tokens = max_turns, max_tokens
        self.thinking, self.effort = thinking, effort
        self._schema: str | None = None

    @property
    def schema(self) -> str:
        if self._schema is None:
            self._schema = self.db.describe()
        return self._schema

    def system_prompt(self, today: dt.date) -> str:
        return build_system_prompt(self.schema, today)

    def ask(self, question: str, today: dt.date | None = None) -> Answer:
        started = time.monotonic()
        answer = self._loop(question, today or local_today())
        answer.latency_s = round(time.monotonic() - started, 2)
        answer.model = self.model
        return answer

    def _create(self, system: str, messages: list[dict]) -> Any:
        options: dict[str, Any] = {}
        if self.thinking:
            options["thinking"] = {"type": "adaptive"}
        if self.effort:
            options["output_config"] = {"effort": self.effort}
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system}],
            tools=TOOLS,
            messages=messages,
            # Caches the growing prefix (system + schema + earlier turns).
            cache_control={"type": "ephemeral"},
            **options,
        )

    def _loop(self, question: str, today: dt.date) -> Answer:
        system = self.system_prompt(today)
        messages: list[dict] = [{"role": "user", "content": question}]
        usage: dict[str, int] = {}
        executed: list[str] = []

        def fail(turns: int, error: str) -> Answer:
            return Answer(
                question,
                "error",
                error=error,
                executed_sql=executed,
                turns=turns,
                usage=usage,
            )

        for turn in range(1, self.max_turns + 1):
            try:
                response = self._create(system, messages)
            except anthropic.APIError as exc:
                return fail(turn, f"{type(exc).__name__}: {exc}")
            add_usage(usage, response.usage)
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                messages.append(
                    {"role": "user", "content": "Finish by calling submit_answer."}
                )
                continue

            results = []
            for block in tool_uses:
                if block.name == "submit_answer":
                    final = self._submit(question, block.input, executed)
                    if isinstance(final, Answer):
                        final.turns, final.usage = turn, usage
                        return final
                    results.append(tool_result(block.id, final, is_error=True))
                elif block.name == "run_sql":
                    results.append(self._run_sql(block, executed))
                else:
                    results.append(
                        tool_result(block.id, f"unknown tool {block.name}", True)
                    )
            messages.append({"role": "user", "content": results})
        return fail(self.max_turns, f"no submit_answer after {self.max_turns} turns")

    def _execute(self, sql: str, executed: list[str]) -> QueryResult:
        check_sql(sql)  # rejected SQL never reaches the database: not "executed"
        executed.append(sql)
        return self.db.run(sql)

    def _run_sql(self, block: Any, executed: list[str]) -> dict:
        sql = (block.input or {}).get("sql") or ""
        try:
            return tool_result(block.id, preview(self._execute(sql, executed)))
        except UnsafeSQLError as exc:
            return tool_result(block.id, f"rejected before execution: {exc}", True)
        except psycopg.Error as exc:
            return tool_result(block.id, f"database error: {exc}".strip(), True)

    def _submit(
        self, question: str, args: dict[str, Any], executed: list[str]
    ) -> Answer | str:
        """An Answer, or an error message for the model to fix and resubmit."""
        behavior = args.get("behavior")
        if behavior not in BEHAVIORS:
            return f"behavior must be one of {', '.join(BEHAVIORS)}"
        answer = Answer(
            question,
            behavior,
            message=args.get("message") or "",
            assumptions=list(args.get("assumptions") or []),
            executed_sql=executed,
        )
        if behavior != "answer":
            return answer
        sql = (args.get("sql") or "").strip()
        if not sql:
            return "behavior answer requires the final sql"
        try:
            result = self._execute(sql, executed)
        except UnsafeSQLError as exc:
            return f"final sql rejected before execution: {exc}. Fix it and resubmit."
        except psycopg.Error as exc:
            return f"final sql failed: {exc}. Fix it and resubmit."
        answer.sql = sql
        answer.columns, answer.rows = result.columns, result.rows
        answer.truncated = result.truncated
        return answer
