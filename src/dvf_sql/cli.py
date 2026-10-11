"""dvf-sql: ask one question, or run the eval set and write predictions.

    dvf-sql ask "Combien de ventes à Bordeaux en 2024 ?"
    dvf-sql eval ../dvf-text-to-sql-eval/dvf_eval_questions.yaml -o runs/baseline.jsonl
    dvf-sql prompt                      # print the system prompt (with the live schema)
    dvf-sql --base-url http://localhost:11434 --model qwen3:14b ask "..."   # Ollama

Settings come from the environment: ANTHROPIC_API_KEY, DVF_DSN (default
postgresql://dvf:dvf@localhost:5433/dvf), DVF_SQL_MODEL (default claude-opus-5-5),
DVF_SQL_BASE_URL (any server speaking the Anthropic Messages API, e.g. Ollama;
then no Anthropic key is needed).
"""

import argparse
import datetime as dt
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import yaml

from dvf_sql.db import Database
from dvf_sql.engine import Answer, Engine, local_today

DEFAULT_DSN = "postgresql://dvf:dvf@localhost:5433/dvf"
DEFAULT_MODEL = "claude-opus-5-5"


def build_client(base_url: str | None) -> anthropic.Anthropic:
    if base_url:
        # A local server (Ollama...): it ignores the key, the SDK requires one.
        # Local models can take many minutes per turn: no 10-minute default.
        return anthropic.Anthropic(
            base_url=base_url, api_key="local", max_retries=1, timeout=3600
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (see .env.example)")
    return anthropic.Anthropic(max_retries=4)


def build_engine(args: argparse.Namespace, with_client: bool = True) -> Engine:
    db = Database(os.environ.get("DVF_DSN", DEFAULT_DSN))
    return Engine(
        db,
        build_client(args.base_url) if with_client else None,
        model=args.model,
        max_turns=args.max_turns,
        thinking=not args.no_thinking,
        effort=args.effort,
    )


def cmd_ask(args: argparse.Namespace) -> int:
    engine = build_engine(args)
    answer = engine.ask(args.question, args.today)
    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=1, default=str))
        return 0
    print(f"[{answer.behavior}] {answer.message or answer.error}")
    if answer.assumptions:
        print("hypothèses: " + "; ".join(answer.assumptions))
    if answer.sql:
        print(f"\n{answer.sql}\n")
        print(" | ".join(answer.columns))
        for row in answer.rows[:30]:
            print(" | ".join(str(v) for v in row))
        if len(answer.rows) > 30 or answer.truncated:
            print("...")
    print(
        f"\n({answer.turns} turns, {answer.latency_s}s, {answer.usage}, {answer.model})"
    )
    return 0 if answer.behavior != "error" else 1


def cmd_eval(args: argparse.Namespace) -> int:
    spec = yaml.safe_load(args.questions.read_text())
    today = dt.date.fromisoformat(str(spec["eval_today"]))
    questions = spec["questions"]
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    done: set[str] = set()
    if args.output.exists():  # resume: skip ids already predicted (not errors)
        for line in args.output.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("behavior") != "error":
                done.add(rec["id"])
    todo = [q for q in questions if q["id"] not in done][: args.limit]
    print(
        f"eval-v{spec['eval_version']}: {len(todo)} to run, {len(done)} already done,"
        f" today={today}, model={args.model}",
        file=sys.stderr,
    )
    engine = build_engine(args)
    engine.schema  # noqa: B018  introspect once, before the threads start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def one(q: dict) -> Answer:
        # Only the question text reaches the engine: never notes, gold or category.
        try:
            return engine.ask(q["question"], today)
        except Exception as exc:  # noqa: BLE001
            # An uncaught exception would surface only after the pool has run
            # every remaining question, and nothing would be written. Record it
            # as an error instead: the next run retries it.
            print(f"{q['id']}: {traceback.format_exc()}", file=sys.stderr)
            return Answer(q["question"], "error", error=f"{type(exc).__name__}: {exc}")

    errors = 0
    with (
        ThreadPoolExecutor(max_workers=args.concurrency) as pool,
        args.output.open("a") as out,
    ):
        futures = {pool.submit(one, q): q for q in todo}
        for n, future in enumerate(as_completed(futures), 1):
            q = futures[future]
            answer = future.result()
            errors += answer.behavior == "error"
            record = {"id": q["id"], "eval_version": spec["eval_version"]}
            record |= answer.to_dict()
            with lock:
                out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                out.flush()
            detail = answer.error or answer.message
            print(
                f"[{n}/{len(todo)}] {q['id']:<7} {answer.behavior:<12} "
                f"{answer.latency_s:>5.1f}s  {detail[:80]!r}",
                file=sys.stderr,
            )
    print(f"wrote {args.output} ({errors} errors)", file=sys.stderr)
    return 0 if not errors else 1


def cmd_prompt(args: argparse.Namespace) -> int:
    print(
        build_engine(args, with_client=False).system_prompt(args.today or local_today())
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=os.environ.get("DVF_SQL_MODEL", DEFAULT_MODEL))
    ap.add_argument(
        "--base-url",
        default=os.environ.get("DVF_SQL_BASE_URL"),
        help="Anthropic-compatible server, e.g. http://localhost:11434 for Ollama",
    )
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    p.add_argument("--today", type=dt.date.fromisoformat)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="run the eval set, write predictions JSONL")
    p.add_argument("questions", type=Path, help="dvf_eval_questions.yaml")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--ids", help="comma-separated ids to run")
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=4)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("prompt", help="print the system prompt")
    p.add_argument("--today", type=dt.date.fromisoformat)
    p.set_defaults(func=cmd_prompt)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
