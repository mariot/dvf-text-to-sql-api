# dvf-text-to-sql-api

Text-to-SQL over French property sales (DVF, Gironde, 2021-2025): a French
question in, and one of four behaviors out (an answer with its SQL and rows,
a clarifying question, "the data can't answer", or a refusal).

| Repo | Role |
|---|---|
| [`dvf-airflow-pipeline`](https://github.com/mariot/dvf-airflow-pipeline) | loads DVF into PostgreSQL + PostGIS and builds the semantic layer |
| [`dvf-text-to-sql-eval`](https://github.com/mariot/dvf-text-to-sql-eval) | 143 questions with verified gold answers, and `score.py` |
| this repo | the engine (and soon the HTTP API) |

Status: **baseline engine**. The HTTP API comes once the baseline has a score:
every change to the prompt, model or tools is judged on eval-v1.1 first.

## How it works

```
question ─► Claude ──run_sql──► guard ─► read-only Postgres   (explore, a few times)
                    └─submit_answer──► answer | clarify | unanswerable | refuse
                                        └─ answer: the engine runs the final SQL
```

- **Prompt** (`src/dvf_sql/prompt.py`): the product conventions ("prix" =
  median, a "vente" is a mutation, the VEFA trap, Lambert-93 metres...) and the
  semantic layer read live from the catalog (columns, types, comments, data
  coverage, categorical values). It never contains eval questions, notes or
  gold SQL.
- **Tools**: `run_sql` returns the first 50 rows to the model. `submit_answer`
  ends the loop. If the final SQL fails, the error goes back to the model.
- **Grounded answers**: an answer is accepted only once the model has seen the
  result of its final SQL. If it submits a query it never ran, the engine
  runs it, shows the result, and the model must submit again. Otherwise the
  figures in its message could be guesses that contradict the rows the API
  returns (seen with a local 14B model: "125 ventes" over a result of 899).
- **Safety**, three layers on every statement:
  1. `guard.py` (sqlglot): one statement, a query, only `dvf.ventes`, `biens`,
     `lieux`, `zones`, `communes`. No data-modifying CTE, no `SELECT INTO` or
     `FOR UPDATE`, and no server, file, session or sleep functions.
  2. A read-only transaction with a 30 s statement timeout, rolled back.
  3. A prepared statement, which cannot hold two statements, so
     `SELECT 1; COMMIT; DROP ...` can't escape the transaction.

  Planned fourth layer: a dedicated read-only database role (the engine
  currently connects as `dvf`, which owns the tables).
- Every statement actually sent to the database is logged in `executed_sql`.
  The scorer uses it: a refusal passes only if nothing ran.

## Setup

```bash
uv sync --extra dev
cp .env.example .env            # set ANTHROPIC_API_KEY
set -a; source .env; set +a
```

The DVF database must be running (`docker compose up -d` in
`dvf-airflow-pipeline`).

```bash
uv run dvf-sql ask "Quel est le prix médian au m² des appartements à Bordeaux en 2024 ?"
uv run dvf-sql ask --json "..."           # the full Answer object
uv run dvf-sql prompt                     # the exact system prompt
```

Options: `--model` (or `DVF_SQL_MODEL`, default `claude-opus-5-5`),
`--no-thinking` (adaptive thinking is on by default), `--effort
low|medium|high|xhigh|max`, `--max-turns` (default 10).

## Local models (Ollama)

Any server that speaks the Anthropic Messages API works. For Ollama:

```bash
brew install ollama && brew services start ollama && ollama pull qwen3:14b
uv run dvf-sql --base-url http://localhost:11434 --model qwen3:14b ask "..."
```

On an M1 with 16 GB, qwen3:14b runs at about 6 tokens/s, so a question
takes minutes. Run evals with `--concurrency 1`, and set
`OLLAMA_CONTEXT_LENGTH` to at least 12288 so the system prompt and a few
tool turns fit.

## Running the eval

```bash
EVAL=../dvf-text-to-sql-eval
uv run dvf-sql eval $EVAL/dvf_eval_questions.yaml -o runs/baseline.jsonl --concurrency 4
uv run $EVAL/score.py runs/baseline.jsonl --report runs/baseline.report.json
```

- Only the question text reaches the engine, with "today" set to the eval's
  `eval_today`.
- Runs resume: ids already in the output file are skipped, except errors.
  `--ids sim-01,agg-02` reruns a few questions.
- Every prediction records `model` and `engine_version` (`dvf_sql.ENGINE_VERSION`,
  bumped on any prompt, tool or loop change). The scorer warns when a file
  mixes them.
- Report scores with their eval version, e.g. "N/143 on eval-v1.2, model X,
  engine 0.3".
  The scorer refuses to score if the database no longer reproduces the gold
  snapshot (for example after a new DVF release).

## Development

```bash
uv run pytest            # the database tests skip when the DB is down
uv run ruff check . && uv run ruff format --check .
uv run pre-commit install
```
