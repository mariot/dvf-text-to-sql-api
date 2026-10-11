"""Read-only access to the DVF database, for model-written SQL."""

import datetime as dt
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import psycopg

from dvf_sql.guard import ALLOWED_SCHEMA, ALLOWED_TABLES, check_sql


@dataclass(frozen=True, slots=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool  # more than max_rows rows: only the first max_rows are kept
    seconds: float


def to_json(value: Any) -> Any:
    """Postgres values to plain JSON values (numbers stay numbers)."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return str(value)
    if isinstance(value, (bytes, memoryview)):
        return "<binary>"
    return value


class Database:
    def __init__(self, dsn: str, timeout_s: int = 30, max_rows: int = 1000):
        self.dsn, self.timeout_s, self.max_rows = dsn, timeout_s, max_rows

    def run(self, sql: str) -> QueryResult:
        """Run ONE guarded query in a read-only transaction.

        Raises guard.UnsafeSQLError (nothing was sent to the database) or
        psycopg.Error (the database rejected or cancelled it).
        """
        check_sql(sql)
        started = time.monotonic()
        with psycopg.connect(self.dsn, autocommit=False) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute(f"SET LOCAL statement_timeout = '{int(self.timeout_s)}s'")
            conn.execute(f"SET LOCAL search_path = {ALLOWED_SCHEMA}, public")
            # A prepared statement holds exactly one statement.
            cur = conn.execute(sql, prepare=True)
            if cur.description is None:
                raise psycopg.ProgrammingError("the statement returned no rows")
            raw = cur.fetchmany(self.max_rows + 1)
            columns = [d.name for d in cur.description]
            conn.rollback()
        return QueryResult(
            columns=columns,
            rows=[[to_json(v) for v in row] for row in raw[: self.max_rows]],
            truncated=len(raw) > self.max_rows,
            seconds=time.monotonic() - started,
        )

    def describe(self) -> str:
        """The semantic layer as the model sees it: columns, types, comments,
        and the data coverage. Read from the catalog so it never drifts from
        sql/semantic.sql in dvf-airflow-pipeline."""
        with psycopg.connect(self.dsn) as conn:
            cols = conn.execute(
                """
                SELECT c.relname, obj_description(c.oid, 'pg_class'),
                       a.attname, format_type(a.atttypid, a.atttypmod),
                       col_description(c.oid, a.attnum)
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_attribute a ON a.attrelid = c.oid
                WHERE n.nspname = %s AND c.relname = ANY(%s)
                  AND a.attnum > 0 AND NOT a.attisdropped
                ORDER BY c.relname, a.attnum
                """,
                (ALLOWED_SCHEMA, sorted(ALLOWED_TABLES)),
            ).fetchall()
            coverage = conn.execute(
                "SELECT min(date_mutation), max(date_mutation), "
                "array_agg(DISTINCT annee ORDER BY annee) FROM dvf.ventes"
            ).fetchone()
            values = {
                "dvf.ventes.nature_mutation": "SELECT DISTINCT nature_mutation "
                "FROM dvf.ventes ORDER BY 1",
                "dvf.ventes.type_bien": "SELECT DISTINCT type_bien "
                "FROM dvf.ventes ORDER BY 1",
                "dvf.biens.type_local": "SELECT DISTINCT type_local "
                "FROM dvf.biens ORDER BY 1",
                "dvf.lieux.nom": "SELECT nom FROM dvf.lieux ORDER BY 1",
                "dvf.zones.zone": "SELECT DISTINCT zone FROM dvf.zones ORDER BY 1",
                "dvf.loyers.segment": "SELECT DISTINCT segment FROM dvf.loyers ORDER BY 1",
                "dvf.loyers.annee": "SELECT DISTINCT annee FROM dvf.loyers ORDER BY 1",
                "dvf.logements_sociaux.annee": "SELECT DISTINCT annee "
                "FROM dvf.logements_sociaux ORDER BY 1",
                "dvf.logements_vacants.niveau": "SELECT DISTINCT niveau "
                "FROM dvf.logements_vacants ORDER BY 1",
                "dvf.ecoles.type": "SELECT DISTINCT type FROM dvf.ecoles ORDER BY 1",
                "dvf.ecoles_ips.rentree": "SELECT DISTINCT rentree "
                "FROM dvf.ecoles_ips ORDER BY 1",
                "dvf.dpe.type_batiment": "SELECT DISTINCT type_batiment "
                "FROM dvf.dpe ORDER BY 1",
                "dvf.dpe.methode": "SELECT DISTINCT methode FROM dvf.dpe ORDER BY 1",
                "dvf.temps_trajet.pole": "SELECT pole FROM dvf.acces_poles ORDER BY 1",
                "dvf.series_longues.pays": "SELECT DISTINCT pays "
                "FROM dvf.series_longues ORDER BY 1",
                "dvf.series_longues.serie (France)": "SELECT DISTINCT serie "
                "FROM dvf.series_longues WHERE pays = 'France' ORDER BY 1",
            }
            distinct = {
                key: [r[0] for r in conn.execute(q).fetchall()]
                for key, q in values.items()
            }

        lines: list[str] = []
        current = None
        for table, table_comment, col, col_type, col_comment in cols:
            if table != current:
                current = table
                lines.append(f"\n{ALLOWED_SCHEMA}.{table}")
                if table_comment:
                    lines.append(f"  -- {table_comment}")
            comment = f"  -- {col_comment}" if col_comment else ""
            lines.append(f"  {col} {col_type}{comment}")
        first, last, years = coverage
        lines.append(
            f"\nData coverage: date_mutation from {first} to {last}; "
            f"annee in {', '.join(map(str, years))}."
        )
        lines.append("Exact values of categorical columns:")
        for key, vals in distinct.items():
            lines.append(f"  {key}: {', '.join(repr(v) for v in vals)}")
        return "\n".join(lines).strip()
