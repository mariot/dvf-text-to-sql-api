"""Database safety, against the real local DVF database (skipped when it's down)."""

import os
from decimal import Decimal

import psycopg
import pytest

from dvf_sql.db import Database, to_json
from dvf_sql.guard import UnsafeSQLError

DSN = os.environ.get("DVF_DSN", "postgresql://dvf:dvf@localhost:5433/dvf")


@pytest.fixture(scope="module")
def db() -> Database:
    try:
        psycopg.connect(DSN, connect_timeout=2).close()
    except psycopg.OperationalError:
        pytest.skip(f"no DVF database at {DSN}")
    return Database(DSN, timeout_s=2, max_rows=3)


def test_to_json() -> None:
    assert to_json(Decimal("4217.00")) == 4217
    assert isinstance(to_json(Decimal("4217.00")), int)
    assert to_json(Decimal("12.5")) == 12.5


def test_run_returns_rows_and_truncates(db: Database) -> None:
    result = db.run("SELECT x FROM generate_series(1, 10) AS g(x)")
    assert result.columns == ["x"]
    assert result.rows == [[1], [2], [3]]
    assert result.truncated


def test_unqualified_semantic_tables_resolve(db: Database) -> None:
    assert db.run("SELECT count(*) > 0 AS ok FROM ventes").rows == [[True]]


def test_statement_timeout(db: Database) -> None:
    with pytest.raises(psycopg.errors.QueryCanceled):
        db.run("SELECT count(*) FROM dvf.ventes a, dvf.ventes b")


def test_guard_runs_before_the_database(db: Database) -> None:
    with pytest.raises(UnsafeSQLError):
        db.run("SELECT 1; DROP TABLE dvf.lieux")
    assert db.run("SELECT count(*) AS n FROM dvf.lieux").rows[0][0] > 0


def test_describe_lists_only_the_semantic_layer(db: Database) -> None:
    schema = db.describe()
    for table in ("dvf.ventes", "dvf.biens", "dvf.lieux", "dvf.zones", "dvf.communes"):
        assert f"\n{table}\n" in f"\n{schema}\n"
    assert "dvf.mutations" not in schema
    assert "Data coverage: date_mutation from" in schema
