"""Static checks on model-written SQL, BEFORE it reaches the database.

This is the first of three layers. The other two live in db.py: a read-only
transaction with a statement timeout, and a prepared statement, which cannot
hold two statements (so "SELECT 1; COMMIT; DROP ..." cannot escape the
transaction). A dedicated read-only database role is the planned fourth layer.

The rules:
  * exactly one statement, and it is a query (SELECT / WITH / UNION ...);
  * no data-modifying or locking clause anywhere (data-modifying CTEs,
    SELECT INTO, FOR UPDATE, COPY...);
  * tables: only the semantic layer (dvf.ventes, biens, lieux, zones,
    communes, and the housing / living-environment views: loyers,
    logements_sociaux, logements_vacants, series_longues, cout_logement_social,
    ecoles, ecoles_ips, colleges_ivac, arrets_transport, dpe, acces_adresses,
    temps_trajet)
    and the query's own CTEs. No raw table, no system catalog;
  * functions: no server, file, session or sleep functions (pg_*, lo_*,
    dblink, set_config, version(), inet_server_addr()...).
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

ALLOWED_SCHEMA = "dvf"
ALLOWED_TABLES = frozenset(
    {"ventes", "biens", "lieux", "zones", "communes"}
    | {
        "loyers",
        "logements_sociaux",
        "logements_vacants",
        "series_longues",
        "cout_logement_social",
        "ecoles",
        "ecoles_ips",
        "colleges_ivac",
        "arrets_transport",
        "dpe",
        "acces_adresses",
        "temps_trajet",
    }
)

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Set,
    exp.Into,
    exp.Lock,
)
FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "inet_", "txid_", "has_")
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "set_config",
        "current_setting",
        "query_to_xml",
        "query_to_xml_and_xmlschema",
        "table_to_xml",
        "cursor_to_xml",
        "schema_to_xml",
        "database_to_xml",
        "version",
    }
)
# Functions sqlglot parses into typed nodes instead of Anonymous.
FORBIDDEN_TYPED_FUNCTIONS: tuple[type[exp.Expression], ...] = (exp.CurrentVersion,)


class UnsafeSQLError(ValueError):
    """The SQL was rejected before execution. The message is safe to show."""


def function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def check_sql(sql: str) -> str:
    """Return the SQL unchanged if it passes every rule, else raise UnsafeSQLError."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except (ParseError, TokenError) as exc:
        # TokenError (e.g. an unbalanced quote in 'l''état') is not a
        # ParseError: without it, the exception escaped and ended the question.
        raise UnsafeSQLError(f"could not parse the SQL: {exc}") from exc
    if len(statements) != 1:
        raise UnsafeSQLError(f"expected exactly one statement, got {len(statements)}")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise UnsafeSQLError(f"only queries are allowed, got {type(tree).__name__}")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeSQLError(f"{type(node).__name__} is not allowed")
        if isinstance(node, FORBIDDEN_TYPED_FUNCTIONS):
            raise UnsafeSQLError(f"function {function_name(node)}() is not allowed")
        if isinstance(node, exp.Func):
            name = function_name(node)
            if name in FORBIDDEN_FUNCTIONS or name.startswith(
                FORBIDDEN_FUNCTION_PREFIXES
            ):
                raise UnsafeSQLError(f"function {name}() is not allowed")
        if isinstance(node, exp.Table):
            check_table(node, cte_names)
    return sql


def check_table(table: exp.Table, cte_names: set[str]) -> None:
    if not isinstance(table.this, exp.Identifier):
        return  # a table function (generate_series, unnest): checked as a function
    name, schema = table.name.lower(), table.db.lower()
    if table.catalog:
        raise UnsafeSQLError(f"table {table.sql()} is not allowed")
    if not schema and name in cte_names:
        return
    if schema in ("", ALLOWED_SCHEMA) and name in ALLOWED_TABLES:
        return
    allowed = ", ".join(f"{ALLOWED_SCHEMA}.{t}" for t in sorted(ALLOWED_TABLES))
    raise UnsafeSQLError(f"table {table.sql()} is not allowed (use only {allowed})")
