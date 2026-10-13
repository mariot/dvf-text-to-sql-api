import pytest

from dvf_sql.guard import (
    COUNT_NOTE,
    SALES_NOTE,
    UnsafeSQLError,
    check_sql,
    commune_codes,
    review_notes,
)

ALLOWED = [
    "SELECT count(*) FROM dvf.ventes WHERE annee = 2024 AND est_vente",
    "SELECT nom_commune FROM ventes",  # search_path = dvf
    (
        "WITH t AS (SELECT annee, count(*) AS n FROM dvf.ventes GROUP BY 1) "
        "SELECT annee, n, n - lag(n) OVER (ORDER BY annee) FROM t"
    ),
    "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY prix_m2) FROM dvf.ventes",
    (
        "SELECT count(*) FROM dvf.ventes v, dvf.lieux l "
        "WHERE l.nom = 'Gare Saint-Jean' AND ST_DWithin(v.geom, l.geom, 1000)"
    ),
    "SELECT ST_X(ST_Transform(geom, 4326)) FROM dvf.biens LIMIT 1",
    "SELECT x FROM generate_series(1, 3) AS g(x)",
    "SELECT 1 UNION ALL SELECT 2",
    "SELECT v.code_commune FROM dvf.ventes v JOIN dvf.zones z USING (code_commune)",
    "SELECT * FROM dvf.communes WHERE nom_commune ILIKE 'saint-m%'",
    "SELECT loyer_m2 FROM dvf.loyers WHERE code_commune = '33063' AND annee = 2025",
    "SELECT count(*) FROM dvf.acces_adresses WHERE min_college < 10",
    "SELECT pole, minutes FROM dvf.temps_trajet WHERE code_commune = '33281'",
    "SELECT * FROM dvf.chercher_commune('bordeau')",
    (
        "SELECT count(*) FROM dvf.ventes v "
        "JOIN dvf.chercher_commune('Mérignac') c USING (code_commune) WHERE v.annee = 2024"
    ),
    (
        "SELECT e.nom, i.ips FROM dvf.ecoles e JOIN dvf.ecoles_ips i USING (uai) "
        "WHERE i.rentree = '2025-2026'"
    ),
]

REJECTED = [
    ("SELECT 1; DROP TABLE dvf.ventes", "exactly one statement"),
    ("DELETE FROM dvf.ventes WHERE annee = 2021", "only queries"),
    ("UPDATE dvf.ventes SET valeur_fonciere = 1", "only queries"),
    ("WITH d AS (DELETE FROM dvf.ventes RETURNING *) SELECT * FROM d", "Delete"),
    ("SELECT * INTO copie FROM dvf.ventes", "Into"),
    ("SELECT * FROM dvf.ventes FOR UPDATE", "Lock"),
    ("COPY (SELECT 1) TO '/tmp/x'", "only queries"),
    ("CREATE TABLE t AS SELECT 1", "only queries"),
    ("EXPLAIN ANALYZE SELECT 1", "only queries"),
    ("SELECT pg_sleep(600)", "pg_sleep"),
    ("SELECT pg_catalog.pg_read_file('/etc/passwd')", "pg_read_file"),
    ("SELECT lo_import('/etc/passwd')", "lo_import"),
    ("SELECT * FROM dblink('h', 'select 1') AS t(a int)", "dblink"),
    ("SELECT set_config('transaction_read_only', 'off', true)", "set_config"),
    ("SELECT version()", "not allowed"),
    ("SELECT inet_server_addr()", "inet_server_addr"),
    ("SELECT usename, passwd FROM pg_shadow", "pg_shadow"),
    ("SELECT * FROM information_schema.tables", "information_schema"),
    ("SELECT * FROM dvf.mutations", "dvf.mutations"),
    ("SELECT * FROM dvf.load_log", "load_log"),
    ("SELECT * FROM dvf.ref_loyers", "ref_loyers"),  # raw tables behind the views
    ("SELECT * FROM dvf.osm_pieton", "osm_pieton"),
    ("SELECT * FROM public.ventes", "public.ventes"),
    ("SELEC nonsense", "only queries|could not parse"),
    # qwen3:14b wrote this: the second apostrophe is escaped, the first is not.
    (
        (
            "SELECT count(*) FROM dvf.ventes "
            "WHERE nature_mutation = 'Vente en l'état futur d''achèvement'"
        ),
        "could not parse",
    ),
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(sql: str) -> None:
    assert check_sql(sql) == sql


@pytest.mark.parametrize(("sql", "reason"), REJECTED)
def test_rejected(sql: str, reason: str) -> None:
    with pytest.raises(UnsafeSQLError, match=reason):
        check_sql(sql)


def test_cte_name_cannot_shadow_a_schema_table() -> None:
    # A CTE named like a forbidden table is fine; a schema-qualified one is not.
    check_sql("WITH mutations AS (SELECT 1 AS a) SELECT a FROM mutations")
    with pytest.raises(UnsafeSQLError):
        check_sql("WITH mutations AS (SELECT 1 AS a) SELECT * FROM dvf.mutations")


@pytest.mark.parametrize(
    ("sql", "codes"),
    [
        ("SELECT * FROM dvf.ventes WHERE code_commune = '33063'", {"33063"}),
        ("SELECT 1 FROM dvf.ventes v WHERE '33063' = v.code_commune", {"33063"}),
        (
            "SELECT 1 FROM dvf.ventes WHERE code_commune IN ('33063', '33281')",
            {"33063", "33281"},
        ),
        (
            "SELECT 1 FROM dvf.ventes WHERE code_commune = ANY(ARRAY['33049'])",
            {"33049"},
        ),
        ("SELECT 1 FROM dvf.ventes WHERE code_commune LIKE '33063'", {"33063"}),
        # Postal codes, years, names and subqueries on other columns: no.
        ("SELECT 1 FROM dvf.ventes WHERE code_postal = '33000'", set()),
        ("SELECT 1 FROM dvf.ventes WHERE annee = 2024", set()),
        ("SELECT 1 FROM dvf.communes WHERE nom_commune = 'Bordeaux'", set()),
        (
            (
                "SELECT 1 FROM dvf.ventes WHERE code_commune IN "
                "(SELECT code_commune FROM dvf.ventes WHERE code_postal = '33000')"
            ),
            set(),
        ),
        ("SELECT * FROM dvf.chercher_commune('Bordeaux')", set()),
        ("SELEC nonsense '33063'", set()),
    ],
)
def test_commune_codes(sql: str, codes: set[str]) -> None:
    assert commune_codes(sql) == codes


@pytest.mark.parametrize(
    ("sql", "notes"),
    [
        ("SELECT count(*) FROM dvf.ventes WHERE annee = 2024", [SALES_NOTE]),
        ("SELECT count(*) FROM dvf.biens b WHERE b.annee = 2024", [SALES_NOTE]),
        ("SELECT count(*) FROM dvf.ventes WHERE annee = 2024 AND est_vente", []),
        (
            "SELECT avg(prix_m2) FROM dvf.ventes WHERE annee = 2024",
            [COUNT_NOTE],
        ),
        (
            (
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY valeur_fonciere), "
                "count(*) FROM dvf.ventes WHERE vente_reference"
            ),
            [],
        ),
        (
            (
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY loyer_m2) "
                "FROM dvf.loyers WHERE annee = 2025"
            ),
            [COUNT_NOTE],
        ),
        ("SELECT loyer_m2 FROM dvf.loyers WHERE annee = 2025", []),
        ("SELEC nonsense", []),
    ],
)
def test_review_notes(sql: str, notes: list[str]) -> None:
    assert review_notes(sql) == notes
