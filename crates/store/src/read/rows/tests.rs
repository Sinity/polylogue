use super::*;
use crate::fixture::Fixture;

#[test]
fn row_visit_preserves_query_values_and_stops_at_callback_failure() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let sql = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1000) SELECT x, NULL AS absent, x'007fff' AS bytes, 'é' AS word FROM n";
    let collected = reader.query(sql, &[]).unwrap();
    let mut visited = Vec::new();
    reader
        .query_each(sql, &[], |row| {
            visited.push(row);
            Ok::<_, StoreError>(())
        })
        .unwrap();
    assert_eq!(visited, collected);
    let mut visits = 0;
    let error = reader
        .query_each(sql, &[], |_| {
            visits += 1;
            Err::<(), _>(StoreError::Invalid("output stopped".into()))
        })
        .unwrap_err();
    assert_eq!(visits, 1);
    assert_eq!(error.to_string(), "output stopped");
    assert_eq!(
        reader
            .query("SELECT ?1 AS n", &[SqlValue::Int(7), SqlValue::Int(9)])
            .unwrap()[0]["n"],
        7
    );
}

#[test]
fn counts_read_aggregate_columns_by_position() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query_counts(
                "SELECT ?1 AS arbitrary, 2 AS other",
                &[SqlValue::Int(3), SqlValue::Int(8)]
            )
            .unwrap(),
        (3, 2)
    );
    assert_eq!(reader.query_counts("SELECT count(*), count(*) FILTER (WHERE x IS NULL) FROM (SELECT NULL AS x WHERE 0)", &[]).unwrap(), (0, 0));
}

#[test]
fn caller_sql_may_only_read() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query_untrusted("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<3) SELECT count(*) AS c, upper('a') AS u FROM n")
            .unwrap(),
        vec![serde_json::Map::from_iter([("c".into(), 3.into()), ("u".into(), "A".into())])]
    );
    let temp_objects = |reader: &crate::read::Reader<'_>| {
        reader
            .query(
                "SELECT type, name FROM temp.sqlite_master ORDER BY type, name",
                &[],
            )
            .unwrap()
    };
    let databases = |reader: &crate::read::Reader<'_>| {
        reader
            .query("SELECT name FROM pragma_database_list ORDER BY name", &[])
            .unwrap()
    };
    // The reader keeps its own temp tables; refused statements must add nothing.
    let query_only = |reader: &crate::read::Reader<'_>| {
        reader
            .query("SELECT query_only FROM pragma_query_only", &[])
            .unwrap()
    };
    let temp_before = temp_objects(&reader);
    let databases_before = databases(&reader);
    let query_only_before = query_only(&reader);
    for refused in [
        "CREATE TEMP VIEW session AS SELECT 1 AS id",
        "CREATE TEMP TABLE scratch(x)",
        "ATTACH DATABASE ':memory:' AS other",
        "PRAGMA query_only = 0",
        "DETACH DATABASE main",
    ] {
        assert!(
            reader.query_untrusted(refused).is_err(),
            "{refused} was not refused"
        );
    }
    // Refusal leaves the shared connection as it was for ordinary reads.
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
    assert_eq!(temp_objects(&reader), temp_before);
    assert_eq!(databases(&reader), databases_before);
    assert_eq!(query_only(&reader), query_only_before);
}
