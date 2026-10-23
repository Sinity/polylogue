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
