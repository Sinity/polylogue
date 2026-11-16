use super::*;
use crate::StoreError;
#[test]
fn scalar_and_optional_preserve_sql_results() {
    let mut conn = Connection::open_in_memory().unwrap();
    assert_eq!(conn.scalar::<i64>("SELECT ?1", [42]).unwrap(), 42);
    assert_eq!(
        conn.scalar_optional::<i64>("SELECT 1 WHERE false", [])
            .unwrap(),
        None
    );
    assert!(conn.exists("SELECT EXISTS(SELECT 1)", []).unwrap());
    assert!(
        !conn
            .exists("SELECT EXISTS(SELECT 1 WHERE false)", [])
            .unwrap()
    );
    assert!(matches!(
        conn.scalar::<i64>("SELECT 1 WHERE false", []),
        Err(StoreError::Sqlite(rusqlite::Error::QueryReturnedNoRows))
    ));
    let tx = conn.transaction().unwrap();
    assert_eq!(
        tx.scalar::<String>("SELECT 'transaction'", []).unwrap(),
        "transaction"
    );
}
#[test]
fn invalid_conversions_keep_the_description() {
    assert_eq!(to_i64(42_u64, "count").unwrap(), 42);
    assert_eq!(to_u64(42_i64, "count").unwrap(), 42);
    for result in [
        to_i64(u64::MAX, "count"),
        None::<i64>.or_invalid("count"),
        Err::<i64, _>(()).or_invalid("count"),
    ] {
        assert!(matches!(result, Err(StoreError::Invalid(message)) if message == "count"));
    }
    assert!(
        matches!(to_u64(-1_i64, "position"), Err(StoreError::Invalid(message)) if message == "position")
    );
}
#[test]
fn metadata_reads_missing_and_replaced_values() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT")
        .unwrap();
    assert_eq!(meta_get(&conn, "version").unwrap(), None);
    meta_set(&conn, "version", "1").unwrap();
    assert_eq!(meta_get(&conn, "version").unwrap().as_deref(), Some("1"));
    meta_set(&conn, "version", "2").unwrap();
    assert_eq!(meta_get(&conn, "version").unwrap().as_deref(), Some("2"));
}
