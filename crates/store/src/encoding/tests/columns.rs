use super::*;
use polylogue_model::{Role, SessionId};

fn one(value: &str) -> rusqlite::Connection {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch(&format!(
        "CREATE TABLE t(v); INSERT INTO t VALUES ({value});"
    ))
    .unwrap();
    conn
}

fn read<T>(
    conn: &rusqlite::Connection,
    f: impl Fn(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
) -> rusqlite::Result<T> {
    conn.query_row("SELECT v FROM t", [], |row| f(row))
}

#[test]
fn negative_counter_column_is_refused() {
    let conn = one("-1");
    let error = read(&conn, |row| row.get::<_, u64>(0)).unwrap_err();
    assert!(
        matches!(error, rusqlite::Error::IntegralValueOutOfRange(0, -1)),
        "{error:?}"
    );
}

#[test]
fn u64_above_i64_max_is_refused_on_write() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t(v)").unwrap();
    let too_big = u64::try_from(i64::MAX).unwrap() + 1;
    assert!(
        conn.execute("INSERT INTO t VALUES (?1)", [too_big])
            .is_err()
    );
    let count: i64 = conn
        .query_row("SELECT count(*) FROM t", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn stored_values_the_model_cannot_read_are_column_errors() {
    let failure = |error: rusqlite::Error| match error {
        rusqlite::Error::FromSqlConversionFailure(column, _, why) => (column, why.to_string()),
        other => panic!("not a column error: {other:?}"),
    };
    let (column, why) = failure(read(&one("'robot'"), |row| row.name::<Role>(0)).unwrap_err());
    assert_eq!((column, why.as_str()), (0, "unknown stored name \"robot\""));
    let (_, why) = failure(read(&one("'no-origin'"), |row| row.id::<SessionId>(0)).unwrap_err());
    assert!(why.contains("\"no-origin\" does not parse"), "{why}");
    let (_, why) = failure(read(&one("x'0102'"), |row| row.hash(0)).unwrap_err());
    assert_eq!(why, "a stored hash has 2 bytes, not 32");
    let (_, why) = failure(read(&one(&i64::MAX.to_string()), |row| row.time_ms(0)).unwrap_err());
    assert!(why.contains("out of range"), "{why}");
}

#[test]
fn readable_stored_values_decode_and_null_stays_absent() {
    assert_eq!(
        read(&one("'claude-code:s1'"), |row| row.id::<SessionId>(0)).unwrap(),
        SessionId::parse("claude-code:s1").unwrap()
    );
    let null = one("NULL");
    assert_eq!(
        read(&null, |row| row.optional_id::<SessionId>(0)).unwrap(),
        None
    );
    assert_eq!(read(&null, |row| row.optional_hash(0)).unwrap(), None);
    assert_eq!(read(&null, |row| row.optional_time_ms(0)).unwrap(), None);
    assert_eq!(
        read(&one("1700000000000"), |row| row.time_ms(0)).unwrap(),
        Timestamp::from_unix_ms(1_700_000_000_000).unwrap()
    );
}
