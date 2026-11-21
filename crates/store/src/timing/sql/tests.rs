use super::*;

fn traced(conn: &Connection) -> Worker {
    STATE.with(|state| {
        assert!(state.borrow().is_none());
        *state.borrow_mut() = Some(State::default());
    });
    attach(conn);
    Worker(true)
}
fn memory() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE item(id INTEGER PRIMARY KEY,value INTEGER);
        INSERT INTO item VALUES(1,17),(2,23);",
    )
    .unwrap();
    conn
}
fn reads(conn: &Connection) -> (Vec<i64>, i32) {
    let mut statement = conn
        .prepare_cached("SELECT value FROM item WHERE id=?1")
        .unwrap();
    let mut values = Vec::new();
    for n in 0..64 {
        values.push(statement.query_row([n % 2 + 1], |r| r.get(0)).unwrap());
    }
    (
        values,
        statement.get_status(rusqlite::StatementStatus::VmStep),
    )
}
#[test]
fn cached_statement_vm_counts_match_an_independent_untraced_execution() {
    let baseline = memory();
    let (expected, vm) = reads(&baseline);
    let conn = memory();
    let _worker = traced(&conn);
    let actual = {
        let _phase = phase(&conn, "lookup");
        reads(&conn).0
    };
    assert_eq!(actual, expected);
    STATE.with(|state| {
        let state = state.borrow();
        let state = state.as_ref().unwrap();
        let work = state
            .statements
            .values()
            .find(|r| r.verb == "SELECT")
            .unwrap();
        assert_eq!(work.calls, 64);
        assert_eq!(work.vm_steps, u64::try_from(vm).unwrap());
        assert_eq!(work.phase, "lookup");
        assert!(state.active.is_empty());
    });
}
#[test]
fn file_cache_work_and_commit_have_distinct_observed_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.db");
    let mut conn = Connection::open(&path).unwrap();
    conn.execute_batch("CREATE TABLE item(id INTEGER PRIMARY KEY,value INTEGER,payload BLOB);")
        .unwrap();
    let tx = conn.transaction().unwrap();
    for n in 0..512 {
        tx.execute("INSERT INTO item VALUES(?1,?1,zeroblob(4096))", [n])
            .unwrap();
    }
    tx.commit().unwrap();
    drop(conn);
    let mut conn = Connection::open(&path).unwrap();
    conn.execute_batch("PRAGMA cache_size=-8; PRAGMA mmap_size=0;")
        .unwrap();
    let _worker = traced(&conn);
    {
        let _phase = phase(&conn, "lookup");
        for n in (0..512).step_by(17) {
            let v: i64 = conn
                .query_row("SELECT value FROM item WHERE id=?1", [n], |r| r.get(0))
                .unwrap();
            assert_eq!(v, n);
        }
    }
    let tx = conn.transaction().unwrap();
    {
        let _phase = phase(&tx, "write");
        tx.execute("UPDATE item SET value=value+1 WHERE id=400", [])
            .unwrap();
    }
    commit(tx, "commit").unwrap();
    assert_eq!(
        conn.query_row("SELECT value FROM item WHERE id=400", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        401
    );
    STATE.with(|state| {
        let state = state.borrow();
        let state = state.as_ref().unwrap();
        assert!(state.phases["lookup"].cache_misses > 0);
        assert_eq!(state.phases["commit"].calls, 1);
        assert!(state.phases["write"].cache_writes + state.phases["commit"].cache_writes > 0);
        assert!(
            state
                .statements
                .values()
                .any(|r| r.phase == "write" && r.verb == "UPDATE" && r.calls == 1)
        );
        assert!(
            state
                .statements
                .values()
                .any(|r| r.phase == "commit" && r.verb == "COMMIT" && r.calls == 1)
        );
    });
}
#[test]
fn interrupted_sql_records_its_work_without_changing_the_refusal() {
    let conn = memory();
    let _worker = traced(&conn);
    let mut ticks = 0;
    conn.progress_handler(
        100,
        Some(move || {
            ticks += 1;
            ticks > 10
        }),
    )
    .unwrap();
    {
        let _phase = phase(&conn, "cancelled_lookup");
        let result=conn.query_row("WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<1000000) SELECT sum(n) FROM x",[],|r|r.get::<_,i64>(0));
        assert_eq!(
            result.unwrap_err().sqlite_error_code(),
            Some(rusqlite::ErrorCode::OperationInterrupted)
        );
    }
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    STATE.with(|state| {
        let state = state.borrow();
        let state = state.as_ref().unwrap();
        assert!(state.active.is_empty());
        assert!(
            state
                .statements
                .values()
                .any(|r| r.phase == "cancelled_lookup" && r.vm_steps > 0)
        );
        assert_eq!(state.phases["cancelled_lookup"].calls, 1);
    });
}
#[test]
fn statement_overflow_is_bounded_and_explicit() {
    let conn = memory();
    let _worker = traced(&conn);
    for n in 0..STATEMENTS + 17 {
        assert_eq!(
            conn.query_row(&format!("SELECT {n}"), [], |r| r.get::<_, usize>(0))
                .unwrap(),
            n
        );
    }
    STATE.with(|state| {
        let state = state.borrow();
        let state = state.as_ref().unwrap();
        assert_eq!(state.statements.len(), STATEMENTS);
        assert_eq!(state.dropped_statements, 17);
        assert!(state.active.is_empty());
    });
}
