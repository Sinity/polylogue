use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

fn database(primary: bool) -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    if primary {
        crate::schema::ensure_primary_index(&conn).unwrap();
    } else {
        crate::schema::ensure_index(&conn).unwrap();
    }
    crate::versions::track_unreferenced(&conn).unwrap();
    conn.execute_batch("INSERT INTO input_ref(pk,id) VALUES(1,'first'),(2,'second'),(3,'unrelated');
        INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(1,'codex:s','codex','s','main',1);
        INSERT INTO message_version VALUES(1,x'01'),(2,x'02'),(3,x'03');
        INSERT INTO message_version_body VALUES(1,x'00'),(2,x'00'),(3,x'00');
        INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk,source_chain)
        VALUES(1,'codex:m',1,'0','assistant','model',1,1);").unwrap();
    conn
}

fn claim(input: i64, version: i64) -> Claim {
    Claim {
        message: 1,
        version,
        reading: Reading {
            input,
            chain: input,
            start: 11,
            end: 29,
            session: 1,
            main: input == 1,
            route: Route::Live,
            completion: Completion::InProgress,
            seq: 17,
            position: 2,
            order_time: Some(41),
            at: Some(37),
            content: false,
        },
    }
}

fn install(conn: &Connection, c: &Claim) -> rusqlite::Result<usize> {
    let r = &c.reading;
    conn.execute(
        "UPDATE message SET input_pk=?2,source_chain=?3,source_start=?4,source_end=?5,
        source_id_kind=?6,source_session_pk=?7,source_main=?8,source_route=?9,
        source_seq=?10,source_record_position=?11,source_order_time=?12,source_at=?13,
        source_completion=?14,source_version_pk=?15 WHERE pk=?1",
        params![
            c.message,
            r.input,
            r.chain,
            r.start,
            r.end,
            r.id_kind(),
            r.session,
            r.main,
            r.route.as_str(),
            r.seq,
            r.position,
            r.order_time,
            r.at,
            r.completion.as_str(),
            c.version
        ],
    )
}

fn snapshot(conn: &Connection) -> Vec<String> {
    conn.prepare(
        "SELECT json_array(message_pk,input_pk,chain,source_start,source_end,id_kind,
        session_pk,main,route,seq,record_position,order_time,at,completion,version_pk)
        FROM message_source ORDER BY message_pk,input_pk,chain",
    )
    .unwrap()
    .query_map([], |r| r.get(0))
    .unwrap()
    .collect::<rusqlite::Result<_>>()
    .unwrap()
}

#[test]
fn promotion_and_retraction_preserve_one_complete_claim_and_roll_back_together() {
    for primary in [false, true] {
        let conn = database(primary);
        conn.execute_batch("BEGIN").unwrap();
        let first = claim(1, 1);
        first.put(&conn).unwrap();
        conn.execute_batch("INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(2,'codex:source-session','codex','source-session','main',1);
            UPDATE message SET source_session_pk=2 WHERE pk=1;").unwrap();
        // The displayed session remains 1. Only the raw inline claim protects 2.
        assert!(conn.execute("DELETE FROM session WHERE pk=2", []).is_err());
        conn.execute_batch(
            "UPDATE message SET source_session_pk=1 WHERE pk=1; DELETE FROM session WHERE pk=2;",
        )
        .unwrap();

        let mut second = claim(2, 2);
        second.reading.completion = Completion::Finished;
        second.reading.route = Route::Export;
        second.reading.order_time = None;
        second.reading.at = Some(19);
        second.put(&conn).unwrap();
        let expected = snapshot(&conn);
        assert_eq!(expected.len(), 2);
        conn.execute_batch("CREATE TEMP TRIGGER fail_promotion BEFORE UPDATE OF source_version_pk ON message
            WHEN NEW.source_version_pk=2 BEGIN SELECT RAISE(ABORT,'synthetic promotion failure'); END;
            SAVEPOINT promotion;").unwrap();
        select(&conn, 1, 2, 2).unwrap();
        assert!(install(&conn, &second).is_err());
        conn.execute_batch(
            "ROLLBACK TO promotion; RELEASE promotion; DROP TRIGGER fail_promotion;",
        )
        .unwrap();
        assert_eq!(snapshot(&conn), expected);
        assert_eq!(
            conn.scalar::<i64>("SELECT source_version_pk FROM message WHERE pk=1", [])
                .unwrap(),
            1
        );
        select(&conn, 1, 2, 2).unwrap();
        install(&conn, &second).unwrap();
        assert_eq!(snapshot(&conn), expected);
        assert_eq!(
            conn.scalar::<i64>("SELECT count(*) FROM message_source_extra", [])
                .unwrap(),
            1
        );
        assert_eq!(
            conn.scalar::<i64>("SELECT source_version_pk FROM message WHERE pk=1", [])
                .unwrap(),
            2
        );
        // The schema also rejects a second physical owner without cooperation
        // from the caller. This checks keys and all FK-bearing references.
        assert!(
            conn.execute(
                "INSERT INTO message_source_extra SELECT * FROM message_source WHERE input_pk=2",
                []
            )
            .is_err()
        );
        assert!(
            conn.execute("UPDATE message SET source_version_pk=999 WHERE pk=1", [])
                .is_err()
        );
        retract(&conn, 2, 2).unwrap();
        select(&conn, 1, 1, 1).unwrap();
        install(&conn, &first).unwrap();
        assert_eq!(snapshot(&conn), vec![expected[0].clone()]);
        assert_eq!(
            conn.scalar::<i64>("SELECT count(*) FROM message_source_extra", [])
                .unwrap(),
            0
        );
        let texts = crate::text_rows::TextIds::new(
            &conn,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap();
        crate::versions::remove_unreferenced(
            &conn,
            &crate::writer::resolve::IndexUpkeep::PerTransaction,
            &texts,
        )
        .unwrap();
        assert_eq!(
            conn.scalar::<i64>("SELECT count(*) FROM message_version WHERE pk=1", [])
                .unwrap(),
            1
        );
        assert_eq!(
            conn.scalar::<i64>("SELECT count(*) FROM message_version WHERE pk=2", [])
                .unwrap(),
            0
        );
        // Moving an identity is not an observation UPSERT. A target that
        // already owns this claim refuses the move and rolls back its source.
        conn.execute_batch("INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk,source_chain)
            VALUES(2,'codex:target',1,'1','assistant','model',1,1);").unwrap();
        let mut target = claim(1, 1);
        target.message = 2;
        install(&conn, &target).unwrap();
        let before_move = snapshot(&conn);
        conn.execute_batch("SAVEPOINT move_collision").unwrap();
        assert!(move_message(&conn, 1, 2, None, false).is_err());
        conn.execute_batch("ROLLBACK TO move_collision; RELEASE move_collision")
            .unwrap();
        assert_eq!(snapshot(&conn), before_move);
        // Once the conflicting reading is actually withdrawn, the same move
        // succeeds and its complete raw claim becomes the selected header.
        remove(&conn, 2, 1, 1).unwrap();
        assert!(!move_message(&conn, 1, 2, None, false).unwrap());
        select(&conn, 2, 1, 1).unwrap();
        install(&conn, &target).unwrap();
        assert_eq!(snapshot(&conn), vec![before_move[1].clone()]);
        assert_eq!(
            conn.scalar::<i64>("SELECT count(*) FROM message_source_extra", [])
                .unwrap(),
            0
        );
        // A target projected from an extra reading receives that reading
        // once. Its other claims move in SQL with their complete metadata.
        let mut additional = claim(2, 3);
        additional.message = 2;
        additional.put(&conn).unwrap();
        conn.execute_batch("INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk,source_chain)
            VALUES(3,'codex:projected',1,'2','assistant','model',2,2);").unwrap();
        let mut projected = claim(2, 3);
        projected.message = 3;
        install(&conn, &projected).unwrap();
        assert!(move_message(&conn, 2, 3, None, true).unwrap());
        let transferred: Vec<(i64, i64, i64)> = conn
            .prepare("SELECT message_pk,input_pk,version_pk FROM message_source ORDER BY input_pk")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(transferred, [(3, 1, 1), (3, 2, 3)]);

        assert_eq!(
            conn.scalar::<i64>("SELECT count(*) FROM pragma_foreign_key_check", [])
                .unwrap(),
            0
        );
        conn.execute_batch("COMMIT").unwrap();
    }
}

fn steps(conn: &Connection, sql: &str, values: &[&dyn rusqlite::ToSql]) -> (u64, u64) {
    let count = Arc::new(AtomicU64::new(0));
    let sampled = count.clone();
    conn.progress_handler(
        1,
        Some(move || {
            sampled.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    let mut statement = conn.prepare(sql).unwrap();
    let mut rows = statement.query(values).unwrap();
    let mut row_count = 0;
    while rows.next().unwrap().is_some() {
        row_count += 1;
    }
    drop(rows);
    drop(statement);
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    (row_count, count.load(Ordering::Relaxed))
}

#[test]
fn keyed_claims_predecessors_and_suffixes_do_not_visit_unrelated_readings() {
    let conn = database(false);
    claim(1, 1).put(&conn).unwrap();
    claim(2, 2).put(&conn).unwrap();
    let queries = [
        (
            "SELECT s.* FROM json_each(?1) k CROSS JOIN message_source s ON s.message_pk=k.value ORDER BY s.message_pk,s.input_pk,s.chain",
            2u64,
        ),
        (
            "SELECT pk FROM message WHERE source_session_pk=?1+998",
            0u64,
        ),
        (
            "SELECT seq,record_position,order_time FROM message_source WHERE input_pk=?1 AND chain=1 AND session_pk=1 AND (seq,record_position)<(18,0) ORDER BY seq DESC,record_position DESC LIMIT 1",
            1u64,
        ),
        (
            "SELECT (SELECT chain FROM message_source WHERE input_pk=?1 ORDER BY chain DESC LIMIT 1)",
            1u64,
        ),
        (
            "SELECT message_pk,seq,record_position,at,order_time FROM message_source WHERE input_pk=?1 AND chain=1 AND session_pk=1 AND (seq,record_position)>(16,0)",
            1u64,
        ),
    ];
    let measure = || {
        queries
            .iter()
            .enumerate()
            .map(|(i, (sql, expected))| {
                let observed = if i == 0 {
                    steps(&conn, sql, &[&"[1]"])
                } else {
                    steps(&conn, sql, &[&1i64])
                };
                assert_eq!(observed.0, *expected);
                observed.1
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(conn.scalar::<Option<i64>>("SELECT (SELECT chain FROM message_source WHERE input_pk=3 ORDER BY chain DESC LIMIT 1)",[]).unwrap(),None);
    let before = measure();
    conn.execute_batch("WITH RECURSIVE ids(n) AS (VALUES(2) UNION ALL SELECT n+1 FROM ids WHERE n<8193)
        INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk,source_chain,
            source_start,source_end,source_id_kind,source_session_pk,source_main,source_route,
            source_seq,source_record_position,source_completion,source_version_pk)
        SELECT n,'codex:unrelated-'||n,1,cast(n AS TEXT),'assistant','model',1,3,
            0,1,'native',1,1,'live',n,0,'finished',3 FROM ids;
        INSERT INTO message_source_extra SELECT pk,2,3,0,1,'native',1,1,'live',pk,0,NULL,NULL,'finished',3 FROM message WHERE pk>1;").unwrap();
    let after = measure();
    for (before, after) in before.into_iter().zip(after) {
        assert!(
            after <= before + 128,
            "bounded keyed work: before={before} after={after}"
        );
    }
}
