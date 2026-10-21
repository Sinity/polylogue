use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn readiness_work(messages: i64, ready: bool) -> u64 {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    let tx = conn.transaction().unwrap();
    tx.execute(
        "INSERT INTO input_ref(pk,id,format) VALUES (1,'synthetic-input','synthetic')",
        [],
    )
    .unwrap();
    tx.execute("INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES (1,'claude-code:work','claude-code','work','main',1)", []).unwrap();
    tx.execute("INSERT INTO parse_state(input_pk,chain,parser_version,retired_version,source_eof,source_length,parsed_upto,status) VALUES (1,1,2,1,?1,1,1,'done')", [ready]).unwrap();
    tx.execute(
        "INSERT INTO required_parser(format,version) VALUES ('synthetic',2)",
        [],
    )
    .unwrap();
    tx.execute("INSERT INTO parsed VALUES (1,1,1,1)", [])
        .unwrap();
    for n in 0..messages {
        tx.execute("INSERT INTO message(pk,id,session_pk,order_input_pk,order_input_id,seq,role,written_by,input_pk,parser_version) VALUES (?1,?2,1,1,'synthetic-input',?1,'user','human',1,1)", params![n+1,format!("message-{n}")]).unwrap();
        tx.execute("INSERT INTO message_source(message_pk,input_pk,chain,session_pk,parser_version,source_start,source_end,preferred) VALUES (?1,1,1,1,1,?1,?1+1,1)", [n+1]).unwrap();
    }
    let steps = Arc::new(AtomicU64::new(0));
    let observed = steps.clone();
    tx.progress_handler(
        1,
        Some(move || {
            observed.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    let packed = HashMap::new();
    let ctx = Ctx {
        tx: &tx,
        input_pk: 1,
        input_id: "synthetic-input",
        chain: 1,
        snapshot: None,
        rank: 0,
        parser_version: 2,
        counter: 2,
        packed: &packed,
        bulk: false,
    };
    let affected = ctx.ready_sessions().unwrap();
    tx.progress_handler(0, None::<fn() -> bool>).unwrap();
    let work = steps.load(Ordering::Relaxed);
    assert_eq!(
        affected
            .db
            .query_row("SELECT count(*) FROM item WHERE kind='session'", [], |r| {
                r.get::<_, i64>(0)
            })
            .unwrap(),
        i64::from(ready)
    );
    assert_eq!(
        affected
            .db
            .query_row("SELECT count(*) FROM scope", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        i64::from(ready)
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM message_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        messages
    );
    work
}

#[test]
fn readiness_work_grows_with_contributions_instead_of_their_square() {
    for ready in [true, false] {
        let small = readiness_work(32, ready);
        let large = readiness_work(128, ready);
        println!("ready={ready}: SQL steps32={small},128={large}");
        assert!(
            large < small * 8,
            "ready={ready}: four times the contributions repeated session-wide SQL work: {small}->{large}"
        );
    }
}
