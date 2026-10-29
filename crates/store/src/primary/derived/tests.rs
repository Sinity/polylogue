use super::*;
use crate::InputKey;
use crate::primary::Reading;
use polylogue_model::{MessageRow, SessionKind};
use serde_json::json;

fn database() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch(crate::schema::INDEX).unwrap();
    db.execute_batch(super::SCHEMA).unwrap();
    db.execute_batch(crate::primary::projection::SCHEMA)
        .unwrap();
    db.execute_batch("CREATE TEMP TABLE primary_candidate AS SELECT * FROM primary_fact WHERE 0; CREATE TEMP VIEW primary_next AS SELECT * FROM primary_candidate UNION ALL SELECT f.* FROM primary_fact f WHERE NOT EXISTS(SELECT 1 FROM primary_candidate c WHERE c.event=f.event);").unwrap();
    db.execute(
        "INSERT INTO input_ref(pk,id,format) VALUES(1,'input-x','codex')",
        [],
    )
    .unwrap();
    db.execute("INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(1,'codex:morrow','codex','morrow','main',1)",[]).unwrap();
    db
}
fn reading() -> Reading {
    Reading {
        input: InputKey {
            id: "input-x".into(),
            format: "codex".into(),
            arrival_rank: 0,
        },
        chain: 1,
        reading_version: "reader-1".into(),
        order_from: None,
        repeats_whole_conversations: false,
        start: 0,
        parsed_upto: 10,
        at_end: true,
        no_parser_yet: false,
    }
}
fn fact(db: &Connection, event: &str, value: &Product) {
    let reference = if let Product::Session { facts, .. } = value {
        Some(format!("session:{}", facts.id))
    } else {
        None
    };
    db.execute("INSERT INTO primary_candidate(event,admitted,changed,source_order,state,product,input,chain,reference,kind,step_start,step_end,reading_version,session) VALUES(?1,1,1,0,'live',?2,'input-x',1,?3,'parsed',0,10,'reader-1','codex:morrow')",params![event,pack(value).unwrap(),reference]).unwrap();
}
fn invocation(id: &str, call: &str) -> Product {
    Product::Message{reading:reading(),order:None,projection:None,facts:Box::new(MessageRow{offset:0,end:10,message:serde_json::from_value(json!({"id":id,"session":"codex:morrow","role":"assistant","stop_reason":"end_turn","blocks":[{"type":"tool_call","call_id":call,"tool":"shell","action":"shell","input":{},"command":"true"}]})).unwrap()})}
}

fn show(db: &Connection, value: &Product, active: bool) {
    let Product::Message { facts, .. } = value else {
        panic!()
    };
    db.execute("INSERT INTO message(pk,id,session_pk,order_input_pk,order_input_id,seq,role,written_by,input_pk,source_chain,source_start,source_end,on_active_path) VALUES(1,?1,1,1,'input-x',0,'assistant','model',1,1,0,10,?2)",params![facts.message.id.to_string(),active]).unwrap();
    db.execute("INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,shown,session_pk,main,version_hash) VALUES(1,1,1,0,10,1,1,1,?1)",[&hash_message(&facts.message).unwrap()[..]]).unwrap();
}
fn staged(db: &Connection) -> WorkSnapshot {
    let body:Vec<u8>=db.query_row("SELECT body FROM primary_derived WHERE cancelled_at IS NULL ORDER BY source_position DESC LIMIT 1",[],|row|row.get(0)).unwrap();
    crate::encoding::unpack_json(&body).unwrap()
}

#[test]
fn complete_snapshot_uses_selected_full_facts_and_empty_replacement_retires_members() {
    let mut db = database();
    let id = SessionId::parse("codex:morrow").unwrap();
    fact(
        &db,
        "session-event",
        &Product::Session {
            reading: reading(),
            facts: Session::new(id.clone(), SessionKind::Main),
        },
    );
    let call = invocation("codex:call", "native-call");
    fact(&db, "call-event", &call);
    show(&db, &call, true);
    let tx = db.transaction().unwrap();
    stage(&tx, 1, 1, true).unwrap();
    tx.commit().unwrap();
    let first = staged(&db);
    assert_eq!(first.calls.len(), 1);
    assert_eq!(first.calls[0].command.as_deref(), Some("true"));
    assert_eq!(first.messages[0].source_event, "call-event");
    let marker: polylogue_model::Event = serde_json::from_value(
        json!({"session":"codex:morrow","kind":"active_path_empty","range":[0,10],"data":{}}),
    )
    .unwrap();
    db.execute("INSERT INTO event(id,session_pk,seq,kind,input_pk,source_chain,source_start,source_end,fields,data) VALUES('marker-native',1,1,'active_path_empty',1,1,0,10,?1,?2)",params![pack(&marker).unwrap(),pack(&marker.data).unwrap()]).unwrap();
    fact(
        &db,
        "marker-event",
        &Product::Event {
            id: "marker-native".into(),
            reading: reading(),
            facts: marker,
            imported: None,
        },
    );
    let tx = db.transaction().unwrap();
    stage(&tx, 2, 2, true).unwrap();
    tx.commit().unwrap();
    let empty = staged(&db);
    assert!(empty.calls.is_empty());
    assert!(empty.turns.is_empty());
    assert!(empty.messages.is_empty());
    assert_eq!(empty.session_id, id);
    assert_eq!(empty.sources.len(), 3);
}

#[test]
fn replayed_projection_produces_identical_intent_and_snapshot_confirmation_has_no_feedback() {
    let build = || {
        let mut db = database();
        let id = SessionId::parse("codex:morrow").unwrap();
        fact(
            &db,
            "session-event",
            &Product::Session {
                reading: reading(),
                facts: Session::new(id, SessionKind::Main),
            },
        );
        let call = invocation("codex:call", "call-x");
        fact(&db, "call-event", &call);
        show(&db, &call, true);
        let tx = db.transaction().unwrap();
        stage(&tx, 3, 3, true).unwrap();
        tx.commit().unwrap();
        db
    };
    let mut first = build();
    let second = build();
    let identity = |db: &Connection| {
        db.query_row("SELECT identity FROM primary_derived", [], |row| {
            row.get::<_, String>(0)
        })
        .unwrap()
    };
    assert_eq!(identity(&first), identity(&second));
    assert_eq!(staged(&first), staged(&second));
    let snapshot = staged(&first);
    first.execute("DELETE FROM primary_candidate", []).unwrap();
    first.execute("INSERT INTO primary_candidate(event,admitted,changed,source_order,state,product,kind) VALUES('snapshot-event',4,4,0,'live',?1,'derived')",[pack(&Product::WorkSnapshot{snapshot}).unwrap()]).unwrap();
    let tx = first.transaction().unwrap();
    stage(&tx, 4, 4, true).unwrap();
    tx.commit().unwrap();
    assert_eq!(
        first
            .query_row("SELECT count(*) FROM primary_derived", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn publication_waits_without_losing_changes_when_authority_is_unavailable() {
    let mut db = database();
    fact(
        &db,
        "session-event",
        &Product::Session {
            reading: reading(),
            facts: Session::new(SessionId::parse("codex:morrow").unwrap(), SessionKind::Main),
        },
    );
    let tx = db.transaction().unwrap();
    stage(&tx, 1, 1, false).unwrap();
    tx.commit().unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM primary_derived", [], |row| row.get::<_, i64>(0)).unwrap(),
        0
    );
    db.execute("INSERT INTO primary_fact SELECT * FROM primary_candidate", []).unwrap();
    db.execute("DELETE FROM primary_candidate", []).unwrap();
    let tx = db.transaction().unwrap();
    stage(&tx, 2, 2, true).unwrap();
    tx.commit().unwrap();
    let snapshot = staged(&db);
    assert_eq!(snapshot.position, 2);
    assert_eq!(snapshot.session_id.to_string(), "codex:morrow");
    assert_eq!(
        db.query_row("SELECT count(*) FROM primary_work_dirty", [], |row| row.get::<_, i64>(0)).unwrap(),
        0
    );
}
