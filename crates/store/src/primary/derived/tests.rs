use super::*;
use crate::InputKey;
use crate::primary::Reading;
use polylogue_model::{MessageRow, SessionKind};
use serde_json::json;

fn proof(position: i64) -> crate::primary::RawInputProof {
    crate::primary::RawInputProof {
        stamp: "owner-write-proof".into(),
        position,
        policy_epoch: 1,
        scope: "synthetic-scope".into(),
    }
}

fn database() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch(crate::schema::INDEX).unwrap();
    db.execute_batch(super::SCHEMA).unwrap();
    db.execute(
        "INSERT INTO meta(key,value) VALUES ('generation','synthetic-generation')",
        [],
    )
    .unwrap();
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
    let reference = match value {
        Product::Session { facts, .. } => Some(format!("session:{}", facts.id)),
        Product::Link { facts, .. } => Some(facts.selection_tag()),
        _ => None,
    };
    let native = crate::primary::NativeFactContext {
        kind: crate::primary::NativeKind {
            name: match value {
                Product::Link { .. } => "polylogue.link",
                Product::Message { .. } => "polylogue.message",
                Product::Event { .. } => "polylogue.event",
                _ => "polylogue.session",
            }
            .into(),
            version: if matches!(value, Product::Message { .. }) {
                4
            } else {
                1
            },
        },
        kind_definition_hash: "a".repeat(64),
        projection: crate::primary::ProjectionIdentity {
            policy_epoch: 1,
            payload_hash: "b".repeat(64),
        },
        origin: crate::primary::ConfirmedOrigin {
            producer_family: Some("polylogue".into()),
            source: Some(crate::primary::NativeSource {
                name: "polylogue".into(),
                version: 2,
            }),
            source_fingerprint: Some("c".repeat(64)),
            producer_identity: Some("synthetic-producer".into()),
        },
    };
    db.execute("INSERT INTO primary_candidate(event,admitted,changed,source_order,state,product,input,chain,reference,kind,step_start,step_end,reading_version,session,native_context) VALUES(?1,1,1,0,'live',?2,'input-x',1,?3,'parsed',0,10,'reader-1',?5,?4)",params![event,pack(value).unwrap(),reference,pack(&native).unwrap(),session(value).map(ToString::to_string)]).unwrap();
}
fn invocation(id: &str, call: &str) -> Product {
    Product::Message{reading:reading(),order:None,projection:None,native_fields:Vec::new(),declared_injections:Vec::new(),declared_attachment_bindings:Vec::new(),facts:Box::new(MessageRow{offset:0,end:10,message:serde_json::from_value(json!({"id":id,"session":"codex:morrow","role":"assistant","stop_reason":"end_turn","blocks":[{"type":"tool_call","call_id":call,"tool":"shell","action":"shell","input":{},"command":"true"}]})).unwrap()})}
}

fn show(db: &Connection, value: &Product, active: bool) {
    let Product::Message { facts, .. } = value else {
        panic!()
    };
    let prepared = crate::encoding::PreparedMessage::new(&facts.message).unwrap();
    db.execute(
        "INSERT INTO message_version(version_hash,body) VALUES(?1,?2)",
        params![prepared.hash.as_slice(), prepared.body],
    )
    .unwrap();
    db.execute("INSERT INTO message(pk,id,session_pk,order_input_pk,order_input_id,seq,role,written_by,input_pk,source_chain,source_start,source_end,on_active_path) VALUES(1,?1,1,1,'input-x',0,'assistant','model',1,1,0,10,?2)",params![facts.message.id.to_string(),active]).unwrap();
    db.execute("INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,shown,session_pk,main,version_hash) VALUES(1,1,1,0,10,1,1,1,?1)",[&hash_message(&facts.message).unwrap()[..]]).unwrap();
}
fn staged(db: &Connection) -> WorkSnapshot {
    let body:Vec<u8>=db.query_row("SELECT body FROM primary_derived WHERE cancelled_at IS NULL ORDER BY rowid DESC LIMIT 1",[],|row|row.get(0)).unwrap();
    let writes: Vec<ProductWrite> = crate::encoding::unpack_json(&body).unwrap();
    writes
        .into_iter()
        .find_map(|write| match write.product {
            Product::WorkSnapshot { snapshot } => Some(snapshot),
            _ => None,
        })
        .unwrap()
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
    stage(&tx, 1, 1, Some(&proof(1))).unwrap();
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
    stage(&tx, 2, 2, Some(&proof(2))).unwrap();
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
        stage(&tx, 3, 3, Some(&proof(3))).unwrap();
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
    let original_intent = identity(&first);
    // The applied source facts remain present when their canonical output arrives.
    first
        .execute(
            "INSERT INTO primary_fact SELECT * FROM primary_candidate",
            [],
        )
        .unwrap();
    first.execute("DELETE FROM primary_candidate", []).unwrap();
    first.execute("INSERT INTO primary_candidate(event,admitted,changed,source_order,state,product,kind) VALUES('snapshot-event',4,4,0,'live',?1,'derived')",[pack(&Product::WorkSnapshot{snapshot}).unwrap()]).unwrap();
    let tx = first.transaction().unwrap();
    stage(&tx, 4, 4, Some(&proof(4))).unwrap();
    tx.commit().unwrap();
    // Matching payload bytes without owner context cannot discharge custody.
    assert_eq!(identity(&first), original_intent);
    assert_eq!(
        first
            .query_row(
                "SELECT count(*) FROM primary_derived WHERE cancelled_at IS NULL",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
    let native = crate::primary::NativeFactContext {
        kind: crate::primary::NativeKind {
            name: "polylogue.work_snapshot".into(),
            version: 5,
        },
        kind_definition_hash: "a".repeat(64),
        projection: crate::primary::ProjectionIdentity {
            policy_epoch: 1,
            payload_hash: "b".repeat(64),
        },
        origin: crate::primary::ConfirmedOrigin {
            producer_family: Some("polylogue".into()),
            producer_identity: Some("synthetic-producer".into()),
            source: None,
            source_fingerprint: None,
        },
    };
    first
        .execute(
            "UPDATE primary_candidate SET native_context=?1 WHERE event='snapshot-event'",
            [pack(&native).unwrap()],
        )
        .unwrap();
    let tx = first.transaction().unwrap();
    stage(&tx, 4, 4, Some(&proof(4))).unwrap();
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
    stage(&tx, 1, 1, None).unwrap();
    tx.commit().unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM primary_derived", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    db.execute(
        "INSERT INTO primary_fact SELECT * FROM primary_candidate",
        [],
    )
    .unwrap();
    db.execute("DELETE FROM primary_candidate", []).unwrap();
    let tx = db.transaction().unwrap();
    stage(&tx, 2, 2, Some(&proof(2))).unwrap();
    tx.commit().unwrap();
    let snapshot = staged(&db);
    assert_eq!(snapshot.position, 2);
    assert_eq!(snapshot.session_id.to_string(), "codex:morrow");
    assert_eq!(
        db.query_row("SELECT count(*) FROM primary_work_dirty", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn canonical_custody_survives_restart_without_proof_and_rolls_back_proof_attachment() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.db");
    let mut conn = Connection::open(&path).unwrap();
    conn.execute_batch(SCHEMA).unwrap();
    let snapshot = WorkSnapshot {
        reducer_version: work_products::VERSION,
        position: 7,
        session_id: SessionId::parse("codex:retained").unwrap(),
        membership: work_products::SnapshotMembership::Retired {
            basis: work_products::component::MembershipBasis {
                membership_version: work_products::membership::VERSION,
                selection_version: work_products::SelectionVersion::SUPPORTED,
                completed_position: 7,
                scope: "synthetic-scope".into(),
                membership_hash: "prior-native-membership".into(),
            },
        },
        session: None,
        retired: true,
        unavailable: Vec::new(),
        messages: Vec::new(),
        calls: Vec::new(),
        turns: Vec::new(),
        sources: Vec::new(),
        native: Vec::new(),
    };
    let writes = vec![ProductWrite {
        product: Product::WorkSnapshot { snapshot },
        location: None,
        parents: Vec::new(),
        machine: None,
        output_index: 0,
    }];
    let tx = conn.transaction().unwrap();
    stage_products(&tx, "batch", &writes, None).unwrap();
    tx.commit().unwrap();
    drop(conn);
    let mut conn = Connection::open(&path).unwrap();
    let frozen: Vec<u8> = conn
        .query_row(
            "SELECT body FROM primary_derived WHERE identity='batch'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    {
        let tx = conn.transaction().unwrap();
        stage_products(&tx, "batch", &writes, Some(&proof(7))).unwrap();
        tx.rollback().unwrap();
    }
    let stored: Option<Vec<u8>> = conn
        .query_row(
            "SELECT proof FROM primary_derived WHERE identity='batch'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(stored.is_none());
    let tx = conn.transaction().unwrap();
    stage_products(&tx, "batch", &writes, Some(&proof(7))).unwrap();
    tx.commit().unwrap();
    let mut refreshed = proof(7);
    refreshed.stamp = "refreshed-owner-stamp".into();
    let tx = conn.transaction().unwrap();
    stage_products(&tx, "batch", &writes, Some(&refreshed)).unwrap();
    tx.commit().unwrap();
    let (body, stored): (Vec<u8>, Vec<u8>) = conn
        .query_row(
            "SELECT body,proof FROM primary_derived WHERE identity='batch'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(body, frozen);
    assert_eq!(
        crate::encoding::unpack_json::<crate::primary::RawInputProof>(&stored).unwrap(),
        proof(7)
    );
    let mut changed = writes.clone();
    changed[0].output_index = 1;
    let tx = conn.transaction().unwrap();
    assert!(matches!(
        stage_products(&tx, "batch", &changed, None),
        Err(StoreError::Conflict(_))
    ));
    tx.rollback().unwrap();
}

fn spawn_copy_graph() -> (Connection, SessionId, SessionId) {
    let db = database();
    let root = SessionId::parse("a:root").unwrap();
    let child = SessionId::parse("a-b:child").unwrap();
    for (event, id) in [("root-fact", root.clone()), ("child-fact", child.clone())] {
        fact(
            &db,
            event,
            &Product::Session {
                reading: reading(),
                facts: Session::new(id, SessionKind::Main),
            },
        );
    }
    let link = polylogue_model::Link {
        kind: polylogue_model::LinkKind::Spawn,
        parent: root.clone(),
        child: child.clone(),
        at: None,
        call_id: None,
        evidence: "explicit source parent".into(),
    };
    fact(
        &db,
        "spawn-fact",
        &Product::Link {
            reading: reading(),
            facts: link.clone(),
        },
    );
    db.execute("INSERT INTO link(pk,parent_id,child_id,kind,how,evidence,input_pk,source_chain) VALUES (1,?1,?2,'subagent','copied','explicit source parent',1,1)",params![root.to_string(),child.to_string()]).unwrap();
    db.execute(
        "INSERT INTO link_value(pk,hash,body) VALUES (1,zeroblob(32),?1)",
        [pack(&link).unwrap()],
    )
    .unwrap();
    db.execute("INSERT INTO link_source(link_pk,input_pk,chain,value_pk,arrival_rank,admitted) VALUES (1,1,1,1,0,1)",[]).unwrap();
    (db, root, child)
}

#[test]
fn explicit_spawn_copy_keeps_its_parent_as_root_and_canonicalizes_physical_refs() {
    let (mut db, root, child) = spawn_copy_graph();
    let tx = db.transaction().unwrap();
    stage(&tx, 1, 1, Some(&proof(1))).unwrap();
    tx.commit().unwrap();
    let snapshot = staged(&db);
    assert_eq!(snapshot.session_id, root);
    let work_products::SnapshotMembership::Established {
        work,
        physical_refs,
        ..
    } = snapshot.membership
    else {
        panic!("explicit source links establish the root");
    };
    assert_eq!(work.links.len(), 1);
    assert_eq!(work.links[0].event, "spawn-fact");
    assert_eq!(
        physical_refs
            .iter()
            .map(|reference| reference.session.to_string())
            .collect::<Vec<_>>(),
        vec![child.to_string(), root.to_string()]
    );
}

#[test]
fn unmatched_source_link_does_not_publish_disconnected_complete_roots() {
    for mismatch in ["chain", "reference"] {
        let (mut db, _, _) = spawn_copy_graph();
        match mismatch {
            "chain" => db.execute("UPDATE link_source SET chain=2", []).unwrap(),
            _ => {
                db.execute("UPDATE link SET how='fresh'", []).unwrap();
                db.execute(
                    "UPDATE primary_candidate SET reference='unrelated' WHERE event='spawn-fact'",
                    [],
                )
                .unwrap()
            }
        };
        let tx = db.transaction().unwrap();
        stage(&tx, 1, 1, Some(&proof(1))).unwrap();
        tx.commit().unwrap();
        assert_eq!(
            db.query_row("SELECT count(*) FROM primary_derived", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            0,
            "{mismatch}"
        );
        assert!(
            db.query_row(
                "SELECT EXISTS(SELECT 1 FROM primary_work_dirty)",
                [],
                |row| row.get::<_, bool>(0)
            )
            .unwrap(),
            "{mismatch}"
        );
    }
}
