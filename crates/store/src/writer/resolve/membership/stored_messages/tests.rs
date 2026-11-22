use super::*;
use crate::archive::{cache, open_write};
use rusqlite::{functions::FunctionFlags, params};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

fn schema(conn: &Connection) {
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    crate::sql_functions::register(conn).unwrap();
    crate::schema::create_index(conn).unwrap();
    conn.execute_batch(
        "INSERT INTO input_ref(pk,id) VALUES(1,'first'),(2,'second'),(3,'unrelated');
        INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES
        (1,'test:main','test','main','main',1),(2,'test:copy','test','copy','compaction',1);",
    )
    .unwrap();
}

fn hash(id: i64) -> Vec<u8> {
    let mut hash = vec![0; 32];
    hash[..8].copy_from_slice(&id.to_be_bytes());
    hash
}

fn version(conn: &Connection, id: i64, layout: Option<&[u8]>) {
    conn.execute(
        "INSERT INTO message_version(pk,version_hash) VALUES(?1,?2)",
        params![id, hash(id)],
    )
    .unwrap();
    // Metadata reads must not fetch or decode a body page.
    conn.execute("INSERT INTO message_version_body VALUES(?1,x'00')", [id])
        .unwrap();
    if let Some(layout) = layout {
        conn.execute(
            "INSERT INTO message_block_selection(version_pk,layout) VALUES(?1,?2)",
            params![id, layout],
        )
        .unwrap();
    }
}

fn message(conn: &Connection, id: i64, input: i64, chain: Option<i64>) {
    conn.execute(
        "INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk,source_chain)
        VALUES(?1,?2,1,?2,'assistant','model',?3,?4)",
        params![id, format!("m{id}"), input, chain],
    )
    .unwrap();
}

fn source(conn: &Connection, message: i64, input: i64, chain: i64, version: i64) {
    conn.execute(
        "INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,
        id_kind,session_pk,main,route,seq,record_position,order_time,at,completion,version_pk)
        VALUES(?1,?2,?3,11,29,'native',1,1,'live',17,2,41,37,'finished',?4)",
        params![message, input, chain, version],
    )
    .unwrap();
}

#[test]
fn stored_readings_preserve_all_chains_routes_layouts_and_missing_facts() {
    let conn = Connection::open_in_memory().unwrap();
    schema(&conn);
    version(&conn, 101, Some(&[1, 2, 3]));
    version(&conn, 102, None);
    message(&conn, 10, 2, Some(7));
    message(&conn, 20, 1, None);
    source(&conn, 10, 1, 1, 101);
    source(&conn, 10, 1, 2, 102);
    source(&conn, 10, 2, 7, 102);
    source(&conn, 20, 1, 1, 101);
    conn.execute_batch(
        "INSERT INTO session_record(session_pk,input_pk,chain,facts) VALUES
        (1,1,1,'{\"kind\":\"main\"}'),(1,1,2,'{\"kind\":\"compaction\"}');
        INSERT INTO message_route VALUES(10,'time',1,1),(10,'model',1,2);",
    )
    .unwrap();
    let mut layouts = BTreeMap::new();
    let stored = load(&conn, &[20, 10, 999, 10], &mut layouts).unwrap();
    assert_eq!(stored.len(), 3);
    let readings = &stored[&10].members;
    let mut kinds = BTreeMap::new();
    assert_eq!(
        readings.len(),
        3,
        "duplicate requested keys cannot duplicate claims"
    );
    assert_eq!(
        readings
            .iter()
            .map(|member| (
                member.input_id.as_str(),
                member.chain,
                super::super::explicit_session(&conn, &mut kinds, member).unwrap()
            ))
            .collect::<Vec<_>>(),
        [
            ("first", 1, true),
            ("first", 2, false),
            ("second", 7, false)
        ]
    );
    for member in readings {
        assert_eq!(
            (member.start, member.end, member.seq, member.position),
            (11, 29, 17, 2)
        );
        assert_eq!((member.order_time, member.at), (Some(41), Some(37)));
        assert_eq!(member.session, 1);
        assert!(member.main);
        assert!(!member.content);
        assert_eq!(member.completion, polylogue_model::Completion::Finished);
        assert_eq!(member.route, polylogue_model::Route::Live);
    }
    let shown = stored[&10].shown.as_ref().unwrap();
    assert_eq!(shown.hash, hash(102));
    assert_eq!(shown.session, 1);
    assert_eq!(shown.layout, None);
    let mut expected = vec![
        (FieldGroup::Time, 1, 1, hash(101)),
        (FieldGroup::Model, 1, 2, hash(102)),
    ];
    expected.sort();
    assert_eq!(shown.routes, expected);
    assert!(stored[&20].shown.is_none(), "no selected source chain");
    assert_eq!(stored[&20].members.len(), 1);
    assert!(stored[&999].members.is_empty());
    assert!(stored[&999].shown.is_none());
    assert_eq!(layouts[&hash(101)], Some(vec![1, 2, 3]));
    assert_eq!(layouts[&hash(102)], None);
    assert_eq!(
        conn.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        0
    );
}

#[derive(Default)]
struct Reads {
    versions: Arc<AtomicU64>,
    facts: Arc<AtomicU64>,
    fact_bytes: Arc<AtomicU64>,
}

fn observe(conn: &Connection) -> Reads {
    let reads = Reads::default();
    let versions = reads.versions.clone();
    conn.create_scalar_function(
        "observed_version",
        1,
        FunctionFlags::SQLITE_UTF8,
        move |context| {
            let hash: Vec<u8> = context.get(0)?;
            versions.fetch_add(1, Ordering::Relaxed);
            Ok(hash)
        },
    )
    .unwrap();
    let facts = reads.facts.clone();
    let bytes = reads.fact_bytes.clone();
    conn.create_scalar_function(
        "observed_facts",
        1,
        FunctionFlags::SQLITE_UTF8,
        move |context| {
            let value: String = context.get(0)?;
            facts.fetch_add(1, Ordering::Relaxed);
            bytes.fetch_add(value.len() as u64, Ordering::Relaxed);
            Ok(value)
        },
    )
    .unwrap();
    conn.execute_batch("ALTER TABLE message_version RENAME TO stored_version;
        CREATE VIEW message_version AS SELECT pk,observed_version(version_hash) AS version_hash FROM stored_version;
        ALTER TABLE session_record RENAME TO stored_record;
        CREATE VIEW session_record AS SELECT session_pk,input_pk,chain,observed_facts(facts) AS facts,committed FROM stored_record;").unwrap();
    reads
}

type ReadingFact = (i64, i64, i64, Vec<u8>, bool);

fn measured(conn: &Connection, pks: &[i64]) -> (Vec<ReadingFact>, u64) {
    let count = Arc::new(AtomicU64::new(0));
    let seen = count.clone();
    conn.progress_handler(
        1,
        Some(move || {
            seen.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    let result = {
        let mut all = Vec::new();
        let mut layouts = BTreeMap::new();
        let mut kinds = BTreeMap::new();
        for pks in pks.chunks(BATCH) {
            for (pk, stored) in load(conn, pks, &mut layouts).unwrap() {
                for member in stored.members {
                    let explicit =
                        super::super::explicit_session(conn, &mut kinds, &member).unwrap();
                    all.push((pk, member.input, member.chain, member.hash, explicit));
                }
            }
        }
        all
    };
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    (result, count.load(Ordering::Relaxed))
}

#[test]
fn requested_shared_facts_are_read_once_per_snapshot_not_per_claim() {
    const MESSAGES: i64 = 1024;
    let conn = Connection::open_in_memory().unwrap();
    schema(&conn);
    let facts = serde_json::json!({"kind":"main","instructions":"x".repeat(64 << 10)}).to_string();
    for input in [1, 2] {
        conn.execute(
            "INSERT INTO session_record(session_pk,input_pk,chain,facts) VALUES(1,?1,1,?2)",
            params![input, facts],
        )
        .unwrap();
    }
    for id in 1..=MESSAGES {
        version(&conn, id, None);
        message(&conn, id, 1, Some(1));
        source(&conn, id, 1, 1, id);
        source(&conn, id, 2, 1, id);
    }
    let reads = observe(&conn);
    let pks: Vec<_> = (1..=MESSAGES).collect();
    let (actual, vm) = measured(&conn, &pks);
    let version_reads = reads.versions.swap(0, Ordering::Relaxed);
    let fact_reads = reads.facts.swap(0, Ordering::Relaxed);
    let fact_bytes = reads.fact_bytes.swap(0, Ordering::Relaxed);
    assert_eq!(actual.len(), (MESSAGES * 2) as usize);
    assert_eq!(version_reads, MESSAGES as u64);
    assert_eq!(
        fact_reads, 2,
        "each needed source-session fact belongs to the whole snapshot"
    );
    assert_eq!(fact_bytes, facts.len() as u64 * fact_reads);
    // Independent correlated SQL is the semantic oracle and a positive
    // control for the observed column reads of the former per-claim shape.
    let expected: Vec<ReadingFact> = conn
        .prepare(
            "SELECT s.message_pk,s.input_pk,s.chain,
         (SELECT version_hash FROM message_version WHERE pk=s.version_pk),
         coalesce((SELECT json_extract(facts,'$.kind')<>'compaction' FROM session_record f
         WHERE f.session_pk=s.session_pk AND f.input_pk=s.input_pk AND f.chain=s.chain),0)
         FROM message_source s ORDER BY s.message_pk,s.input_pk,s.chain",
        )
        .unwrap()
        .query_map([], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(
        reads.versions.swap(0, Ordering::Relaxed),
        (MESSAGES * 2) as u64
    );
    assert_eq!(
        reads.facts.swap(0, Ordering::Relaxed),
        (MESSAGES * 2) as u64
    );
    let point_fact_bytes = reads.fact_bytes.swap(0, Ordering::Relaxed);
    assert_eq!(point_fact_bytes, facts.len() as u64 * (MESSAGES * 2) as u64);
    // Unrelated history must not add scanning, payload access or JSON work.
    conn.execute_batch(
        "DROP VIEW message_version; ALTER TABLE stored_version RENAME TO message_version;
        DROP VIEW session_record; ALTER TABLE stored_record RENAME TO session_record;",
    )
    .unwrap();
    conn.execute("INSERT INTO session_record(session_pk,input_pk,chain,facts) VALUES(2,3,1,'invalid unrelated JSON')", []).unwrap();
    for id in 10_000..18_192 {
        version(&conn, id, None);
        message(&conn, id, 3, Some(1));
        source(&conn, id, 3, 1, id);
        conn.execute(
            "UPDATE message_source SET session_pk=2 WHERE message_pk=?1",
            [id],
        )
        .unwrap();
    }
    let reads = observe(&conn);
    let (with_history, history_vm) = measured(&conn, &pks);
    assert_eq!(actual, with_history);
    assert!(
        history_vm <= vm + 1000,
        "unrelated rows changed native work: {vm}->{history_vm}"
    );
    assert_eq!(reads.facts.load(Ordering::Relaxed), 2);
    assert_eq!(reads.versions.load(Ordering::Relaxed), MESSAGES as u64);
    eprintln!(
        "MEMBERSHIP_SNAPSHOT_WORK {}",
        serde_json::json!({
            "messages":MESSAGES,"claims":actual.len(),"groups":2,"batch_vm":vm,"unrelated_history_vm":history_vm,
            "batch_version_values":version_reads,"batch_session_facts":fact_reads,"batch_fact_bytes":fact_bytes,
            "correlated_fact_bytes":point_fact_bytes,
            "note":"Logical column reads and SQLite VM work, not physical bytes or end-to-end throughput."
        })
    );
}

#[test]
fn all_snapshot_queries_keep_the_callers_existing_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let conn = open_write(&path, cache::SETUP).unwrap();
    schema(&conn);
    version(&conn, 1, Some(&[1]));
    message(&conn, 1, 1, Some(1));
    source(&conn, 1, 1, 1, 1);
    conn.execute("INSERT INTO session_record(session_pk,input_pk,chain,facts) VALUES(1,1,1,'{\"kind\":\"main\"}')", []).unwrap();
    let read = crate::archive::open_read(&path).unwrap();
    read.execute_batch("BEGIN").unwrap();
    let mut layouts = BTreeMap::new();
    let before = load(&read, &[1], &mut layouts).unwrap();
    conn.execute(
        "UPDATE session_record SET facts='{\"kind\":\"compaction\"}'",
        [],
    )
    .unwrap();
    conn.execute("UPDATE message_block_selection SET layout=x'0203'", [])
        .unwrap();
    let held = load(&read, &[1], &mut layouts).unwrap();
    assert_eq!(before[&1].shown.as_ref().unwrap().layout, Some(vec![1]));
    assert_eq!(held[&1].shown.as_ref().unwrap().layout, Some(vec![1]));
    assert!(
        super::super::explicit_session(&read, &mut BTreeMap::new(), &held[&1].members[0]).unwrap()
    );
    read.execute_batch("COMMIT").unwrap();
    let current = load(&read, &[1], &mut layouts).unwrap();
    assert_eq!(current[&1].shown.as_ref().unwrap().layout, Some(vec![2, 3]));
    assert!(
        !super::super::explicit_session(&read, &mut BTreeMap::new(), &current[&1].members[0])
            .unwrap()
    );
}
