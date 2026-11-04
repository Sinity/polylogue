use super::super::*;
use polylogue_model::{
    BlobHash,
    historical::{AssertionEvidence, Integrity},
};
use rusqlite::params;
const REQUEST:&[u8]=br#"{"id":"legacy","machine":"sender","at":"1970-01-01T00:00:00.001Z","write":{"write":"add_note","note":{"kind":"note","value":"kept","targets":["repo:synthetic"]}}}"#;
const INTENT:&[u8]=br#"["sender",{"write":"add_note","note":{"kind":"note","value":"kept","targets":["repo:synthetic"]}}]"#;
const OUTCOME: &[u8] = br#"{"id":"legacy","repeat":false,"pending":false}"#;
const EFFECT:&[u8]=br#"{"notes":[{"note":{"id":"legacy","first_note":"legacy","kind":"note","key":null,"value":"kept","author":"operator","created_at":"1970-01-01T00:00:00.001Z","valid_until":null,"targets":["repo:synthetic"],"relations":[]},"machine":"sender","revises":[]}],"withdrawals":[],"settings":[],"variants":{"variants":[],"variant_parts":[]},"kept_results":[],"deliveries":[],"chats":[],"backfills":[]}"#;

fn predecessor() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    legacy_variant_time(&conn).unwrap();
    conn.execute_batch(include_str!("../fixtures/archive_2.sql"))
        .unwrap();
    for sql in [
        include_str!("../variant_3.sql"),
        include_str!("../archive_3.sql"),
        include_str!("../archive_4.sql"),
    ] {
        conn.execute_batch(&crate::source::Source::Ledger.sql(sql))
            .unwrap();
    }
    conn.pragma_update(None, "user_version", 4).unwrap();
    conn.execute("INSERT INTO note(id,first_note,kind,value,author,machine,created_at,commit_seq) VALUES('legacy','legacy','note','kept','operator','sender',1,3)",[]).unwrap();
    conn.execute(
        "INSERT INTO note_target(note_id,ref) VALUES('legacy','repo:synthetic')",
        [],
    )
    .unwrap();
    conn
}

fn insert_request(
    conn: &Connection,
    request: Option<&[u8]>,
    effect: &[u8],
) -> (Option<Vec<u8>>, Vec<u8>, Vec<u8>) {
    let request = request
        .map(crate::encoding::pack_bytes)
        .transpose()
        .unwrap();
    let outcome = crate::encoding::pack_bytes(OUTCOME).unwrap();
    let effect = crate::encoding::pack_bytes(effect).unwrap();
    conn.execute("INSERT INTO client_request(id,content_hash,created_at,request,outcome,effect,imports,commit_seq) VALUES('legacy',?1,1,?2,?3,?4,?5,3)",params![BlobHash::of(INTENT).0.as_slice(),request,outcome,effect,r#"{"observed":[{"machine":"receiver","input":"source","chain":1,"actor":"operator","at":"1970-01-01T00:00:00.001Z"}],"declared":[]}"#]).unwrap();
    (request, outcome, effect)
}

#[test]
fn source_five_conserves_original_accepted_bytes_and_typed_historical_facts() {
    let conn = predecessor();
    let before = insert_request(&conn, Some(REQUEST), EFFECT);
    ensure_archive(&conn).unwrap();
    ensure_archive(&conn).unwrap();
    assert_eq!(version(&conn).unwrap(), 5);
    assert!(same_schema(&conn, &[ARCHIVE]).unwrap());
    let record = crate::primary::authored::historical::get(&conn, "legacy")
        .unwrap()
        .unwrap();
    assert_eq!(record.validate().unwrap(), Integrity::HashMatches);
    assert_eq!(
        record.authorship, None,
        "recorded operator on the fact is not invented Invocation evidence"
    );
    let original = record.original.as_ref().unwrap();
    assert_eq!(original.request.as_deref(), Some(REQUEST));
    assert_eq!(original.effect.as_deref(), Some(EFFECT));
    assert_eq!(original.imports.as_ref(), Some(&record.imports));
    let after = conn
        .query_row(
            "SELECT request,outcome,effect FROM client_request WHERE id='legacy'",
            [],
            |row| {
                Ok((
                    row.get::<_, Option<Vec<u8>>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(
        after, before,
        "existing historical compressed columns remain unchanged"
    );
    let Some(AssertionEvidence::Historical {
        fact, commit_seq, ..
    }) = crate::assertions::get_evidence(&conn, "legacy").unwrap()
    else {
        panic!("typed historical fact missing")
    };
    assert_eq!(fact.value.as_deref(), Some("kept"));
    assert_eq!(fact.targets, ["repo:synthetic"]);
    assert_eq!(commit_seq, 3);
    assert_eq!(
        record.facts,
        [polylogue_model::Reference::parse("assertion:legacy@legacy").unwrap()]
    );
    let bad: i64 = conn
        .query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(bad, 0);
}

#[test]
fn source_five_missing_original_body_retains_known_outcome_and_effect_without_verified_intent() {
    let conn = predecessor();
    insert_request(&conn, None, EFFECT);
    ensure_archive(&conn).unwrap();
    let record = crate::primary::authored::historical::get(&conn, "legacy")
        .unwrap()
        .unwrap();
    assert_eq!(record.validate().unwrap(), Integrity::Unrecorded);
    assert_eq!(record.intent, None);
    assert_eq!(record.machine, None);
    let original = record.original.as_ref().unwrap();
    assert_eq!(original.request, None);
    assert_eq!(original.intent, None);
    assert_eq!(original.outcome.as_deref(), Some(OUTCOME));
    assert_eq!(original.effect.as_deref(), Some(EFFECT));
    assert_eq!(record.content_hash, BlobHash::of(INTENT));
}

#[test]
fn source_five_unknown_nonempty_effect_rolls_back_schema_rows_and_version() {
    let conn = predecessor();
    let effect = br#"{"unmapped_effect":[{"id":"must-remain"}]}"#;
    let before = insert_request(&conn, Some(REQUEST), effect);
    assert!(ensure_archive(&conn).is_err());
    assert_eq!(version(&conn).unwrap(), 4);
    assert_eq!(
        conn.query_row("SELECT value FROM note WHERE id='legacy'", [], |row| row
            .get::<_, String>(
            0
        ))
        .unwrap(),
        "kept"
    );
    let after = conn
        .query_row(
            "SELECT request,outcome,effect FROM client_request WHERE id='legacy'",
            [],
            |row| {
                Ok((
                    row.get::<_, Option<Vec<u8>>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(before, after);
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM sqlite_schema WHERE name='assertion'",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
}

#[test]
fn source_five_keeps_frozen_submillisecond_content_order_and_original_fact_admission() {
    // Predecessor SQL used Timestamp::unix_ms (whole milliseconds truncated
    // toward zero). Exact submillisecond times remain in the frozen JSON and
    // historical fact; the integer projection is never the precision authority.
    for (at, millis) in [
        ("1970-01-01T00:00:00.001123456Z", 1_i64),
        ("1969-12-31T23:59:59.999123456Z", 0),
        ("1969-12-31T23:59:59.998123456Z", -1),
    ] {
        let conn = predecessor();
        let request = String::from_utf8(REQUEST.to_vec())
            .unwrap()
            .replace("1970-01-01T00:00:00.001Z", at)
            .replace("[\"repo:synthetic\"]", "[\"repo:synthetic\",\"repo:aaa\"]");
        let effect = String::from_utf8(EFFECT.to_vec())
            .unwrap()
            .replace("1970-01-01T00:00:00.001Z", at)
            .replace("[\"repo:synthetic\"]", "[\"repo:synthetic\",\"repo:aaa\"]");
        insert_request(&conn, Some(request.as_bytes()), effect.as_bytes());
        let intent = br#"["sender",{"write":"add_note","note":{"kind":"note","value":"kept","targets":["repo:synthetic","repo:aaa"]}}]"#;
        conn.execute(
            "UPDATE client_request SET content_hash=?1,created_at=?2 WHERE id='legacy'",
            params![BlobHash::of(intent).0.as_slice(), millis],
        )
        .unwrap();
        conn.execute(
            "UPDATE note SET created_at=?1,commit_seq=1 WHERE id='legacy'",
            [millis],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO note_target(note_id,ref) VALUES('legacy','repo:aaa')",
            [],
        )
        .unwrap();
        ensure_archive(&conn).unwrap();
        let Some(AssertionEvidence::Historical {
            fact, commit_seq, ..
        }) = crate::assertions::get_evidence(&conn, "legacy").unwrap()
        else {
            panic!("original fact missing")
        };
        assert_eq!(fact.created_at.unwrap().to_string(), at);
        assert_eq!(fact.targets, ["repo:synthetic", "repo:aaa"]);
        assert_eq!(
            commit_seq, 1,
            "whole request admission is not the earlier fact admission"
        );
        let request = crate::primary::authored::historical::get(&conn, "legacy")
            .unwrap()
            .unwrap();
        assert_eq!(request.at.unwrap().to_string(), at);
        assert_eq!(request.original.unwrap().effect.unwrap(), effect.as_bytes());
    }
}

#[test]
fn source_five_fresh_and_upgrade_keep_old_run_bytes_and_admit_raw_object_bindings() {
    use polylogue_model::query_objects::{
        ExecutionPresentation, ObjectExecutionBinding, QueryRun, RunBinding,
    };
    let historical = crate::fixture::query_objects::synthetic_source3_run().unwrap();
    let old: QueryRun = crate::encoding::read_json(&historical.encoded_run).unwrap();
    for upgrade in [false, true] {
        let conn = if upgrade {
            predecessor()
        } else {
            Connection::open_in_memory().unwrap()
        };
        let packed = crate::encoding::pack_bytes(&historical.encoded_run).unwrap();
        if upgrade {
            conn.execute("INSERT INTO query_run(id,binding_hash,run_hash,query,commit_seq,source_order,run) VALUES(?1,?2,?3,?4,3,1,?5)",
                params![old.id.as_str(),old.binding_hash().unwrap().to_string(),old.record_hash().unwrap().to_string(),old.question.unwrap().to_string(),packed]).unwrap();
        }
        ensure_archive(&conn).unwrap();
        assert_eq!(version(&conn).unwrap(), 5);
        if upgrade {
            let after: Vec<u8> = conn
                .query_row(
                    "SELECT run FROM query_run WHERE id=?1",
                    [old.id.as_str()],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(
                after, packed,
                "upgrade must not rewrite the recorded old execution"
            );
            let (retained, _) = crate::query_objects::historical_run_on(&conn, &old.id)
                .unwrap()
                .unwrap();
            assert_eq!(retained.encoded_run, historical.encoded_run);
        }
        let reference = polylogue_model::Reference::Run(old.id.clone());
        let mut raw = old.clone();
        raw.id = polylogue_model::RunId::parse("00000000-0000-4000-8000-000000000004").unwrap();
        raw.question = None;
        raw.binding = RunBinding::OriginalObject {
            execution: Box::new(ObjectExecutionBinding {
                reference: reference.clone(),
                content_hash: BlobHash::of(&historical.encoded_run),
                scope_fingerprint: None,
                versions: Default::default(),
                accepted_request: None,
                presentation: ExecutionPresentation::Read {
                    options: polylogue_model::read::ReadOptions {
                        view: polylogue_model::read::View::Raw,
                        ..Default::default()
                    },
                    delivery: polylogue_model::read::ReadDelivery::Stream,
                    format: polylogue_model::read::Format::Jsonl,
                    page_budget: None,
                    direct_source: None,
                },
            }),
        };
        crate::query_objects::apply_run(&conn, &raw, 4, 2).unwrap();
        let row: (Option<String>, String) = conn
            .query_row(
                "SELECT query,object_ref FROM query_run WHERE id=?1",
                [raw.id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(row, (None, reference.to_string()));
        let mut invalid = raw.clone();
        if let RunBinding::OriginalObject { execution } = &mut invalid.binding
            && let ExecutionPresentation::Read { options, .. } = &mut execution.presentation
        {
            options.view = polylogue_model::read::View::Transcript;
        }
        assert!(
            crate::query_objects::apply_run(&conn, &invalid, 5, 3).is_err(),
            "a non-Raw Run object must still refuse"
        );
        assert_eq!(
            conn.query_row(
                "SELECT object_ref FROM query_run WHERE id=?1",
                [raw.id.as_str()],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
            reference.to_string()
        );
    }
}
