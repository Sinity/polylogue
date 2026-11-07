use super::*;
use crate::{query_objects, schema};
use polylogue_model::{BlobHash, ChangePosition};
use query_language::identity;
use serde_json::json;
use std::collections::BTreeMap;

fn old_object() -> Value {
    let form = json!({"source":{"kind":"unit","unit":"sessions","condition":null},"attach":[],"stages":[]});
    let id = identity::question_id_from_normal_form(&form, &BTreeMap::new()).unwrap();
    let mut ast = form.clone();
    ast["source"]["implicit"] = false.into();
    json!({"id":id,"normal_form":form,"parameters":{},"identity_ast":ast,"source_outputs":[],"definitions":[],"dependencies":[],"edges":[]})
}
fn independent_hash(domain: &[u8], value: &Value) -> BlobHash {
    fn ordered(value: &Value) -> Value {
        match value {
            Value::Object(fields) => {
                let fields: BTreeMap<_, _> = fields
                    .iter()
                    .map(|(key, value)| (key.clone(), ordered(value)))
                    .collect();
                serde_json::to_value(fields).unwrap()
            }
            Value::Array(values) => Value::Array(values.iter().map(ordered).collect()),
            value => value.clone(),
        }
    }
    let mut digest = blake3::Hasher::new();
    digest.update(domain);
    digest.update(&serde_json::to_vec(&ordered(value)).unwrap());
    BlobHash(*digest.finalize().as_bytes())
}
fn baseline() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    schema::ensure_archive(&conn).unwrap();
    conn.pragma_update(None, "user_version", 1).unwrap();
    conn
}
fn insert(conn: &Connection, source: &Value) {
    let id: query_language::identity::QuestionId =
        serde_json::from_value(source["id"].clone()).unwrap();
    conn.execute("INSERT INTO query(hash,normal_form,object,first_run) VALUES (?1,?2,?3,'retained-first-run')",params![id.to_string(),source["normal_form"].to_string(),encoding::pack_json(source).unwrap()]).unwrap();
}
#[test]
fn numbered_upgrade_preserves_question_run_completion_and_hash_custody() {
    let conn = baseline();
    let source = old_object();
    insert(&conn, &source);
    let current = query_objects::tests::question("sessions");
    let original_run = query_objects::tests::run(
        &current,
        "old-run",
        1000,
        ChangePosition {
            generation: "synthetic-generation".into(),
            index: 4,
            archive: 2,
        },
    );
    let mut run_source = serde_json::to_value(original_run).unwrap();
    run_source["question"] = source["id"].clone();
    run_source["binding"]["execution"]["replay"]["context"]["question"] = source["id"].clone();
    run_source["binding"]["execution"]["replay"]["bound_ast"] = source["identity_ast"].clone();
    let binding = independent_hash(
        b"polylogue-execution-binding-v1\0",
        &run_source["binding"]["execution"],
    );
    let record = independent_hash(b"polylogue-query-run-begin-v1\0", &run_source);
    let id = run_source["id"].as_str().unwrap();
    let question: identity::QuestionId = serde_json::from_value(source["id"].clone()).unwrap();
    conn.execute(
        "INSERT INTO query_run(id,binding_hash,run_hash,query,run) VALUES (?1,?2,?3,?4,?5)",
        params![
            id,
            binding.to_string(),
            record.to_string(),
            question.to_string(),
            encoding::pack_json(&run_source).unwrap()
        ],
    )
    .unwrap();
    let completion = encoding::pack_json(&json!({"run":id,"binding_hash":binding,"run_hash":record,"at":"1970-01-01T00:00:01Z","outcome":{"state":"ok"},"count":2,"exact":true,"next":null})).unwrap();
    conn.execute(
        "UPDATE query_run SET completion=?1 WHERE id=?2",
        params![completion, id],
    )
    .unwrap();
    schema::ensure_archive(&conn).unwrap();
    assert_eq!(schema::version(&conn).unwrap(), 3);
    let (bytes, first): (Vec<u8>, String) = conn
        .query_row("SELECT object,first_run FROM query", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    let object: QueryObject = encoding::unpack_json(&bytes).unwrap();
    assert_eq!(object.historical_source.as_ref().unwrap().record, source);
    assert_eq!(first, "retained-first-run");
    assert_eq!(object.id, question);
    let (bytes, kept_completion): (Vec<u8>, Vec<u8>) = conn
        .query_row("SELECT run,completion FROM query_run", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    let run: QueryRun = encoding::unpack_json(&bytes).unwrap();
    assert_eq!(run.historical_source.as_ref().unwrap().record, run_source);
    assert_eq!(run.binding_hash().unwrap(), binding);
    assert_eq!(run.record_hash().unwrap(), record);
    assert_eq!(kept_completion, completion);
    assert!(query_objects::known_run(&conn, &run).unwrap());
    let mut forged = run.clone();
    forged
        .binding
        .original_mut()
        .unwrap()
        .replay
        .original_text
        .push_str(" changed");
    assert!(forged.binding_hash().is_err());
    schema::ensure_archive(&conn).unwrap();
}
#[test]
fn malformed_second_record_rolls_back_first_conversion_and_version() {
    let conn = baseline();
    let source = old_object();
    insert(&conn, &source);
    let before: Vec<u8> = conn
        .query_row("SELECT object FROM query", [], |row| row.get(0))
        .unwrap();
    conn.execute(
        "INSERT INTO query(hash,normal_form,object) VALUES (?1,'{}',?2)",
        params![
            "f".repeat(64),
            encoding::pack_json(&json!({"identity_ast":{"stages":[]}})).unwrap()
        ],
    )
    .unwrap();
    assert!(schema::ensure_archive(&conn).is_err());
    assert_eq!(schema::version(&conn).unwrap(), 1);
    let after: Vec<u8> = conn
        .query_row(
            "SELECT object FROM query WHERE hash<>?1",
            ["f".repeat(64)],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(after, before);
    assert_eq!(
        conn.query_row("SELECT count(*) FROM query", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        2
    );
}

#[test]
fn unavailable_legacy_question_does_not_block_archive_open_or_unrelated_execution() {
    use polylogue_model::query_objects::{QueryUnavailable, QuestionBody, RunBinding};
    let conn = baseline();
    let ordinary = old_object();
    insert(&conn, &ordinary);
    let mut source = old_object();
    source["identity_ast"]["source"]["condition"] = json!({"span":{"start":0,"end":1},"kind":{"kind":"overlaps","from":{"kind":"scalar","text":"2026-01-01","quoted":false},"to":{"kind":"scalar","text":"2026-01-02","quoted":false}}});
    let form = polylogue_model::query_objects::evolution::legacy_definition_normal_form(
        &source,
        &polylogue_query::Bindings::default().catalog(),
        &[],
    )
    .unwrap();
    let id = identity::question_id_from_normal_form(&form, &BTreeMap::new()).unwrap();
    source["id"] = serde_json::to_value(id).unwrap();
    source["normal_form"] = form;
    insert(&conn, &source);
    let current = query_objects::tests::question("sessions");
    let run = query_objects::tests::run(
        &current,
        "old-unbuilt-attempt",
        1000,
        ChangePosition {
            generation: "synthetic-generation".into(),
            index: 4,
            archive: 2,
        },
    );
    let mut original_run = serde_json::to_value(run).unwrap();
    original_run["question"] = source["id"].clone();
    original_run["binding"]["execution"]["replay"]["bound_ast"] = source["identity_ast"].clone();
    original_run["binding"]["execution"]["replay"]["context"]["question"] = source["id"].clone();
    let binding = independent_hash(
        b"polylogue-execution-binding-v1\0",
        &original_run["binding"]["execution"],
    );
    let record = independent_hash(b"polylogue-query-run-begin-v1\0", &original_run);
    conn.execute(
        "INSERT INTO query_run(id,binding_hash,run_hash,query,run) VALUES (?1,?2,?3,?4,?5)",
        params![
            original_run["id"].as_str().unwrap(),
            binding.to_string(),
            record.to_string(),
            id.to_string(),
            encoding::pack_json(&original_run).unwrap()
        ],
    )
    .unwrap();
    schema::ensure_archive(&conn).unwrap();
    assert_eq!(schema::version(&conn).unwrap(), 3);
    let bytes: Vec<u8> = conn
        .query_row(
            "SELECT object FROM query WHERE hash=?1",
            [id.to_string()],
            |row| row.get(0),
        )
        .unwrap();
    let question: QueryObject = encoding::unpack_json(&bytes).unwrap();
    assert!(matches!(
        question.body,
        QuestionBody::Unavailable {
            reason: QueryUnavailable::SpanOverlap
        }
    ));
    assert_eq!(question.historical_source.as_ref().unwrap().record, source);
    let saved = polylogue_query::plan::saved_query_bindings(&question).unwrap();
    let bindings = polylogue_query::Bindings {
        saved,
        ..Default::default()
    };
    assert!(matches!(
        polylogue_query::plan::prepare(&format!("query:{id}"), &bindings, |_| Ok(())),
        Err(polylogue_query::QueryError::Unavailable { .. })
    ));
    polylogue_query::plan::prepare("sessions", &bindings, |_| Ok(())).unwrap();
    let bytes: Vec<u8> = conn
        .query_row("SELECT run FROM query_run", [], |row| row.get(0))
        .unwrap();
    let run: QueryRun = encoding::unpack_json(&bytes).unwrap();
    assert!(matches!(
        run.binding,
        RunBinding::Unavailable {
            reason: QueryUnavailable::SpanOverlap
        }
    ));
    assert!(run.binding.original().is_none());
    assert_eq!(run.binding_hash().unwrap(), binding);
    assert_eq!(run.record_hash().unwrap(), record);
    assert_eq!(run.historical_source.as_ref().unwrap().record, original_run);
}
