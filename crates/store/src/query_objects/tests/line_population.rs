use super::*;
use crate::query_objects::line_population::{convert_preserved, convert_run};
use polylogue_model::{BlobHash, Reference};
use serde_json::Value;

fn preserved_tree(
    root: &std::path::Path,
) -> std::collections::BTreeMap<std::path::PathBuf, (Option<blake3::Hash>, std::time::SystemTime)> {
    fn visit(
        root: &std::path::Path,
        path: &std::path::Path,
        files: &mut std::collections::BTreeMap<
            std::path::PathBuf,
            (Option<blake3::Hash>, std::time::SystemTime),
        >,
    ) {
        let metadata = std::fs::metadata(path).unwrap();
        let digest = metadata
            .is_file()
            .then(|| blake3::hash(&std::fs::read(path).unwrap()));
        files.insert(
            path.strip_prefix(root).unwrap().to_owned(),
            (digest, metadata.modified().unwrap()),
        );
        if metadata.is_dir() {
            for entry in std::fs::read_dir(path).unwrap() {
                visit(root, &entry.unwrap().path(), files);
            }
        }
    }
    let mut files = std::collections::BTreeMap::new();
    visit(root, root, &mut files);
    files
}

fn read_presentation() -> ExecutionPresentation {
    ExecutionPresentation::Read {
        options: Default::default(),
        delivery: Default::default(),
        format: polylogue_model::read::Format::Json,
        page_budget: Some(Default::default()),
        direct_source: None,
    }
}
fn old_body(reference: Option<Reference>) -> (Value, QueryRunOutcome) {
    let object = question("sessions");
    let mut current = run(
        &object,
        "preserved-run",
        100,
        ChangePosition {
            generation: "preserved".into(),
            index: 0,
            archive: 0,
        },
    );
    if reference.is_some() {
        current.binding.original_mut().unwrap().presentation = read_presentation();
    }
    current
        .binding
        .original_mut()
        .unwrap()
        .query_options
        .references = reference.clone().map(|reference| vec![reference]);
    let mut body = serde_json::to_value(&current).unwrap();
    let options = body["binding"]["execution"]["query_options"]
        .as_object_mut()
        .unwrap();
    options.remove("references");
    options.insert(
        "direct_reference".into(),
        serde_json::to_value(reference).unwrap(),
    );
    let hash = |tag, value: &Value| {
        BlobHash(
            *canonical_json::canonical_hash(tag, value)
                .unwrap()
                .as_bytes(),
        )
    };
    let mut completion = outcome(&current, 7);
    completion.binding_hash = hash(
        "polylogue-execution-binding-v1",
        &body["binding"]["execution"],
    );
    completion.run_hash = hash("polylogue-query-run-begin-v1", &body);
    (body, completion)
}

#[test]
fn preserved_execution_conversion_maps_none_and_one_and_is_idempotent() {
    for reference in [
        None,
        Some(Reference::parse("session:codex:preserved").unwrap()),
    ] {
        let (body, completion) = old_body(reference.clone());
        let converted = convert_run(body).unwrap();
        assert_eq!(
            converted
                .run
                .binding
                .original()
                .unwrap()
                .query_options
                .references,
            reference.map(|reference| vec![reference])
        );
        let completed = converted.completion(completion.clone()).unwrap();
        assert_eq!(completed.count, Some(7));
        assert_eq!(completed.run, completion.run);
        assert_eq!(completed.at, completion.at);
        assert_eq!(completed.outcome, completion.outcome);
        assert_eq!(
            completed.binding_hash,
            converted.run.binding_hash().unwrap()
        );
        assert_eq!(completed.run_hash, converted.run.record_hash().unwrap());
        let again = convert_run(serde_json::to_value(&converted.run).unwrap()).unwrap();
        assert_eq!(again.run, converted.run);
        assert_eq!(again.completion(completed.clone()).unwrap(), completed);
    }
}

#[test]
fn current_zero_and_ordered_populations_and_object_runs_stay_intact() {
    let object = question("sessions");
    let mut current = run(
        &object,
        "unchanged-run",
        100,
        ChangePosition {
            generation: "preserved".into(),
            index: 0,
            archive: 0,
        },
    );
    current.binding.original_mut().unwrap().presentation = read_presentation();
    for references in [
        vec![],
        vec![
            Reference::parse("message:codex:m2").unwrap(),
            Reference::parse("session:codex:s1").unwrap(),
        ],
    ] {
        current
            .binding
            .original_mut()
            .unwrap()
            .query_options
            .references = Some(references);
        let converted = convert_run(serde_json::to_value(&current).unwrap()).unwrap();
        assert_eq!(converted.run, current);
        assert_eq!(converted.old_binding, current.binding_hash().unwrap());
        assert_eq!(converted.old_run, current.record_hash().unwrap());
    }
    current.question = None;
    current.binding = RunBinding::OriginalObject {
        execution: Box::new(polylogue_model::query_objects::ObjectExecutionBinding {
            reference: Reference::Request("preserved-request".into()),
            content_hash: BlobHash::of(b"original bytes"),
            scope_fingerprint: None,
            versions: Default::default(),
            presentation: read_presentation(),
            accepted_request: None,
        }),
    };
    let converted = convert_run(serde_json::to_value(&current).unwrap()).unwrap();
    assert_eq!(converted.run, current);
    assert_eq!(converted.old_binding, current.binding_hash().unwrap());
    assert_eq!(converted.old_run, current.record_hash().unwrap());
}

fn preserved_register(path: &std::path::Path, corrupt: bool) -> (Value, QueryRunOutcome) {
    std::fs::write(path.parent().unwrap().join("lock"), b"").unwrap();
    let connection = rusqlite::Connection::open(path).unwrap();
    crate::schema::ensure_archive(&connection).unwrap();
    super::super::apply_object(&connection, &question("sessions"), 1).unwrap();
    let (body, completion) = old_body(Some(Reference::parse("session:codex:preserved").unwrap()));
    connection
        .execute(
            "INSERT INTO meta(key,value) VALUES('preserved-fixture','unaltered')",
            [],
        )
        .unwrap();
    connection.execute("INSERT INTO query_run(id,binding_hash,run_hash,query,run,completion) VALUES(?1,?2,?3,?4,?5,?6)",
        params![completion.run.as_str(),completion.binding_hash.to_string(),
            if corrupt {BlobHash::of(b"wrong").to_string()} else {completion.run_hash.to_string()},
            question("sessions").id.to_string(),super::super::pack(&body).unwrap(),super::super::pack(&completion).unwrap()]).unwrap();
    connection.pragma_update(None, "user_version", 3).unwrap();
    (body, completion)
}

#[test]
fn preserved_register_conversion_keeps_source_bytes_and_publishes_once() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("preserved.db");
    let output = tempfile::tempdir().unwrap();
    let target = output.path().join("current.db");
    let vectors = directory.path().join("embeddings.db");
    std::fs::write(&vectors, b"synthetic paid output bytes").unwrap();
    let (_, completion) = preserved_register(&source, false);
    let original = std::fs::read(&source).unwrap();
    assert_eq!(
        convert_preserved(&source, &target, |_| Ok(()))
            .unwrap()
            .changed_runs,
        1
    );
    assert_eq!(std::fs::read(&source).unwrap(), original);
    assert_eq!(
        std::fs::read(&vectors).unwrap(),
        b"synthetic paid output bytes"
    );
    let output = Connection::open(&target).unwrap();
    crate::schema::ensure_archive(&output).unwrap();
    assert_eq!(crate::schema::version(&output).unwrap(), 4);
    assert_eq!(
        output
            .query_row(
                "SELECT value FROM meta WHERE key='preserved-fixture'",
                [],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
        "unaltered"
    );
    let (run, completed): (Vec<u8>, Vec<u8>) = output
        .query_row(
            "SELECT run,completion FROM query_run WHERE id=?1",
            [completion.run.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    let run: QueryRun = crate::encoding::unpack_json(&run).unwrap();
    let completed: QueryRunOutcome = crate::encoding::unpack_json(&completed).unwrap();
    assert_eq!(
        run.binding.original().unwrap().query_options.references,
        Some(vec![Reference::parse("session:codex:preserved").unwrap()])
    );
    assert_eq!(completed.count, Some(7));
    assert_eq!(completed.binding_hash, run.binding_hash().unwrap());
    assert_eq!(completed.run_hash, run.record_hash().unwrap());
    drop(output);
    let current_bytes = std::fs::read(&target).unwrap();
    assert_eq!(
        convert_preserved(&source, &target, |_| Ok(()))
            .unwrap()
            .changed_runs,
        0
    );
    assert_eq!(std::fs::read(&target).unwrap(), current_bytes);
    assert_eq!(std::fs::read(&source).unwrap(), original);
}

#[test]
fn invalid_preserved_hash_publishes_nothing_and_keeps_the_source() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("invalid.db");
    let output = tempfile::tempdir().unwrap();
    let target = output.path().join("unpublished.db");
    preserved_register(&source, true);
    let original = std::fs::read(&source).unwrap();
    assert!(matches!(
        convert_preserved(&source, &target, |_| Ok(())),
        Err(StoreError::Conflict(_))
    ));
    assert!(!target.exists());
    assert_eq!(std::fs::read(&source).unwrap(), original);
}

#[test]
fn pending_begin_and_completion_join_the_preserved_population_once() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("preserved.db");
    let output = tempfile::tempdir().unwrap();
    let target = output.path().join("current.db");
    std::fs::write(directory.path().join("lock"), b"").unwrap();
    let connection = Connection::open(&source).unwrap();
    crate::schema::ensure_archive(&connection).unwrap();
    connection.pragma_update(None, "user_version", 3).unwrap();
    drop(connection);
    let original = std::fs::read(&source).unwrap();
    let (body, completion) = old_body(None);
    let groups = [
        serde_json::json!({"product":{"product":"query","object":question("sessions")}}),
        serde_json::json!({"product":{"product":"query_run_outcome","outcome":completion}}),
        serde_json::json!({"product":{"product":"query_run","run":body}}),
    ];
    let append = |connection: &rusqlite::Transaction<'_>| {
        crate::query_objects::line_population::stage_products(connection, &groups)?;
        crate::query_objects::line_population::stage_products(connection, &groups)
    };
    assert_eq!(
        convert_preserved(&source, &target, append)
            .unwrap()
            .changed_runs,
        1
    );
    assert_eq!(std::fs::read(&source).unwrap(), original);
    let output = Connection::open(&target).unwrap();
    assert_eq!(
        output
            .query_row("SELECT count(*) FROM query_run", [], |row| row
                .get::<_, u64>(0))
            .unwrap(),
        1
    );
    let packed: Vec<u8> = output
        .query_row("SELECT completion FROM query_run", [], |row| row.get(0))
        .unwrap();
    let current: QueryRunOutcome = crate::encoding::unpack_json(&packed).unwrap();
    assert_eq!(current.count, completion.count);
    assert_eq!(current.run, completion.run);
}

#[test]
fn wal_child() {
    let Some(path) = std::env::var_os("POLYLOGUE_EXECUTION_CONVERSION_CHILD") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    std::fs::write(path.parent().unwrap().join("lock"), b"").unwrap();
    let connection = Connection::open(path).unwrap();
    crate::schema::ensure_archive(&connection).unwrap();
    connection
        .pragma_update(None, "journal_mode", "WAL")
        .unwrap();
    connection
        .pragma_update(None, "wal_autocheckpoint", 0)
        .unwrap();
    connection
        .execute(
            "INSERT INTO meta(key,value) VALUES('committed-in-wal','present')",
            [],
        )
        .unwrap();
    connection.pragma_update(None, "user_version", 3).unwrap();
    // A stopped process may leave committed WAL frames. Conversion must read
    // their stable copied set without opening or checkpointing the originals.
    std::process::exit(0);
}

#[test]
fn preserved_wal_commits_survive_without_altering_original_files() {
    let source = tempfile::tempdir().unwrap();
    let destination = tempfile::tempdir().unwrap();
    let register = source.path().join("archive.db");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "query_objects::tests::line_population::wal_child",
            "--nocapture",
        ])
        .env("POLYLOGUE_EXECUTION_CONVERSION_CHILD", &register)
        .status()
        .unwrap();
    assert!(status.success());
    assert!(
        source
            .path()
            .join("archive.db-wal")
            .metadata()
            .unwrap()
            .len()
            > 0
    );
    let original = preserved_tree(source.path());
    let current = destination.path().join("current.db");
    assert_eq!(
        convert_preserved(&register, &current, |_| Ok(()))
            .unwrap()
            .changed_runs,
        0
    );
    let converted = Connection::open(current).unwrap();
    assert_eq!(
        converted
            .query_row(
                "SELECT value FROM meta WHERE key='committed-in-wal'",
                [],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
        "present"
    );
    let after = preserved_tree(source.path());
    assert_eq!(after, original);
}

#[test]
fn active_archive_or_missing_custody_lock_changes_no_preserved_entries() {
    let source = tempfile::tempdir().unwrap();
    let destination = tempfile::tempdir().unwrap();
    let register = source.path().join("archive.db");
    preserved_register(&register, false);
    let original = preserved_tree(source.path());
    let lock = std::fs::File::open(source.path().join("lock")).unwrap();
    lock.lock().unwrap();
    let target = destination.path().join("current.db");
    assert!(convert_preserved(&register, &target, |_| Ok(())).is_err());
    assert!(!target.exists());
    assert_eq!(preserved_tree(source.path()), original);
    drop(lock);
    std::fs::remove_file(source.path().join("lock")).unwrap();
    let before = preserved_tree(source.path());
    assert!(convert_preserved(&register, &target, |_| Ok(())).is_err());
    assert_eq!(preserved_tree(source.path()), before);
    assert!(!source.path().join("lock").exists());
    assert!(!target.exists());
    assert_eq!(preserved_tree(source.path()), before);
}

#[test]
fn converted_products_reconstruct_exact_objects_begins_and_completions() {
    let source = tempfile::tempdir().unwrap();
    let destination = tempfile::tempdir().unwrap();
    let original = source.path().join("archive.db");
    let current = destination.path().join("current.db");
    let (body, completion) = preserved_register(&original, false);
    assert_eq!(
        convert_preserved(&original, &current, |_| Ok(()))
            .unwrap()
            .changed_runs,
        1
    );
    let mut products = vec![];
    assert_eq!(
        crate::query_objects::line_population::visit_current_products(&current, |product| {
            products.push(product);
            Ok(())
        })
        .unwrap(),
        3
    );
    let expected = convert_run(body).unwrap();
    assert_eq!(
        products,
        vec![
            crate::primary::Product::Query {
                object: Box::new(question("sessions"))
            },
            crate::primary::Product::QueryRun {
                run: Box::new(expected.run.clone())
            },
            crate::primary::Product::QueryRunOutcome {
                outcome: Box::new(expected.completion(completion).unwrap())
            },
        ]
    );
    let mut repeated = vec![];
    crate::query_objects::line_population::visit_current_products(&current, |product| {
        repeated.push(product);
        Ok(())
    })
    .unwrap();
    assert_eq!(products, repeated);
}

#[test]
fn primary_cache_transition_uses_source_tables_and_current_products() {
    let source = tempfile::tempdir().unwrap();
    let destination = tempfile::tempdir().unwrap();
    let original = source.path().join("index.db");
    let current = destination.path().join("current.jsonl");
    std::fs::write(source.path().join("lock"), b"").unwrap();
    let connection = Connection::open(&original).unwrap();
    crate::schema::ensure_primary_index(&connection).unwrap();
    super::super::apply_object(&connection, &question("sessions"), 1).unwrap();
    let (body, completion) = old_body(None);
    connection.execute("INSERT INTO source_query_run(id,binding_hash,run_hash,query,run,completion) VALUES(?1,?2,?3,?4,?5,?6)", params![completion.run.as_str(), completion.binding_hash.to_string(), completion.run_hash.to_string(), question("sessions").id.to_string(), super::super::pack(&body).unwrap(), super::super::pack(&completion).unwrap()]).unwrap();
    // The protected cache8 has inline uniqueness constraints. B3's cache9
    // declares separate build indexes, so retagging this file cannot open it.
    connection.execute_batch("DROP TABLE message_version; CREATE TABLE message_version(pk INTEGER PRIMARY KEY,version_hash BLOB NOT NULL UNIQUE) STRICT;
        DROP TABLE event_value; CREATE TABLE event_value(pk INTEGER PRIMARY KEY,hash BLOB NOT NULL UNIQUE,body BLOB NOT NULL) STRICT;
        DROP TABLE link_value; CREATE TABLE link_value(pk INTEGER PRIMARY KEY,hash BLOB NOT NULL UNIQUE,body BLOB NOT NULL) STRICT;").unwrap();
    connection.pragma_update(None, "user_version", 8).unwrap();
    drop(connection);
    let before = preserved_tree(source.path());
    let converted_artifact = convert_preserved(&original, &current, |_| Ok(())).unwrap();
    assert_eq!(converted_artifact.changed_runs, 1);
    assert_eq!(
        converted_artifact.output,
        crate::query_objects::line_population::OutputKind::Products
    );
    let bytes = std::fs::read(&current).unwrap();
    assert!(!bytes.starts_with(b"SQLite format 3"));
    let products: Vec<crate::primary::Product> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(products.len(), 3);
    let converted = convert_run(body).unwrap();
    assert_eq!(
        products[1],
        crate::primary::Product::QueryRun {
            run: Box::new(converted.run.clone())
        }
    );
    assert_eq!(
        products[2],
        crate::primary::Product::QueryRunOutcome {
            outcome: Box::new(converted.completion(completion).unwrap())
        }
    );
    assert_eq!(
        convert_preserved(&original, &current, |_| Ok(()))
            .unwrap()
            .changed_runs,
        0
    );
    assert_eq!(std::fs::read(&current).unwrap(), bytes);
    assert!(
        crate::query_objects::line_population::visit_current_products(&original, |_| Ok(()))
            .is_err()
    );
    assert_eq!(preserved_tree(source.path()), before);
}
