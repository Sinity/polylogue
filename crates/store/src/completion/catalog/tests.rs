use super::*;
fn metadata(index: i64) -> Metadata {
    Metadata {
        archive: "synthetic".into(),
        position: ChangePosition {
            generation: "generation".into(),
            index,
            archive: index,
        },
        written_at: 100 + index,
    }
}
fn session(id: &str, model: &str, n: u64) -> Session {
    Session {
        id: id.into(),
        native_id: id.split_once(':').unwrap().1.into(),
        prefix: String::new(),
        title: "A neutral title".into(),
        origin: "codex".into(),
        repo: Some("project".into()),
        last_activity: Some(10),
        first_prompt: Some("A neutral prompt".into()),
        values: BTreeMap::from([("model".into(), BTreeMap::from([(model.into(), n)]))]),
    }
}
fn writer(path: &Path) -> CatalogWriter {
    CatalogWriter::build(path, &metadata(1), |writer| {
        writer.put_catalog(&query_language::catalog::Catalog::new("sessions"))
    })
    .unwrap()
}
#[test]
fn replacement_subtracts_only_its_contributions_and_commits_the_position() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut writer = writer(&path);
    writer
        .update(&metadata(2), |writer| {
            writer.put_session(&session("codex:aaa", "model-a", 3))?;
            writer.put_session(&session("codex:aab", "model-a", 2))
        })
        .unwrap();
    writer
        .update(&metadata(3), |writer| {
            writer.put_session(&session("codex:aaa", "model-b", 1))
        })
        .unwrap();
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(reader.metadata.position, metadata(3).position);
    assert_eq!(
        reader
            .values("model", "")
            .unwrap()
            .iter()
            .map(|v| (v.value.as_str(), v.count))
            .collect::<Vec<_>>(),
        vec![("model-a", 2), ("model-b", 1)]
    );
    assert_eq!(
        reader
            .session_details("codex:aaa")
            .unwrap()
            .unwrap()
            .first_prompt
            .as_deref(),
        Some("A neutral prompt")
    );
    drop(reader);
    writer
        .update(&metadata(4), |writer| writer.remove_session("codex:aab"))
        .unwrap();
    assert_eq!(
        CatalogReader::open(&path, "synthetic")
            .unwrap()
            .unwrap()
            .values("model", "")
            .unwrap()[0]
            .value,
        "model-b"
    );
}
#[test]
fn a_failed_publication_rolls_back_values_and_position() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut writer = writer(&path);
    assert!(
        writer
            .update(&metadata(2), |writer| {
                writer.put_session(&session("codex:abc", "temporary", 1))?;
                Err(StoreError::Invalid("synthetic failure".into()))
            })
            .is_err()
    );
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(reader.metadata.position, metadata(1).position);
    assert!(reader.values("model", "").unwrap().is_empty());
}
#[test]
fn prefix_updates_include_neighbors_unicode_and_removed_sessions() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut writer = writer(&path);
    writer
        .update(&metadata(2), |writer| {
            writer.put_session(&session("codex:żaa", "m", 1))?;
            writer.put_session(&session("codex:żab", "m", 1))?;
            writer.put_session(&session("codex:żac", "m", 1))
        })
        .unwrap();
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    let rows = reader.sessions("ż").unwrap();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|row| row.prefix == row.id));
    drop(reader);
    writer
        .update(&metadata(3), |writer| {
            writer.remove_session("codex:żaa")?;
            writer.remove_session("codex:żab")
        })
        .unwrap();
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(reader.sessions("ż").unwrap()[0].prefix, "codex:ż");
    assert_eq!(
        reader.sessions("ntrl").unwrap().len(),
        1,
        "fuzzy title fallback"
    );
}
#[test]
fn read_only_snapshot_never_writes_and_typed_items_use_exact_identity() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut writer = writer(&path);
    let schema = query_language::catalog::OutputSchema {
        match_units: None,
        unit: Some("messages".into()),
        fields: vec![],
        main_time: None,
    };
    writer
        .update(&metadata(2), |writer| {
            for name in ["one", "one-more"] {
                writer.put_saved(&Saved {
                    relation_scope: None,
                    family: name.into(),
                    revision: format!("{name}-r1"),
                    name: name.into(),
                    text: "messages".into(),
                    parameters: vec![],
                    help: "A saved query".into(),
                    schema: Some(schema.clone()),
                })?;
            }
            writer.put_kept(&Kept {
                id: "result-id".into(),
                name: Some("human name".into()),
                unit: Some("messages".into()),
                row_count: Some(3),
                rows: Some(schema.clone()),
                matches: None,
            })
        })
        .unwrap();
    let files = fs::read_dir(directory.path()).unwrap().count();
    let before = fs::metadata(&path).unwrap().modified().unwrap();
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(reader.saved("one").unwrap().len(), 1);
    assert!(reader.saved("on").unwrap().is_empty());
    assert_eq!(reader.saved_names("on").unwrap().len(), 2);
    assert_eq!(
        reader.kept("result-id").unwrap().unwrap().name.as_deref(),
        Some("human name")
    );
    assert!(reader.kept("result").unwrap().is_none());
    assert!(reader.conn.execute("DELETE FROM session", []).is_err());
    drop(reader);
    assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), before);
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), files);
    assert!(CatalogReader::open(&path, "different").unwrap().is_none());
}

#[test]
fn assertion_cache_candidates_include_tag_only_owners_and_remove_only_their_contribution() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog");
    let mut cache = writer(&path);
    cache
        .update(&metadata(2), |cache| {
            cache.put_saved(&Saved {
                relation_scope: None,
                family: "saved-root".into(),
                revision: "saved-revision".into(),
                name: "saved-name".into(),
                text: "sessions".into(),
                parameters: vec![],
                help: String::new(),
                schema: None,
            })?;
            cache.put_values(
                "assertion:tag-root",
                &BTreeMap::from([("tag".into(), BTreeMap::from([("research".into(), 1)]))]),
            )?;
            cache.put_values(
                "session:codex:other",
                &BTreeMap::from([("tag".into(), BTreeMap::from([("research".into(), 3)]))]),
            )?;
            cache.put_setting(&Setting {
                key: "browser.actions".into(),
                choices: vec!["on".into(), "off".into()],
                value: polylogue_model::settings::Current {
                    key: "browser.actions".into(),
                    kind: polylogue_model::settings::Kind::Boolean,
                    default: Some(polylogue_model::settings::Value::Boolean(false)),
                    scope: polylogue_model::settings::Scope::Archive,
                    owner: "polylogue".into(),
                    value: None,
                    authored: None,
                    conflicts: vec![
                        polylogue_model::Reference::parse("assertion:setting@left").unwrap(),
                        polylogue_model::Reference::parse("assertion:setting@right").unwrap(),
                    ],
                },
            })?;
            Ok(())
        })
        .unwrap();
    assert_eq!(
        cache.definition_families().unwrap(),
        ["saved-root", "tag-root"]
    );
    assert_eq!(cache.setting_keys().unwrap(), ["browser.actions"]);
    {
        let read = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
        assert_eq!(read.values("tag", "").unwrap()[0].count, 4);
        let settings = read.settings("browser.actions").unwrap();
        assert_eq!(settings[0].value.conflicts.len(), 2);
        assert!(settings[0].value.value.is_none());
    }
    // An absent-family update removes only that cached contribution.
    cache
        .update(&metadata(3), |cache| {
            cache.put_values("assertion:tag-root", &BTreeMap::new())?;
            cache.remove_setting("browser.actions")
        })
        .unwrap();
    assert_eq!(cache.definition_families().unwrap(), ["saved-root"]);
    assert!(cache.setting_keys().unwrap().is_empty());
    let read = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(read.values("tag", "").unwrap()[0].count, 3);
    assert!(read.settings("browser.actions").unwrap().is_empty());
}

#[test]
fn derived_saved_pattern_codec_is_atomic_and_old_versions_are_invalidated() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut writer = writer(&path);
    let patterns = Some(vec!["tool_calls".to_owned()]);
    let saved = Saved {
        relation_scope: Some(OutputSchema {
            match_units: patterns.clone(),
            unit: Some("sessions".into()),
            ..OutputSchema::default()
        }),
        family: "family".into(),
        revision: "revision".into(),
        name: "qualified".into(),
        text: "sessions where exists tool_calls()".into(),
        parameters: vec![],
        help: "Qualified sessions".into(),
        schema: Some(OutputSchema {
            match_units: patterns.clone(),
            unit: Some("sessions".into()),
            fields: vec![],
            main_time: None,
        }),
    };
    writer
        .update(&metadata(2), |writer| writer.put_saved(&saved))
        .unwrap();
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(
        reader.saved("qualified").unwrap()[0]
            .schema
            .as_ref()
            .unwrap()
            .match_units,
        patterns
    );
    assert_eq!(
        reader.saved_names("qual").unwrap()[0]
            .schema
            .as_ref()
            .unwrap()
            .match_units,
        patterns
    );
    assert_eq!(
        reader.saved("qualified").unwrap()[0]
            .relation_scope
            .as_ref()
            .unwrap()
            .match_units,
        patterns
    );
    drop(reader);
    assert!(
        writer
            .update(&metadata(3), |writer| {
                let mut unavailable = saved.clone();
                unavailable.schema = None;
                writer.put_saved(&unavailable)?;
                Err(StoreError::Invalid("synthetic rollback".into()))
            })
            .is_err()
    );
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(
        reader.saved("qualified").unwrap()[0]
            .schema
            .as_ref()
            .unwrap()
            .match_units,
        patterns
    );
    drop(reader);
    writer
        .conn
        .execute(
            "UPDATE item SET payload=json_remove(payload,'$.relation_scope.match_units') WHERE kind='saved'",
            [],
        )
        .unwrap();
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(
        reader.saved("qualified").unwrap()[0]
            .relation_scope
            .as_ref()
            .unwrap()
            .match_units,
        None,
        "missing provenance stays unknown instead of becoming known empty qualifiers"
    );
    drop(reader);
    writer
        .conn
        .execute("UPDATE metadata SET value='5' WHERE key='version'", [])
        .unwrap();
    assert!(CatalogReader::open(&path, "synthetic").unwrap().is_none());
}

#[test]
fn publication_keeps_old_snapshots_readable_and_offline_reads_create_no_files() {
    const CHILD: &str = "POLYLOGUE_COMPLETION_READER_CASE";
    const VALUE: &str = "POLYLOGUE_COMPLETION_READER_VALUE";
    const POSITION: &str = "POLYLOGUE_COMPLETION_READER_POSITION";
    const VERIFIED: &str = "completion reader verified values and position";
    if let Some(path) = std::env::var_os(CHILD) {
        let reader = CatalogReader::open(Path::new(&path), "synthetic")
            .unwrap()
            .unwrap();
        assert_eq!(
            reader.values("model", "").unwrap()[0].value,
            std::env::var(VALUE).unwrap()
        );
        assert_eq!(
            reader.metadata.position,
            metadata(std::env::var(POSITION).unwrap().parse().unwrap()).position
        );
        eprintln!("{VERIFIED}");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("generation.db");
    use std::os::unix::ffi::OsStringExt;
    let pointer = directory
        .path()
        .join(std::ffi::OsString::from_vec(b"catalog %?#\xff".to_vec()));
    let child = |value: &str, position: i64| {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "completion::catalog::tests::publication_keeps_old_snapshots_readable_and_offline_reads_create_no_files", "--nocapture"])
            .env(CHILD, &pointer).env(VALUE, value).env(POSITION, position.to_string()).output().unwrap();
        assert!(output.status.success(), "reader child: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
            "reader child: {output:?}"
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .filter(|line| *line == VERIFIED)
                .count(),
            1,
            "reader child: {output:?}"
        );
    };
    let files = || {
        fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (entry.file_name(), fs::read(entry.path()).unwrap())
            })
            .collect::<BTreeMap<_, _>>()
    };
    let mut writer = writer(&path);
    std::os::unix::fs::symlink("generation.db", &pointer).unwrap();
    writer
        .update(&metadata(2), |writer| {
            writer.put_session(&session("codex:one", "before", 1))
        })
        .unwrap();
    let old = CatalogReader::open(&pointer, "synthetic").unwrap().unwrap();
    // In WAL mode even an exclusive publication transaction leaves committed
    // snapshots available. Rollback journaling blocks this new reader.
    writer.conn.execute_batch("BEGIN EXCLUSIVE").unwrap();
    writer
        .put_session(&session("codex:one", "uncommitted", 1))
        .unwrap();
    let during = CatalogReader::open(&pointer, "synthetic").unwrap().unwrap();
    assert_eq!(during.values("model", "").unwrap()[0].value, "before");
    assert_eq!(during.metadata.position, metadata(2).position);
    child("before", 2);
    writer.conn.execute_batch("ROLLBACK").unwrap();
    writer
        .update(&metadata(3), |writer| {
            writer.put_session(&session("codex:one", "after", 1))
        })
        .unwrap();
    let before_child = files();
    child("after", 3);
    assert_eq!(files(), before_child);
    assert_eq!(old.values("model", "").unwrap()[0].value, "before");
    assert_eq!(old.metadata.position, metadata(2).position);
    let current = CatalogReader::open(&pointer, "synthetic").unwrap().unwrap();
    assert_eq!(current.values("model", "").unwrap()[0].value, "after");
    assert_eq!(current.metadata.position, metadata(3).position);
    drop((old, during, current));
    drop(writer);
    let before = files();
    assert!(before.contains_key(std::ffi::OsStr::new("generation.db-wal")));
    assert!(before.contains_key(std::ffi::OsStr::new("generation.db-shm")));
    let offline = CatalogReader::open(&pointer, "synthetic").unwrap().unwrap();
    assert_eq!(offline.values("model", "").unwrap()[0].value, "after");
    drop(offline);
    assert_eq!(files(), before);
    child("after", 3);
    assert_eq!(files(), before);
}
