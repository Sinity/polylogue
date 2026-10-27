use super::*;
use polylogue_store::{Batch, Change, InputKey, Observation, fixture::writer::WriterArchive};

fn stored(archive: &WriterArchive, revision: bool, bytes: &[u8]) {
    store(&archive.archive, &archive.writer, revision, bytes);
}

fn store(archive: &Archive, writer: &polylogue_store::Writer, revision: bool, bytes: &[u8]) {
    let prepared = archive.prepare_bytes(bytes).unwrap();
    writer
        .store_bytes(
            prepared,
            Batch {
                input: InputKey {
                    id: "input:synthetic".into(),
                    format: "claude-code".into(),
                    arrival_rank: 0,
                },
                change: if revision {
                    Change::Rewritten
                } else {
                    Change::First
                },
                observation: Some(Observation {
                    machine: "synthetic".into(),
                    host: "synthetic".into(),
                    path: "/absent/session.jsonl".into(),
                    facts: None,
                    seen_at_ms: 1,
                    import: None,
                }),
                continues: false,
                new_blob: None,
                blobs: vec![],
                length: bytes.len() as u64,
                parse: vec![],
                release: Default::default(),
            },
        )
        .unwrap();
}

#[test]
fn offline_selection_reads_only_selected_stored_revisions_and_reports_failures() {
    let archive = WriterArchive::new();
    stored(&archive, false, b"{\"id\":\"secret\"}\nmalformed\n");
    stored(&archive, true, b"{\"count\":2}\n");
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let spec = polylogue_sources::origin::by_name("claude-code").unwrap();
    let inputs = ["input:synthetic".into(), "missing".into()].into();
    let schema = observe(
        &reader,
        spec,
        inputs,
        Some(1),
        ["id".into(), "count".into()].into(),
    )
    .unwrap();
    assert_eq!(
        (
            schema.records,
            schema.unreadable_records,
            schema.unreadable_inputs
        ),
        (1, 1, 1)
    );
    assert!(schema.fields.contains_key("/id"));
    assert!(!schema.fields.contains_key("/count"));
    assert_eq!(schema.selection[0].revision, Some(1));
    let latest = observe(
        &reader,
        spec,
        ["input:synthetic".into()].into(),
        None,
        ["count".into()].into(),
    )
    .unwrap();
    assert_eq!(latest.selection[0].revision, Some(2));
    assert!(latest.fields.contains_key("/count"));
}

#[test]
fn schema_infer_reads_while_the_daemon_archive_lock_is_held_and_diff_reuses_output() {
    let archive = WriterArchive::new();
    stored(&archive, false, b"{\"id\":\"secret\"}\n");
    let bytes = format!(r#"{{"id":"{}"}}"#, "secret".repeat(30_000));
    stored(&archive, true, bytes.as_bytes());
    let blob_tmp = archive.archive.blobs_dir().join("tmp");
    std::fs::remove_dir(&blob_tmp).unwrap();
    assert!(archive.archive.lock_held());
    let archive_before = std::fs::read(archive.archive.archive_db()).unwrap();
    let index_before = std::fs::read(archive.archive.index_db()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("schema.json");
    let diff_output = dir.path().join("diff.json");
    let args = [
        "polylogue",
        "schema",
        "infer",
        "claude-code",
        "--input",
        "input:synthetic",
        "--archive",
        archive.archive.root.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
    ];
    let (command, values) = crate::cli::parse(&args).unwrap();
    let crate::commands::Action::Local(action) = command.action else {
        panic!("local command");
    };
    assert_eq!(action(command, &values, true), ExitCode::SUCCESS);
    let bytes = std::fs::read(&output).unwrap();
    assert!(!String::from_utf8_lossy(&bytes).contains("secret"));
    let schema: Schema = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(schema.records, 1);
    assert!(schema.fields.contains_key("/id"));
    assert!(!blob_tmp.exists());
    assert!(archive.archive.lock_held());
    assert_eq!(
        std::fs::read(archive.archive.archive_db()).unwrap(),
        archive_before
    );
    assert_eq!(
        std::fs::read(archive.archive.index_db()).unwrap(),
        index_before
    );
    let (command, values) = crate::cli::parse(&[
        "polylogue",
        "schema",
        "diff",
        output.to_str().unwrap(),
        output.to_str().unwrap(),
        "--output",
        diff_output.to_str().unwrap(),
    ])
    .unwrap();
    assert_eq!(diff(command, &values, true), ExitCode::SUCCESS);
    let report: schema::Diff =
        serde_json::from_slice(&std::fs::read(diff_output).unwrap()).unwrap();
    assert!(report.changes.is_empty());
    assert!(report.comparable);
}

struct SinexStore(Archive);
impl polylogue_store::primary::Sinex for SinexStore {
    fn send_part(
        &self,
        part: &polylogue_store::primary::Part,
        bytes: &mut dyn io::Read,
    ) -> polylogue_store::Result<String> {
        let hash = self.0.blobs()?.put(bytes, 1)?;
        assert_eq!(hash.0, part.hash.0);
        Ok("synthetic-material".into())
    }
    fn send_change(
        &self,
        _: &polylogue_store::primary::RegisterChange,
    ) -> polylogue_store::Result<String> {
        Ok("synthetic-change".into())
    }
    fn excise(&self, _: &[String]) -> polylogue_store::Result<()> {
        unreachable!()
    }
    fn held(&self, _: i64) -> polylogue_store::Result<Vec<polylogue_store::primary::Held>> {
        Ok(vec![])
    }
    fn send_sessions(&self, _: &ReadPool) -> polylogue_store::Result<()> {
        Ok(())
    }
}

#[test]
fn primary_inference_reads_sinex_bytes_without_a_standalone_archive_or_blob_copy() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("primary");
    let bytes_root = dir.path().join("sinex");
    let sinex = Arc::new(SinexStore(Archive::at(&bytes_root)));
    let (archive, _lock) = Archive::open_primary(&root, &bytes_root, sinex.as_ref()).unwrap();
    let writer = polylogue_store::Writer::start_on_sinex(&archive, sinex).unwrap();
    store(&archive, &writer, false, b"{\"id\":\"only-in-sinex\"}\n");
    let blob_tmp = bytes_root.join("blobs/tmp");
    std::fs::remove_dir(&blob_tmp).unwrap();
    let pool = ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let report = observe(
        &reader,
        polylogue_sources::origin::by_name("claude-code").unwrap(),
        ["input:synthetic".into()].into(),
        None,
        ["id".into()].into(),
    )
    .unwrap();
    assert_eq!(report.records, 1);
    assert_eq!(report.fields["/id"].types["string"], 1);
    assert!(!root.join("archive.db").exists());
    assert!(!root.join("blobs").exists());
    assert!(!blob_tmp.exists());
}

#[test]
fn wrong_origin_and_unavailable_revision_are_reported_without_replacement_inputs() {
    let archive = WriterArchive::new();
    stored(&archive, false, b"{}\n");
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    for (origin, revision, reason) in [
        ("codex", None, "different_origin"),
        ("claude-code", Some(8), "input_or_revision_missing"),
    ] {
        let report = observe(
            &reader,
            polylogue_sources::origin::by_name(origin).unwrap(),
            ["input:synthetic".into()].into(),
            revision,
            BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(report.records, 0);
        assert_eq!(report.unreadable_inputs, 1);
        assert_eq!(report.selection[0].unreadable.as_deref(), Some(reason));
        assert!(report.fields.is_empty());
    }
}
