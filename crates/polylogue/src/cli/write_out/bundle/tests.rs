use super::*;
use polylogue_model::{Author, ChangePosition, Timestamp};
fn begin() -> ReadBegin {
    ReadBegin {
        snapshot: ReadSnapshot {
            query: format!("query:{}", "a".repeat(64)),
            query_text: "sessions".into(),
            ran_at: Timestamp::from_unix_ms(1).unwrap(),
            read_at: ChangePosition {
                generation: "synthetic".into(),
                index: 1,
                archive: 1,
            },
            caught_up: true,
            currency: polylogue_model::currency::IndexCurrency::Standalone,
            enabled_variants: vec![],
            author: Author::Operator,
            notes: vec![],
            incomplete: vec![],
        },
        header: ReadHeader {
            target: "query:synthetic".into(),
            view: View::Transcript,
            title: ValueState::known("Synthetic bundle".into()),
            kind: HeaderKind::Query {
                query: "sessions".into(),
            },
            figures: vec![],
        },
        options: ReadOptions::default(),
    }
}
fn session(native: &str) -> ReadRecord {
    let session =
        serde_json::from_value(serde_json::json!({"id":format!("codex:{native}"),"kind":"main"}))
            .unwrap();
    ReadRecord {
        record: Record::Session(SessionRecord {
            reference: format!("session:codex:{native}"),
            commit: 1,
            session,
        }),
        parts: vec![ReadPart::Item(ReadItem {
            source: Source::record(format!("session:codex:{native}")),
            kind: ItemKind::Text {
                text: "archived <script>text</script>".into(),
            },
        })],
    }
}
fn complete(sessions: u64) -> ReadComplete {
    ReadComplete {
        footer: ReadFooter::default(),
        counts: Counts {
            sessions,
            ..Default::default()
        },
    }
}
fn manifest(path: &Path) -> Manifest {
    let file = File::open(path.join("manifest.json"))
        .or_else(|_| File::open(path.join("records.jsonl")))
        .unwrap();
    match serde_json::Deserializer::from_reader(file)
        .into_iter::<Record>()
        .next()
        .unwrap()
        .unwrap()
    {
        Record::Manifest(manifest) => *manifest,
        _ => panic!("manifest first"),
    }
}
#[test]
fn bundle_formats_pin_inventory_and_keep_canonical_records() {
    for format in [Format::Json, Format::Jsonl, Format::Markdown, Format::Html] {
        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("bundle");
        let mut sink = Bundle::new(&destination, begin(), format, false).unwrap();
        sink.push(session("a/../b")).unwrap();
        sink.push(session("last")).unwrap();
        sink.finish(complete(2)).unwrap();
        let manifest = manifest(&destination);
        assert_eq!(manifest.counts.sessions, 2);
        let pin = manifest.inventory.unwrap();
        assert_eq!(describe(&destination, &pin.path).unwrap(), pin);
        validate_inventory(&destination).unwrap();
        if format == Format::Json {
            let records: Vec<Record> = serde_json::from_reader(
                File::open(destination.join("codex/a%2F%2E%2E%2Fb.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].reference(), Some("session:codex:a/../b"));
        }
        if format == Format::Jsonl {
            let file = File::open(destination.join("records.jsonl")).unwrap();
            assert_eq!(BufReader::new(file).lines().count(), 3);
            let inventory = fs::read_to_string(destination.join("members.jsonl")).unwrap();
            assert!(!inventory.contains("records.jsonl"));
        }
        if format == Format::Html {
            let html = fs::read_to_string(destination.join("codex/last.html")).unwrap();
            assert!(html.contains("&lt;script&gt;"));
            assert!(!html.contains("<script>"));
        }
    }
}
#[test]
fn final_count_failure_and_disconnect_preserve_previous_bundle_and_staging() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("bundle");
    let mut first = Bundle::new(&destination, begin(), Format::Json, false).unwrap();
    first.push(session("previous")).unwrap();
    first.finish(complete(1)).unwrap();
    let original = fs::read(destination.join("manifest.json")).unwrap();
    let mut failed = Bundle::new(&destination, begin(), Format::Json, false).unwrap();
    let staging = failed.staging().to_owned();
    failed.push(session("new")).unwrap();
    assert!(failed.finish(complete(2)).is_err());
    assert!(staging.exists());
    let retained = ExportItems::in_directory(&staging.join("_staging")).unwrap();
    let recovered: ReadRecord = serde_json::from_reader(
        retained
            .source_item("session:codex:new")
            .unwrap()
            .unwrap()
            .reader,
    )
    .unwrap();
    assert_eq!(recovered.record.reference(), Some("session:codex:new"));
    drop(retained);
    assert_eq!(
        fs::read(destination.join("manifest.json")).unwrap(),
        original
    );
    let mut disconnected = Bundle::new(&destination, begin(), Format::Json, false).unwrap();
    let staging = disconnected.staging().to_owned();
    disconnected.push(session("disconnected")).unwrap();
    drop(disconnected);
    assert!(staging.exists());
    let retained = ExportItems::in_directory(&staging.join("_staging")).unwrap();
    assert!(
        retained
            .source_item("session:codex:disconnected")
            .unwrap()
            .is_some()
    );
    assert_eq!(
        fs::read(destination.join("manifest.json")).unwrap(),
        original
    );
}
#[test]
fn replacement_atomically_exchanges_complete_directories() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("bundle");
    for name in ["old", "new"] {
        let mut sink = Bundle::new(&destination, begin(), Format::Json, false).unwrap();
        let staging = sink.staging().to_owned();
        sink.push(session(name)).unwrap();
        sink.finish(complete(1)).unwrap();
        if name == "new" {
            assert!(staging.join("codex/old.json").exists());
        }
    }
    assert!(destination.join("codex/new.json").exists());
    assert!(!destination.join("codex/old.json").exists());
}
#[test]
fn unrelated_directories_and_symlinks_are_refused() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("unrelated");
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("keep"), b"operator bytes").unwrap();
    assert!(Bundle::new(&destination, begin(), Format::Json, false).is_err());
    assert_eq!(
        fs::read(destination.join("keep")).unwrap(),
        b"operator bytes"
    );
    let link = root.path().join("link");
    std::os::unix::fs::symlink(&destination, &link).unwrap();
    assert!(Bundle::new(&link, begin(), Format::Json, false).is_err());
}
#[test]
fn blob_hash_failure_never_publishes_and_chunk_offsets_are_checked() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("bundle");
    let mut sink = Bundle::new(&destination, begin(), Format::Json, true).unwrap();
    let hash = BlobHash::of(b"correct");
    sink.blob_chunk(hash, 0, b"wrong").unwrap();
    sink.push(ReadRecord {
        record: Record::Blob(BlobRecord {
            reference: format!("blob:{hash}"),
            file: FileDescriptor {
                path: format!("blobs/{hash}"),
                hash,
                size: 7,
            },
        }),
        parts: vec![],
    })
    .unwrap();
    assert!(
        sink.finish(ReadComplete {
            footer: ReadFooter::default(),
            counts: Counts {
                blobs: 1,
                ..Default::default()
            }
        })
        .is_err()
    );
    assert!(!destination.exists());
    let mut sink = Bundle::new(&destination, begin(), Format::Json, true).unwrap();
    assert!(sink.blob_chunk(hash, 1, b"a").is_err());
}
#[test]
fn duplicate_frames_deduplicate_content_and_conflicts_poison_delivery() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("bundle");
    let mut sink = Bundle::new(&destination, begin(), Format::Json, false).unwrap();
    sink.push(session("one")).unwrap();
    sink.push(session("one")).unwrap();
    sink.finish(complete(2)).unwrap();
    assert_eq!(manifest(&destination).counts.sessions, 1);
    let mut sink = Bundle::new(&destination, begin(), Format::Json, false).unwrap();
    sink.push(session("one")).unwrap();
    let mut changed = session("one");
    if let Record::Session(record) = &mut changed.record {
        record.commit = 2;
    }
    assert!(sink.push(changed).is_err());
    assert!(sink.finish(complete(2)).is_err());
}
#[test]
fn long_opaque_names_have_bounded_distinct_components() {
    let native = "x/".repeat(1000);
    let a = SessionId::parse(&format!("codex:{native}")).unwrap();
    let b = SessionId::parse(&format!("codex:{native}b")).unwrap();
    assert!(session_path(&a).len() < 100);
    assert_ne!(session_path(&a), session_path(&b));
}

#[test]
fn sparse_records_union_and_later_partial_does_not_downgrade_content() {
    let message:polylogue_model::Message=serde_json::from_value(serde_json::json!({"id":"codex:message","session":"codex:session","role":"assistant","blocks":[{"type":"text","text":"first"},{"type":"text","text":"second"}]})).unwrap();
    let full = MessageRecord {
        reference: "message:codex:message".into(),
        commit: 2,
        order: 0,
        source_version: message.content_hash().unwrap(),
        block_count: 2,
        block_ordinals: vec![0, 1],
        message,
    };
    let mut first = full.clone();
    first.message.blocks.pop();
    first.block_ordinals.pop();
    let mut second = full.clone();
    second.message.blocks.remove(0);
    second.block_ordinals.remove(0);
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("bundle");
    let mut sink = Bundle::new(&destination, begin(), Format::Json, false).unwrap();
    sink.push(session("session")).unwrap();
    for record in [second, first.clone(), full.clone(), first] {
        sink.push(ReadRecord {
            record: Record::Message(Box::new(record)),
            parts: vec![],
        })
        .unwrap();
    }
    sink.finish(ReadComplete {
        footer: ReadFooter::default(),
        counts: Counts {
            sessions: 1,
            messages: 4,
            ..Default::default()
        },
    })
    .unwrap();
    let records: Vec<Record> =
        serde_json::from_reader(File::open(destination.join("codex/session.json")).unwrap())
            .unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[1], Record::Message(Box::new(full)));
    assert_eq!(manifest(&destination).counts.messages, 1);
}

#[test]
fn valid_blob_chunks_are_pinned_and_unclaimed_bytes_refuse_publication() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("bundle");
    let hash = BlobHash::of(b"complete");
    let mut sink = Bundle::new(&destination, begin(), Format::Jsonl, true).unwrap();
    sink.blob_chunk(hash, 0, b"com").unwrap();
    sink.blob_chunk(hash, 3, b"plete").unwrap();
    sink.push(ReadRecord {
        record: Record::Blob(BlobRecord {
            reference: format!("blob:{hash}"),
            file: FileDescriptor {
                path: format!("blobs/{hash}"),
                hash,
                size: 8,
            },
        }),
        parts: vec![],
    })
    .unwrap();
    sink.finish(ReadComplete {
        footer: ReadFooter::default(),
        counts: Counts {
            blobs: 1,
            ..Default::default()
        },
    })
    .unwrap();
    assert_eq!(
        fs::read(destination.join(format!("blobs/{hash}"))).unwrap(),
        b"complete"
    );
    let mut sink = Bundle::new(&destination, begin(), Format::Jsonl, true).unwrap();
    sink.blob_chunk(hash, 0, b"unclaimed").unwrap();
    assert!(sink.finish(complete(0)).is_err());
    assert_eq!(manifest(&destination).counts.blobs, 1);
}

#[test]
fn injected_member_write_failure_retains_previous_destination() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("bundle");
    let mut sink = Bundle::new(&destination, begin(), Format::Json, false).unwrap();
    sink.push(session("old")).unwrap();
    sink.finish(complete(1)).unwrap();
    let mut sink = Bundle::new(&destination, begin(), Format::Json, false).unwrap();
    let staging = sink.staging().to_owned();
    sink.push(session("new")).unwrap();
    fs::write(staging.join("codex"), b"fault: directory cannot be created").unwrap();
    let error = sink.finish(complete(1)).unwrap_err();
    assert_eq!(error.staging, staging);
    assert!(destination.join("codex/old.json").exists());
    assert!(!destination.join("codex/new.json").exists());
}
