use super::*;
use polylogue_store::{Archive, ArchiveLock, Batch, Change, InputKey, Observation, Writer};
use std::sync::mpsc::channel;

struct Stored {
    writer: Option<Writer>,
    archive: Archive,
    _lock: ArchiveLock,
    _dir: tempfile::TempDir,
}
impl Stored {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let (archive, lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
        let writer = Writer::start(&archive).unwrap();
        Self {
            writer: Some(writer),
            archive,
            _lock: lock,
            _dir: dir,
        }
    }
    fn put(&self, input: &str, format: &str, revision: bool, path: &str, seen: i64, bytes: &[u8]) {
        self.writer
            .as_ref()
            .unwrap()
            .store_bytes(
                self.archive.prepare_bytes(bytes).unwrap(),
                Batch {
                    input: InputKey {
                        id: input.into(),
                        format: format.into(),
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
                        path: path.into(),
                        facts: None,
                        seen_at_ms: seen,
                        import: None,
                        container: None,
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
    fn read(&self, selection: Infer) -> Inference {
        observe(
            &ReadPool::open(&self.archive).unwrap(),
            &Admission::default(),
            selection,
        )
        .unwrap()
    }
}
fn selection(origin: Option<&str>, inputs: &[&str], revision: Option<i64>) -> Infer {
    Infer {
        origin: origin.map(str::to_owned),
        inputs: inputs.iter().map(|s| (*s).into()).collect(),
        revision,
        since: None,
        fields: BTreeSet::new(),
    }
}

fn inventory(root: &Path) -> BTreeMap<std::path::PathBuf, Option<blake3::Hash>> {
    fn visit(
        root: &Path,
        path: &Path,
        files: &mut BTreeMap<std::path::PathBuf, Option<blake3::Hash>>,
    ) {
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            // SQLite's shared memory holds reader coordination, not archive rows.
            if path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with("-shm")
            {
                continue;
            }
            if path.is_dir() {
                files.insert(path.strip_prefix(root).unwrap().to_owned(), None);
                visit(root, &path, files);
            } else {
                files.insert(
                    path.strip_prefix(root).unwrap().to_owned(),
                    Some(blake3::hash(&std::fs::read(path).unwrap())),
                );
            }
        }
    }
    let mut files = BTreeMap::new();
    visit(root, root, &mut files);
    files
}

#[test]
fn schema_tools_no_values_exact_stored_revisions_and_unreadable_records() {
    let stored = Stored::new();
    stored.put(
        "selected",
        "claude-code",
        false,
        "/absent/provider.jsonl",
        10,
        b"{\"id\":\"private scalar\"}\nmalformed\n",
    );
    stored.put(
        "selected",
        "claude-code",
        true,
        "/absent/provider.jsonl",
        20,
        b"{\"count\":2}\n",
    );
    let before = inventory(&stored.archive.root);
    let report = stored.read(selection(
        Some("claude-code"),
        &["selected", "missing"],
        Some(1),
    ));
    assert_eq!(inventory(&stored.archive.root), before);
    let schema = &report.origins["claude-code"];
    assert_eq!(
        (
            schema.records,
            schema.unreadable_records,
            schema.unreadable_inputs
        ),
        (1, 1, 0)
    );
    assert!(schema.fields.contains_key("/id"));
    assert!(!schema.fields.contains_key("/count"));
    assert_eq!(schema.selection[0].revision, Some(1));
    assert_eq!(report.selection.missing_revisions, ["missing"]);
    assert_eq!(report.selection.selected_inputs, 1);
    assert_eq!(report.selection.read_bytes, report.selection.selected_bytes);
    let text = serde_json::to_string(&report).unwrap();
    for secret in ["private scalar", "/absent", "provider.jsonl"] {
        assert!(!text.contains(secret));
    }
    let latest = stored.read(selection(Some("claude-code"), &["selected"], None));
    assert_eq!(latest.origins["claude-code"].selection[0].revision, Some(2));
    assert!(latest.origins["claude-code"].fields.contains_key("/count"));
    let wrong = stored.read(selection(Some("codex"), &["selected"], None));
    assert_eq!(
        wrong.origins["claude-code"].selection[0]
            .unreadable
            .as_deref(),
        Some("different_origin")
    );
    let absent = stored.read(selection(Some("claude-code"), &["selected"], Some(8)));
    assert_eq!(absent.selection.missing_revisions, ["selected"]);
    assert_eq!(absent.selection.selected_inputs, 0);
}

#[test]
fn schema_selection_continues_every_page_and_time_bound_is_observation_time() {
    let stored = Stored::new();
    for i in 0..130 {
        stored.put(
            &format!("input-{i:03}"),
            "claude-code",
            false,
            "/absent/session.jsonl",
            100,
            b"{\"a\":null}\n",
        );
    }
    stored.put(
        "latest",
        "claude-code",
        false,
        "/absent/session.jsonl",
        200,
        b"{\"later\":true}\n",
    );
    let all = stored.read(selection(Some("claude-code"), &[], None));
    assert_eq!(all.origins["claude-code"].selection.len(), 131);
    assert_eq!(all.origins["claude-code"].records, 131);
    assert!(!all.origins["claude-code"].sample);
    let mut since = selection(Some("claude-code"), &[], None);
    since.since = Some("1970-01-01T00:00:00.150Z".into());
    let recent = stored.read(since);
    assert_eq!(recent.origins["claude-code"].selection.len(), 1);
    assert_eq!(recent.origins["claude-code"].selection[0].input, "latest");
    assert!(recent.origins["claude-code"].sample);
}

#[test]
fn compressed_containers_and_decoded_revisions_use_their_stored_framing() {
    let stored = Stored::new();
    let plain = b"{\"native\":\"private value\"}\n";
    let compressed = zstd::stream::encode_all(&plain[..], 1).unwrap();
    stored.put(
        "container",
        "codex-compressed",
        false,
        "/absent/session.jsonl.zst",
        1,
        &compressed,
    );
    stored.put(
        "native",
        "codex",
        false,
        "/absent/session.jsonl.zst",
        1,
        plain,
    );
    stored.put(
        "db",
        "tool-database",
        false,
        "/absent/tool.sqlite",
        1,
        b"{\"schema\":{\"empty\":\"private SQL\"}}\n",
    );
    stored.put(
        "binary",
        "zip-member",
        false,
        "/absent/image.png",
        1,
        b"binary bytes",
    );
    let report = stored.read(selection(None, &[], None));
    for origin in ["codex", "codex-compressed"] {
        assert_eq!(report.origins[origin].records, 1);
        assert!(report.origins[origin].fields.contains_key("/native"));
        assert_eq!(report.origins[origin].unreadable_inputs, 0);
    }
    assert!(report.origins["tool-database"].tables["empty"].unobserved_columns);
    assert_eq!(report.origins["zip-member"].unreadable_inputs, 1);
    assert!(!serde_json::to_string(&report).unwrap().contains("private"));
}

#[test]
fn primary_inference_reads_owned_sinex_extents_without_a_standalone_blob_copy() {
    let mut stored = Stored::new();
    stored.put(
        "primary-input",
        "claude-code",
        false,
        "/absent/provider.jsonl",
        1,
        b"{\"kept\":\"private bytes\"}\n",
    );
    drop(stored.writer.take());
    let root = stored._dir.path().join("primary");
    let sinex = stored._dir.path().join("sinex");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&sinex).unwrap();
    let cache = rusqlite::Connection::open(root.join("index.db")).unwrap();
    polylogue_store::schema::ensure_primary_index(&cache).unwrap();
    cache
        .execute(
            "ATTACH ?1 AS original",
            [stored.archive.archive_db().to_str().unwrap()],
        )
        .unwrap();
    for table in [
        "blob",
        "segment",
        "input",
        "revision",
        "extent",
        "observation",
    ] {
        cache
            .execute_batch(&format!(
                "INSERT INTO source_{table} SELECT * FROM original.{table}"
            ))
            .unwrap();
    }
    drop(cache);
    std::fs::rename(stored.archive.root.join("segments"), sinex.join("segments")).unwrap();
    let archive = Archive::on_sinex(&root, &sinex);
    let pool = ReadPool::open(&archive).unwrap();
    let report = observe(
        &pool,
        &Admission::default(),
        selection(Some("claude-code"), &[], None),
    )
    .unwrap();
    assert_eq!(
        report.origins["claude-code"].fields["/kept"].types["string"],
        1
    );
    assert_eq!(report.origins["claude-code"].records, 1);
    assert!(!root.join("archive.db").exists());
    assert!(!root.join("segments").exists());
}

#[test]
fn idle_inference_yields_at_stream_reads_and_cancellation_releases_its_lease() {
    let admission = Admission::default();
    let foreground = admission.foreground();
    let cancellation = polylogue_store::read::limit::ReadCancellation::default();
    let limit = ReadLimit::without_deadline(cancellation.clone());
    std::thread::scope(|scope| {
        let (started, waiting) = channel();
        let (done, finished) = channel();
        let admission = &admission;
        scope.spawn(move || {
            limit.scope(|| {
                started.send(()).unwrap();
                let result = admission.infer();
                done.send(result.is_ok()).unwrap();
            })
        });
        waiting.recv().unwrap();
        assert!(finished.recv_timeout(Duration::from_millis(40)).is_err());
        cancellation.cancel();
        assert!(!finished.recv_timeout(Duration::from_secs(2)).unwrap());
    });
    drop(foreground);
    let _lease = admission.infer().unwrap();
    std::thread::scope(|scope| {
        let (first, started) = channel();
        let (next, proceed) = channel();
        let (done, finished) = channel();
        let admission_ref = &admission;
        scope.spawn(move || {
            let count = std::cell::Cell::new(0);
            let mut stream = IdleRead {
                source: &b"abcdef"[..],
                admission: admission_ref,
                read: &count,
            };
            let mut byte = [0];
            stream.read_exact(&mut byte).unwrap();
            first.send(byte).unwrap();
            proceed.recv().unwrap();
            stream.read_exact(&mut byte).unwrap();
            done.send((byte, count.get())).unwrap();
        });
        assert_eq!(started.recv().unwrap(), [b'a']);
        let foreground = admission.foreground();
        next.send(()).unwrap();
        assert!(finished.recv_timeout(Duration::from_millis(40)).is_err());
        drop(foreground);
        assert_eq!(
            finished.recv_timeout(Duration::from_secs(2)).unwrap(),
            ([b'b'], 2)
        );
    });
}

#[test]
fn schema_infer_and_unread_use_the_real_daemon_socket_and_read_only_requests() {
    let daemon = crate::test_daemon::start("private scalar");
    let provider = daemon.home.join(".claude/projects/-p/s1.jsonl");
    let mut row: serde_json::Value = serde_json::from_str(&polylogue_testkit::claude_user(
        "u2",
        "s1",
        "private second",
    ))
    .unwrap();
    row["unknown_nested"] = serde_json::json!({"unused":"private unseen"});
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&provider)
        .unwrap()
        .write_all(format!("{row}\n").as_bytes())
        .unwrap();
    let imported = super::super::service::handle(
        super::super::Request::Import {
            request_id: "schema-extra-row".into(),
            paths: vec![provider.clone()],
        },
        &daemon.daemon,
    );
    assert!(matches!(imported, Response::Ok { .. }), "{imported:?}");
    daemon.wait_for(|status| status["store"]["messages"] == 2);
    std::fs::remove_file(provider).unwrap();
    let request = super::super::Request::SchemaInfer {
        selection: selection(Some("claude-code"), &[], None),
    };
    assert!(request.read_only());
    assert!(super::super::execution::request_deadline(&request, None).is_none());
    let socket = daemon.daemon.archive.root.join("daemon.sock");
    let response = crate::client::send(&socket, &request).unwrap();
    let Response::Ok { data, .. } = response else {
        panic!("{response:?}");
    };
    let report: Inference = serde_json::from_value(data).unwrap();
    assert_eq!(report.origins["claude-code"].records, 2);
    assert!(
        !serde_json::to_string(&report)
            .unwrap()
            .contains("private scalar")
    );
    let unread = super::super::Request::SchemaUnread {
        origin: Some("claude-code".into()),
    };
    assert!(unread.read_only());
    let response = crate::client::send(&socket, &unread).unwrap();
    let Response::Ok { data, .. } = response else {
        panic!("{response:?}");
    };
    let shapes = data["shapes"].as_array().unwrap();
    assert!(
        shapes
            .iter()
            .any(|s| s["path"] == "/unknown_nested/unused"
                && s["unused_count"].as_u64().unwrap() > 0)
    );
    assert!(!shapes.iter().any(|s| s["path"] == "/type"));
    assert!(
        !serde_json::to_string(&data)
            .unwrap()
            .contains("private unseen")
    );
    assert!(
        data["shapes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["unused_count"].as_u64().unwrap() > 0)
    );
    assert!(!data["coverage"].as_array().unwrap().is_empty());
}
