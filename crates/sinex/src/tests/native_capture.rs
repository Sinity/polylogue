use super::client;
use crate::Sinex;
use polylogue_engine::Engine;
use polylogue_store::primary::{
    Capture, RawAcquisitionQuery, RawRange, SourceGeneration, SourceSelector,
};
use polylogue_store::{Archive, ArchiveLock, ReadPool};
use polylogue_testkit::ClaudeHome;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

const CHUNK: u64 = 64;

fn open(home: &ClaudeHome) -> (Engine, Archive, ArchiveLock, Arc<Sinex>) {
    let store = home.dir.path().join("sdk");
    let sdk = Arc::new(client(&store));
    let (archive, lock) = Archive::open_primary(&home.root, &store).unwrap();
    let engine = Engine::on_sinex(
        home.config(CHUNK),
        archive.clone(),
        sdk.clone(),
        polylogue_testkit::stopped_attempt(),
    )
    .unwrap();
    (engine, archive, lock, sdk)
}

fn selector(path: &Path) -> SourceSelector {
    SourceSelector {
        origin: "claude-code".into(),
        machine: "test-machine".into(),
        path_bytes: path.as_os_str().as_encoded_bytes().to_vec(),
    }
}

fn retained(sdk: &Sinex, generation: &SourceGeneration) -> (Vec<u8>, usize) {
    let mut headers: Vec<_> = sdk
        .pending_captures()
        .unwrap()
        .into_iter()
        .filter_map(|header| {
            let Capture::Input(input) = header.captured else {
                return None;
            };
            (input.generation.as_ref() == Some(generation)).then_some((header.identity, input))
        })
        .collect();
    headers.sort_by_key(|(_, input)| input.raw_range.unwrap().start);
    let count = headers.len();
    let mut result = Vec::new();
    for (identity, input) in headers {
        let range = input.raw_range.unwrap();
        assert_eq!(range.start, result.len() as u64);
        let body = sdk.capture_body(&identity).unwrap();
        let mut bytes = Vec::new();
        body.descriptor
            .open(body.file, None, Default::default())
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(bytes.len() as u64, range.length().unwrap());
        assert_eq!(polylogue_model::BlobHash::of(&bytes), input.hash);
        result.extend(bytes);
    }
    (result, count)
}

fn assert_complete(sdk: &Sinex, archive: &Archive, path: &Path, expected: &[u8]) -> usize {
    let count = assert_custody_complete(sdk, path, expected);
    let raw = sdk
        .pending_source(&selector(path), "")
        .unwrap()
        .raw
        .unwrap();
    let Capture::Input(header) = &raw.captured else {
        panic!("a native raw capture must have its original input header");
    };
    // Durable SDK custody has not created a confirmed source or parse cursor.
    let reads = ReadPool::open(archive).unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .input_progress(&header.rows.input)
            .unwrap()
            .is_none()
    );
    count
}

fn assert_custody_complete(sdk: &Sinex, path: &Path, expected: &[u8]) -> usize {
    let raw = sdk
        .pending_source(&selector(path), "")
        .unwrap()
        .raw
        .unwrap();
    let Capture::Input(header) = &raw.captured else {
        panic!("a native raw capture must have its original input header");
    };
    let generation = header.generation.clone().unwrap();
    let cursor = sdk
        .raw_acquisition_cursor(&RawAcquisitionQuery {
            generation: generation.clone(),
            anchor: Some(raw.identity),
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        cursor.range(),
        RawRange {
            start: 0,
            end: expected.len() as u64,
        }
    );
    assert_eq!(cursor.known_total(), Some(expected.len() as u64));
    assert!(cursor.is_complete_stream());
    let (bytes, count) = retained(sdk, &generation);
    assert_eq!(bytes, expected);
    count
}

fn assert_custody_waits_for_confirmation(
    sdk: &Sinex,
    archive: &Archive,
    path: &Path,
    identity: &str,
) {
    let source = sdk.pending_source(&selector(path), "").unwrap();
    assert_eq!(
        source.status,
        polylogue_store::primary::SourceStatus::MissingEvidence
    );
    // The accepted private head survives; it is not a confirmed source grant.
    assert_eq!(source.head.as_ref().unwrap().identity, identity);
    assert_eq!(source.raw.as_ref().unwrap().identity, identity);
    assert!(
        sdk.queue()
            .unwrap()
            .capture_group(identity)
            .unwrap()
            .is_none()
    );
    // While Sinex is unreachable, reads answer from the index as it stands.
    let reads = ReadPool::open(archive).unwrap();
    assert_eq!(
        reads.reader().unwrap().confirmed_checkpoint().unwrap(),
        None
    );
}

#[test]
fn database_completion_preserves_pending_provider_files_in_the_same_batch() {
    let home = ClaudeHome::new();
    home.append(
        "pending.jsonl",
        &polylogue_testkit::claude_user("u1", "pending", "owned Claude bytes"),
    );
    let codex = home.home.join(".codex");
    std::fs::create_dir_all(codex.join("sessions")).unwrap();
    std::fs::copy(
        polylogue_testkit::morrow_dir().join("codex/state_5.sqlite"),
        codex.join("state_5.sqlite"),
    )
    .unwrap();
    std::fs::write(codex.join("sessions/rollout-pending.jsonl"),
        "{\"type\":\"session_meta\",\"payload\":{\"id\":\"pending-native\"}}\n{\"type\":\"response_item\",\"payload\":{\"id\":\"m1\",\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"owned Codex bytes\"}]}}\n"
    ).unwrap();
    let (mut engine, _archive, _lock, sdk) = open(&home);
    let pass = engine.run_pass().unwrap();
    assert!(
        pass.errors
            .iter()
            .all(|error| error.starts_with("confirmed stream:")
                && error.contains("Connection refused")),
        "{pass:?}"
    );
    assert_eq!(
        pass.full_scans, 1,
        "database completion revisits only this batch's paths"
    );
    assert!(
        pass.capture_pending,
        "database completion cannot clear pending provider work: {pass:?}"
    );
    let captures = sdk.pending_captures().unwrap();
    assert!(captures.iter().any(|capture| matches!(&capture.captured, Capture::Input(input) if input.rows.format == "claude-code")));
    assert!(captures.iter().any(|capture| matches!(&capture.captured, Capture::Input(input) if input.rows.input == "codex:rollout:pending-native")));
    let identities = |captures: Vec<polylogue_store::primary::PendingCapture>| {
        let mut ids = captures
            .into_iter()
            .map(|capture| capture.identity)
            .collect::<Vec<_>>();
        ids.sort();
        ids
    };
    let original = identities(captures);
    let repeated = engine.run_pass().unwrap();
    assert!(repeated.capture_pending, "{repeated:?}");
    assert_eq!(
        repeated.bytes_read, 0,
        "owned provider bytes are not read again"
    );
    assert_eq!(identities(sdk.pending_captures().unwrap()), original);
}

#[test]
fn quiet_offline_native_file_finishes_all_chunks_and_reopens_without_provider_prefix_reads() {
    let home = ClaudeHome::new();
    let text = polylogue_testkit::claude_user(
        "quiet",
        "offline-native",
        &"quiet native bytes ".repeat(80),
    );
    home.append("quiet.jsonl", &text);
    let path = home.file("quiet.jsonl");
    let (mut engine, archive, lock, sdk) = open(&home);
    let pass = engine.run_pass().unwrap();
    assert!(
        pass.errors
            .iter()
            .all(|error| error.starts_with("confirmed stream:")
                && error.contains("Connection refused")),
        "{pass:?}"
    );
    assert!(pass.capture_pending);
    assert_eq!(pass.bytes_read, text.len() as u64);
    let count = assert_complete(&sdk, &archive, &path, text.as_bytes());
    assert!(count > 2);
    drop(engine);
    drop(lock);
    drop(sdk);

    let (mut engine, archive, _lock, sdk) = open(&home);
    let quiet = engine.run_pass().unwrap();
    assert!(
        quiet
            .errors
            .iter()
            .all(|error| error.starts_with("confirmed stream:")
                && error.contains("Connection refused")),
        "{quiet:?}"
    );
    assert!(quiet.capture_pending);
    assert_eq!(quiet.bytes_read, 0);
    assert_eq!(
        assert_complete(&sdk, &archive, &path, text.as_bytes()),
        count
    );
}

#[test]
fn a_scope_reset_keeps_disappeared_unconfirmed_native_custody_in_the_same_engine() {
    let home = ClaudeHome::new();
    let text = polylogue_testkit::claude_user(
        "one",
        "scope-custody",
        &"retained source bytes ".repeat(40),
    );
    home.append("scope-custody.jsonl", &text);
    let path = home.file("scope-custody.jsonl");
    let (mut engine, archive, _lock, sdk) = open(&home);
    let first = engine.run_pass().unwrap();
    assert!(
        first
            .errors
            .iter()
            .all(|error| error.starts_with("confirmed stream:")
                && error.contains("Connection refused")),
        "{first:?}"
    );
    assert!(first.capture_pending);
    assert_eq!(first.bytes_read, text.len() as u64);
    let count = assert_complete(&sdk, &archive, &path, text.as_bytes());
    let identities = || {
        sdk.pending_captures()
            .unwrap()
            .into_iter()
            .map(|capture| capture.identity)
            .collect::<std::collections::BTreeSet<_>>()
    };
    let before = identities();
    let original_scope = archive.scope_token().unwrap();
    // Reset after actual receiving custody, with no provider left to reacquire.
    std::fs::remove_file(&path).unwrap();
    archive.advance_generation_for_test().unwrap();
    assert!(matches!(
        original_scope.check(),
        Err(polylogue_store::StoreError::ScopeChanged)
    ));
    let resumed = engine.run_pass().unwrap();
    assert!(
        resumed
            .errors
            .iter()
            .all(|error| error.starts_with("confirmed stream:")
                && error.contains("Connection refused")),
        "{resumed:?}"
    );
    assert!(resumed.capture_pending);
    assert_eq!((resumed.bytes_read, resumed.bytes_stored), (0, 0));
    assert_eq!(identities(), before);
    assert_eq!(
        assert_complete(&sdk, &archive, &path, text.as_bytes()),
        count
    );
}

#[test]
fn interrupted_native_capture_child() {
    let Some(root) = std::env::var_os("POLYLOGUE_NATIVE_CAPTURE_ROOT") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let store = root.join("sdk");
    let sdk = Arc::new(client(&store));
    let (archive, _lock) = Archive::open_primary(&root.join("archive"), &store).unwrap();
    let mut config = polylogue_engine::Config::new(
        root.join("home"),
        Default::default(),
        "test-machine".into(),
        "test".into(),
        1,
        polylogue_model::WallClock::fixed("2026-10-07T10:00:00Z".parse().unwrap()),
    )
    .unwrap();
    config.chunk = CHUNK;
    let mut engine =
        Engine::on_sinex(config, archive, sdk, polylogue_testkit::stopped_attempt()).unwrap();
    engine.run_pass().unwrap();
    panic!("the actual acquisition publication boundary was not reached");
}

#[test]
fn restart_after_first_published_chunk_uses_immutable_sdk_bytes_before_the_next_provider_read() {
    let home = ClaudeHome::new();
    let text = polylogue_testkit::claude_user(
        "restart",
        "offline-restart",
        &"retained prefix ".repeat(80),
    );
    home.append("restart.jsonl", &text);
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "tests::native_capture::interrupted_native_capture_child",
            "--nocapture",
        ])
        .env("POLYLOGUE_NATIVE_CAPTURE_ROOT", home.dir.path())
        .env("POLYLOGUE_SPOOL_CRASH", "acquisition_published")
        .status()
        .unwrap();
    assert_eq!(child.code(), Some(77));
    let path = home.file("restart.jsonl");
    let (mut engine, archive, _lock, sdk) = open(&home);
    let first = sdk
        .pending_source(&selector(&path), "")
        .unwrap()
        .raw
        .unwrap();
    let Capture::Input(first) = first.captured else {
        panic!("the published first member is a provider capture");
    };
    assert_eq!(
        first.raw_range,
        Some(RawRange {
            start: 0,
            end: CHUNK
        })
    );
    let pass = engine.run_pass().unwrap();
    assert!(
        pass.errors
            .iter()
            .all(|error| error.starts_with("confirmed stream:")
                && error.contains("Connection refused")),
        "{pass:?}"
    );
    assert_eq!(pass.bytes_read, text.len() as u64 - CHUNK);
    assert!(assert_complete(&sdk, &archive, &path, text.as_bytes()) > 2);
}

#[test]
fn offline_empty_native_file_has_an_actual_body_and_eof_with_no_confirmed_source() {
    let home = ClaudeHome::new();
    home.append("empty.jsonl", "");
    let path = home.file("empty.jsonl");
    let (mut engine, archive, lock, sdk) = open(&home);
    let original_scope = archive.scope_token().unwrap();
    assert_eq!(
        ReadPool::open(&archive)
            .unwrap()
            .reader()
            .unwrap()
            .status()
            .unwrap()
            .inputs,
        0
    );
    let pass = engine.run_pass().unwrap();
    assert!(
        pass.errors
            .iter()
            .all(|error| error.starts_with("confirmed stream:")
                && error.contains("Connection refused")),
        "{pass:?}"
    );
    assert!(pass.capture_pending);
    assert_eq!(assert_custody_complete(&sdk, &path, b""), 1);
    let pending = sdk.pending_source(&selector(&path), "").unwrap();
    let private_head = pending.head.as_ref().unwrap().identity.clone();
    let raw = pending.raw.unwrap();
    let Capture::Input(input) = &raw.captured else {
        panic!("input capture expected");
    };
    assert!(input.rows.extent.is_none());
    assert_eq!(input.rows.revision.as_ref().unwrap().length, 0);
    assert_eq!(input.rows.observation.as_ref().unwrap().length, Some(0));
    let group = input.group.as_ref().unwrap().identity.clone();
    let identity = raw.identity;
    assert_eq!(private_head, identity);
    // A real encoded empty body can receive a material acknowledgment without
    // creating a logical source extent or a SourceAccepted position.
    let physical = crate::tests::extent(&home.dir.path().join("sdk"), b"");
    assert_eq!(physical.decoded_len, 0);
    assert!(physical.len > 0);
    {
        let queue = sdk.queue().unwrap();
        let sinex_protocol::Ingress::Material { material, .. } =
            queue.decode_frame(&identity).unwrap()
        else {
            panic!("provider material expected");
        };
        let receipt = serde_json::from_value(serde_json::json!({
            "number":1,"frame":identity,"material":material.id,"material_extents":[physical.encode()],
            "admitted":[],"outcome":{},"refusal":null,"commit_seq":null
        })).unwrap();
        queue.acknowledge(receipt).unwrap();
    }
    drop(engine);
    // Run the real confirmed-consumer path in writer order. An unreachable
    // Sinex fails the read without changing the read scope.
    let writer = polylogue_store::Writer::start_on_sinex(&archive, sdk.clone()).unwrap();
    assert!(matches!(
        writer.synchronize_primary(),
        Err(polylogue_store::StoreError::Sinex(_))
    ));
    original_scope.check().unwrap();
    archive.scope_token().unwrap();
    assert_custody_waits_for_confirmation(&sdk, &archive, &path, &identity);
    assert_eq!(assert_custody_complete(&sdk, &path, b""), 1);
    drop(writer);
    drop(lock);
    drop(sdk);
    drop(original_scope);
    drop(archive);
    // Reopen uses the retained acquisition, even after the provider is gone.
    std::fs::remove_file(&path).unwrap();
    let (mut engine, archive, _lock, sdk) = open(&home);
    // A pass while Sinex is unreachable is not refused by the read scope.
    engine.run_pass().unwrap();
    assert_custody_waits_for_confirmation(&sdk, &archive, &path, &identity);
    assert_eq!(assert_custody_complete(&sdk, &path, b""), 1);
    let pending = sdk.pending_captures().unwrap();
    let retained = pending
        .iter()
        .find(|pending| pending.identity == identity)
        .unwrap();
    assert_eq!(retained.material.as_ref().unwrap().length, 0);
    assert_eq!(
        retained.material.as_ref().unwrap().extents,
        vec![physical.encode()]
    );
    let paged = sdk.pending_group_headers(&group, None, 1).unwrap();
    assert_eq!(paged.headers.len(), 1);
    assert_eq!(paged.headers[0].identity, identity);
    assert_eq!(paged.headers[0].material, retained.material);
    let source = sdk.pending_source(&selector(&path), "").unwrap();
    assert_eq!(
        source.status,
        polylogue_store::primary::SourceStatus::MissingEvidence
    );
    assert_eq!(source.raw.unwrap().material, retained.material);
}

fn acquired_file(
    home: &ClaudeHome,
    archive: &Archive,
    writer: &polylogue_store::Writer,
    path: &Path,
    route: polylogue_store::primary::AcquisitionRoute,
) -> polylogue_store::staging::PreparedBatch {
    use polylogue_store::primary::{
        AcquisitionReceipt, AcquisitionRoute, FileSnapshot, FileTime, RawStream, RawStreamRole,
    };
    use std::os::unix::fs::MetadataExt;
    let source = selector(path);
    let basis = writer.source_snapshot(source.clone(), "").unwrap().basis;
    let mut file = std::fs::File::open(path).unwrap();
    let metadata = file.metadata().unwrap();
    let facts = polylogue_store::FileFacts::of(&metadata);
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    assert_eq!(
        facts,
        polylogue_store::FileFacts::of(&file.metadata().unwrap())
    );
    assert_eq!(
        facts,
        polylogue_store::FileFacts::of(&std::fs::metadata(path).unwrap())
    );
    let prepared = archive.prepare_bytes(bytes.as_slice()).unwrap();
    let hash = prepared.hash();
    let size = prepared.len();
    let imported = route == AcquisitionRoute::ExplicitImport;
    let input = polylogue_store::InputKey {
        id: if imported {
            hash.to_string()
        } else {
            path.file_stem().unwrap().to_str().unwrap().into()
        },
        format: source.origin.clone(),
        arrival_rank: if imported { 2 } else { 0 },
    };
    let range = RawRange {
        start: 0,
        end: size,
    };
    let generation = (!imported).then(|| SourceGeneration::File {
        stream: RawStream {
            source: source.clone(),
            input: input.clone(),
            role: RawStreamRole::ProviderBytes,
            namespace: None,
        },
        observed: FileSnapshot {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.size(),
            modified: FileTime {
                seconds: metadata.mtime(),
                nanos: metadata.mtime_nsec().try_into().unwrap(),
            },
            changed: FileTime {
                seconds: metadata.ctime(),
                nanos: metadata.ctime_nsec().try_into().unwrap(),
            },
        },
        window: range,
    });
    let clock = home.config(CHUNK).clock;
    let at = clock.now();
    polylogue_store::staging::PreparedBatch {
        batch: polylogue_store::Batch {
            input,
            change: polylogue_store::Change::First,
            observation: Some(polylogue_store::Observation {
                machine: source.machine.clone(),
                host: "test".into(),
                path: path.display().to_string(),
                path_bytes: source.path_bytes.clone(),
                facts: Some(facts),
                seen_at_ms: at.as_millisecond(),
                import: imported.then(|| polylogue_store::ImportMember {
                    version: 1,
                    import_id: "synthetic-package".into(),
                    member: path.file_name().unwrap().to_str().unwrap().into(),
                    authorship: Some(polylogue_model::authorship::Authorship {
                        author: polylogue_model::Author::Script("package-fixture".into()),
                        via: polylogue_model::authorship::Channel::Import,
                    }),
                    format: Some("claude-code".into()),
                }),
                container: None,
            }),
            continues: false,
            new_blob: Some(polylogue_store::NewBlob { hash, size }),
            blobs: Vec::new(),
            length: size,
            parse: Vec::new(),
            release: Default::default(),
            acquisition: Some(AcquisitionReceipt {
                version: AcquisitionReceipt::VERSION,
                route,
                source,
                range,
                hash,
                observed_at: at.into(),
                generation,
                raw_total: (!imported).then_some(size),
                raw_predecessor: None,
            }),
            basis: Some(basis),
            required_disposition: None,
        },
        files: vec![prepared],
    }
}

#[test]
fn a_raw_only_file_does_not_freeze_an_independent_file_in_the_same_writer_flush() {
    use polylogue_store::primary::{AcquisitionRoute, CaptureDisposition};
    let home = ClaudeHome::new();
    home.append(
        "file-a.jsonl",
        &polylogue_testkit::claude_user("a", "boundary-a", "A"),
    );
    home.append(
        "file-b.jsonl",
        &polylogue_testkit::claude_user("b", "boundary-b", "B"),
    );
    let store = home.dir.path().join("sdk");
    let sdk = Arc::new(client(&store));
    let (archive, _lock) = Archive::open_primary(&home.root, &store).unwrap();
    let writer = polylogue_store::Writer::start_on_sinex(&archive, sdk.clone()).unwrap();
    let mut a = acquired_file(
        &home,
        &archive,
        &writer,
        &home.file("file-a.jsonl"),
        AcquisitionRoute::NativeCapture,
    );
    let b = acquired_file(
        &home,
        &archive,
        &writer,
        &home.file("file-b.jsonl"),
        AcquisitionRoute::NativeCapture,
    );
    a.batch.required_disposition = Some(CaptureDisposition::UnresolvedBasisConflict);
    let results = writer
        .sender()
        .unwrap()
        .store_group(vec![a, b], Arc::new(archive.blobs().unwrap()))
        .unwrap();
    assert_eq!(results.len(), 2);
    let Err(polylogue_store::StoreError::UnresolvedBasisConflict { custody }) = &results[0] else {
        panic!("A must retain raw-only custody: {results:?}")
    };
    assert_eq!(custody.count, 1);
    assert!(matches!(
        &results[1],
        Err(polylogue_store::StoreError::PrimaryPending { .. })
    ));
    assert!(
        sdk.pending_source(&selector(&home.file("file-a.jsonl")), "")
            .unwrap()
            .head
            .is_none()
    );
    assert!(
        sdk.pending_source(&selector(&home.file("file-b.jsonl")), "")
            .unwrap()
            .raw
            .is_some()
    );
    let page = sdk
        .pending_group_headers(&custody.identity, None, 32)
        .unwrap();
    assert_eq!(
        page.disposition,
        Some(CaptureDisposition::UnresolvedBasisConflict)
    );
    assert_eq!(page.headers.len(), 1);
}

#[test]
fn an_explicit_package_keeps_its_member_custody_atomic() {
    use polylogue_store::primary::{AcquisitionRoute, CaptureDisposition};
    let home = ClaudeHome::new();
    home.append(
        "package-a.jsonl",
        &polylogue_testkit::claude_user("a", "package-a", "A"),
    );
    home.append(
        "package-b.jsonl",
        &polylogue_testkit::claude_user("b", "package-b", "B"),
    );
    let store = home.dir.path().join("sdk");
    let sdk = Arc::new(client(&store));
    let (archive, _lock) = Archive::open_primary(&home.root, &store).unwrap();
    let writer = polylogue_store::Writer::start_on_sinex(&archive, sdk.clone()).unwrap();
    let mut a = acquired_file(
        &home,
        &archive,
        &writer,
        &home.file("package-a.jsonl"),
        AcquisitionRoute::ExplicitImport,
    );
    let b = acquired_file(
        &home,
        &archive,
        &writer,
        &home.file("package-b.jsonl"),
        AcquisitionRoute::ExplicitImport,
    );
    a.batch.required_disposition = Some(CaptureDisposition::UnresolvedBasisConflict);
    let results = writer
        .sender()
        .unwrap()
        .store_group(vec![a, b], Arc::new(archive.blobs().unwrap()))
        .unwrap();
    assert_eq!(results.len(), 2);
    let receipts: Vec<_> = results
        .into_iter()
        .map(|result| match result {
            Err(polylogue_store::StoreError::UnresolvedBasisConflict { custody }) => custody,
            other => panic!("a package must retain every member together: {other:?}"),
        })
        .collect();
    assert_eq!(receipts[0], receipts[1]);
    assert_eq!(receipts[0].count, 2);
    let page = sdk
        .pending_group_headers(&receipts[0].identity, None, 32)
        .unwrap();
    assert_eq!(
        page.disposition,
        Some(CaptureDisposition::UnresolvedBasisConflict)
    );
    assert_eq!(page.headers.len(), 2);
    assert!(
        sdk.pending_source(&selector(&home.file("package-b.jsonl")), "")
            .unwrap()
            .head
            .is_none()
    );
}

#[test]
fn a_finished_database_worker_keeps_its_original_custody_pending_across_batches() {
    let home = ClaudeHome::new();
    let db = home.home.join(".codex/state_5.sqlite");
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    std::fs::copy(
        polylogue_testkit::morrow_dir().join("codex/state_5.sqlite"),
        &db,
    )
    .unwrap();
    let (mut engine, _archive, _lock, sdk) = open(&home);
    let first = engine.run_pass().unwrap();
    assert!(first.capture_pending, "{first:?}");
    let identities = || {
        let mut captures: Vec<_> = sdk
            .pending_captures()
            .unwrap()
            .into_iter()
            .map(|capture| {
                let Capture::Input(input) = capture.captured else {
                    panic!("database input capture")
                };
                assert_eq!(input.rows.format, "codex-database");
                (capture.identity, input.hash, input.length)
            })
            .collect();
        captures.sort_by(|a, b| a.0.cmp(&b.0));
        captures
    };
    let original = identities();
    assert!(!original.is_empty());
    std::fs::remove_file(&db).unwrap();
    for _ in 0..2 {
        let pending = engine.run_pass().unwrap();
        assert!(
            pending.capture_pending,
            "durable database bytes still await confirmation: {pending:?}"
        );
        assert!(
            pending
                .errors
                .iter()
                .all(|error| error.starts_with("confirmed stream:")
                    && error.contains("Connection refused")),
            "{pending:?}"
        );
        assert_eq!(pending.bytes_read, 0);
        assert_eq!(pending.bytes_stored, 0);
        assert_eq!(identities(), original);
    }
}

#[test]
fn a_deferred_database_read_error_preserves_another_files_confirmation_wait() {
    use polylogue_model::BlobHash;
    use polylogue_store::Writer;
    use polylogue_store::primary::{Confirmed, ConfirmedChange, Material, Product, ReplayControl};

    for pending_file in [false, true] {
        let home = ClaudeHome::new();
        let db = home.home.join(".codex/state_5.sqlite");
        std::fs::create_dir_all(db.parent().unwrap()).unwrap();
        std::fs::copy(
            polylogue_testkit::morrow_dir().join("codex/state_5.sqlite"),
            &db,
        )
        .unwrap();
        if pending_file {
            home.append(
                "pending-with-error.jsonl",
                &polylogue_testkit::claude_user("u1", "pending", "retained despite database error"),
            );
        }
        let store = home.dir.path().join("sdk");
        let sdk = Arc::new(client(&store));
        let (archive, _lock) = Archive::open_primary(&home.root, &store).unwrap();
        let writer = Writer::start_on_sinex(&archive, sdk.clone()).unwrap();
        let control = |control| Confirmed {
            native: None,
            position: 1,
            output_index: 0,
            event: String::new(),
            change: ConfirmedChange::Control(Box::new(control)),
        };
        writer
            .apply_confirmed(vec![control(ReplayControl::Reset {
                scope: "database-error-scope".into(),
            })])
            .unwrap();
        let bytes = b"{\"db\":\"state_5.sqlite\",\"schema\":{}}\n";
        let physical = super::extent(&store, bytes);
        let hash = BlobHash::of(bytes);
        let input = "confirmed-database";
        let rows = serde_json::from_value(serde_json::json!({
            "input":input, "format":"codex-database",
            "revision":{"number":1,"produced":"first","chain":1,"length":bytes.len()},
            "extent":{"chain":1,"offset":0,"length":bytes.len(),"blob":hash,"blob_offset":0},
            "observation":{
                "id":"confirmed-database-observation", "machine":"test-machine", "host":"test",
                "path":db.to_string_lossy(), "path_bytes":db.as_os_str().as_encoded_bytes(),
                "class":"first", "revision":1, "length":bytes.len(),
                "device":null,"inode":null,"size":null,"mtime_ns":null,"ctime_ns":null,
                "polylogue_version":"synthetic", "seen_at":1,"details_json":null
            }
        }))
        .unwrap();
        writer
            .apply_confirmed(vec![
                Confirmed {
                    native: None,
                    position: 1,
                    output_index: 0,
                    event: "confirmed-database-event".into(),
                    change: ConfirmedChange::Live(Box::new(Product::Input {
                        rows: Box::new(rows),
                        material: Some(Material {
                            id: "confirmed-database-material".into(),
                            hash,
                            length: bytes.len() as u64,
                            extents: vec![physical.encode()],
                        }),
                    })),
                },
                control(ReplayControl::Completion { position: 1 }),
            ])
            .unwrap();
        drop(writer);
        // A real stored-byte read fails. Its database must remain deferred,
        // without a new database acquisition or an active worker to wait on.
        std::fs::remove_file(store.join("segments/test.zst")).unwrap();
        let mut engine = Engine::on_sinex(
            home.config(CHUNK),
            archive.clone(),
            sdk.clone(),
            polylogue_testkit::stopped_attempt(),
        )
        .unwrap();
        let pass = engine.run_pass().unwrap();
        assert!(
            pass.errors
                .iter()
                .any(|error| error.contains("state_5.sqlite")),
            "the stored database read must fail: {pass:?}"
        );
        assert_eq!(pass.capture_pending, pending_file, "{pass:?}");
        let captures = sdk.pending_captures().unwrap();
        assert_eq!(captures.is_empty(), !pending_file);
        assert!(captures.iter().all(|capture| matches!(
            &capture.captured, Capture::Input(input) if input.rows.format == "claude-code"
        )));
        if pending_file {
            assert_custody_complete(
                &sdk,
                &home.file("pending-with-error.jsonl"),
                &std::fs::read(home.file("pending-with-error.jsonl")).unwrap(),
            );
        }
        let reads = ReadPool::open(&archive).unwrap();
        let progress = reads
            .reader()
            .unwrap()
            .input_progress(input)
            .unwrap()
            .unwrap();
        assert!(!progress.parsed);
        assert_eq!(progress.length, bytes.len() as u64);
    }
}

#[test]
fn an_idle_continuation_reports_queued_writer_errors_without_clearing_custody() {
    let home = ClaudeHome::new();
    let db = home.home.join(".codex/state_5.sqlite");
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    std::fs::copy(
        polylogue_testkit::morrow_dir().join("codex/state_5.sqlite"),
        &db,
    )
    .unwrap();
    let (mut engine, archive, _lock, sdk) = open(&home);
    let captured = engine.run_pass().unwrap();
    assert!(captured.capture_pending, "{captured:?}");
    let mut before: Vec<_> = sdk
        .pending_captures()
        .unwrap()
        .into_iter()
        .map(|capture| capture.identity)
        .collect();
    before.sort();
    assert!(!before.is_empty());
    std::fs::remove_file(&db).unwrap();
    // The real writer rejects this queued reading because its source input
    // does not exist. No parse work is needed to observe the resulting report.
    engine
        .writer()
        .unwrap()
        .send(polylogue_store::Batch {
            required_disposition: None,
            input: polylogue_store::InputKey {
                id: "idle-continuation-error".into(),
                format: "codex-database".into(),
                arrival_rank: 0,
            },
            change: polylogue_store::Change::Reparse { chain: 1 },
            observation: None,
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length: 0,
            parse: Vec::new(),
            acquisition: None,
            basis: None,
            release: Default::default(),
        })
        .unwrap();
    archive.advance_generation_for_test().unwrap();
    let continued = engine.run_pass().unwrap();
    assert!(
        continued.errors.iter().any(|error| {
            error.starts_with("input idle-continuation-error:")
                && error.contains("stored reading input identity differs")
        }),
        "{continued:?}"
    );
    assert!(continued.capture_pending, "{continued:?}");
    assert_eq!(continued.inputs_changed, 0);
    assert_eq!(continued.bytes_read, 0);
    assert_eq!(continued.bytes_stored, 0);
    assert_eq!(continued.sessions_derived, 0);
    let mut after: Vec<_> = sdk
        .pending_captures()
        .unwrap()
        .into_iter()
        .map(|capture| capture.identity)
        .collect();
    after.sort();
    assert_eq!(after, before);
    let next = engine.run_pass().unwrap();
    assert!(next.capture_pending, "{next:?}");
    assert!(
        next.errors
            .iter()
            .all(|error| !error.contains("idle-continuation-error"))
    );
}

#[test]
fn a_disappeared_pending_source_survives_reader_failure_and_no_event_continuations() {
    use polylogue_engine::{Lifetime, LoopMsg, LoopStatus};
    use std::sync::{Mutex, mpsc::channel};
    use std::time::{Duration, Instant};

    let home = ClaudeHome::new();
    let bytes = polylogue_testkit::claude_user("u1", "pending", "owned before the reader failed");
    home.append("pending-before-error.jsonl", &bytes);
    let (mut engine, archive, _lock, sdk) = open(&home);
    // This file capture starts no database worker. Only real SDK confirmation
    // is outstanding when the next turn starts.
    let captured = engine.run_pass().unwrap();
    assert!(captured.capture_pending, "{captured:?}");
    let identities = || {
        let mut ids: Vec<_> = sdk
            .pending_captures()
            .unwrap()
            .into_iter()
            .map(|capture| capture.identity)
            .collect();
        ids.sort();
        ids
    };
    let before = identities();
    assert!(!before.is_empty());
    assert_custody_complete(
        &sdk,
        &home.file("pending-before-error.jsonl"),
        bytes.as_bytes(),
    );
    std::fs::remove_file(home.file("pending-before-error.jsonl")).unwrap();
    // Preserve the actual index while making its next reader acquisition fail.
    let held = archive.root.join("held-index.db");
    std::fs::rename(archive.index_db(), &held).unwrap();
    engine.config.live_every = Duration::from_millis(10);
    let status = Arc::new(Mutex::new(LoopStatus::default()));
    let shared = Arc::clone(&status);
    let (tx, rx) = channel();
    let worker_tx = tx.clone();
    let worker = std::thread::spawn(move || {
        let result = engine.run(worker_tx, &rx, &shared, Lifetime::UntilIdle, || {});
        (engine, result)
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    let observed = loop {
        let now = status.lock().unwrap().clone();
        if now.passes > 0 {
            break Some(now);
        }
        if worker.is_finished() || Instant::now() >= deadline {
            break None;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    // Restore the same index, then require successful ordinary continuations
    // to keep visiting the owned source without another watcher event.
    std::fs::rename(&held, archive.index_db()).unwrap();
    let recovered = loop {
        let now = status.lock().unwrap().clone();
        if now.passes > observed.as_ref().map_or(0, |s| s.passes) && now.last_pass.errors.is_empty()
        {
            break Some(now);
        }
        if worker.is_finished() || Instant::now() >= deadline {
            break None;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    let continued = loop {
        let now = status.lock().unwrap().clone();
        if recovered
            .as_ref()
            .is_some_and(|s| now.passes >= s.passes + 2)
            || worker.is_finished()
            || Instant::now() >= deadline
        {
            break now;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    let still_running = !worker.is_finished();
    let _ = tx.send(LoopMsg::Stop);
    let (engine, result) = worker.join().unwrap();
    let summary = result.unwrap();
    let observed = observed.expect("the failed turn must report its outcome");
    assert!(!observed.last_pass.errors.is_empty(), "{observed:?}");
    assert_eq!(observed.last_pass.inputs_changed, 0);
    assert!(observed.last_pass.capture_pending, "{observed:?}");
    assert!(observed.busy, "{observed:?}");
    let recovered = recovered.expect("restoring the index must resume the same capture");
    assert!(recovered.last_pass.capture_pending, "{recovered:?}");
    assert!(continued.passes >= recovered.passes + 2, "{continued:?}");
    assert!(continued.last_pass.capture_pending, "{continued:?}");
    assert_eq!(continued.last_pass.bytes_read, 0);
    assert_eq!(continued.last_pass.bytes_stored, 0);
    assert!(
        still_running,
        "UntilIdle returned before confirmation: {summary:?}"
    );
    assert!(!summary.work.errors.is_empty(), "{summary:?}");
    assert_eq!(identities(), before);
    assert_custody_complete(
        &sdk,
        &home.file("pending-before-error.jsonl"),
        bytes.as_bytes(),
    );
    drop(engine);
}

#[test]
fn a_gone_observation_waits_on_the_confirmed_input_through_the_capture_route() {
    use polylogue_store::primary::{Confirmed, ConfirmedChange, Product, ReplayControl};
    use polylogue_store::{FileFacts, Writer};

    let home = ClaudeHome::new();
    let path = home.home.join(".pi/agent/sessions/retained.jsonl");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"").unwrap();
    let facts = FileFacts::of(&std::fs::metadata(&path).unwrap());
    let store = home.dir.path().join("sdk");
    let sdk = Arc::new(client(&store));
    let (archive, _lock) = Archive::open_primary(&home.root, &store).unwrap();
    let writer = Writer::start_on_sinex(&archive, sdk.clone()).unwrap();
    let control = |control| Confirmed {
        native: None,
        position: 1,
        output_index: 0,
        event: String::new(),
        change: ConfirmedChange::Control(Box::new(control)),
    };
    writer
        .apply_confirmed(vec![control(ReplayControl::Reset {
            scope: "confirmed-empty-source".into(),
        })])
        .unwrap();
    let rows = serde_json::from_value(serde_json::json!({
        "input":"retained-raw-input", "format":"pi",
        "revision":{"number":1,"produced":"first","chain":1,"length":0},
        "observation":{
            "id":"original-empty-observation", "machine":"test-machine", "host":"test",
            "path":path.to_string_lossy(), "path_bytes":path.as_os_str().as_encoded_bytes(),
            "class":"first", "revision":1, "length":0,
            "device":facts.device,"inode":facts.inode,"size":facts.size,
            "mtime_ns":facts.mtime_ns,"ctime_ns":facts.ctime_ns,
            "polylogue_version":"synthetic", "seen_at":1,"details_json":null
        }
    }))
    .unwrap();
    writer
        .apply_confirmed(vec![
            Confirmed {
                native: None,
                position: 1,
                output_index: 0,
                event: "original-empty-source".into(),
                change: ConfirmedChange::Live(Box::new(Product::Input {
                    rows: Box::new(rows),
                    material: None,
                })),
            },
            control(ReplayControl::Completion { position: 1 }),
        ])
        .unwrap();
    drop(writer);
    std::fs::remove_file(&path).unwrap();
    let mut engine = Engine::on_sinex(
        home.config(CHUNK),
        archive.clone(),
        sdk.clone(),
        polylogue_testkit::stopped_attempt(),
    )
    .unwrap();
    let first = engine.run_pass().unwrap();
    assert!(first.capture_pending, "{first:?}");
    assert!(
        first
            .errors
            .iter()
            .all(|error| error.starts_with("confirmed stream:")
                && error.contains("Connection refused")),
        "{first:?}"
    );
    let pending = sdk.pending_captures().unwrap();
    assert_eq!(pending.len(), 1);
    let Capture::Input(input) = &pending[0].captured else {
        panic!("disappearance is an input observation");
    };
    assert_eq!(input.rows.input, "retained-raw-input");
    assert_eq!(input.rows.observation.as_ref().unwrap().class, "deleted");
    assert!(input.rows.revision.is_none());
    assert!(input.rows.extent.is_none());
    assert!(input.raw_range.is_none());
    let reads = ReadPool::open(&archive).unwrap();
    let known = reads
        .reader()
        .unwrap()
        .known_watched_input_at("test-machine", path.as_os_str().as_encoded_bytes())
        .unwrap()
        .unwrap();
    assert_eq!(known.input_id, "retained-raw-input");
    assert_eq!(
        known.facts,
        Some(facts),
        "preparation does not confirm Gone"
    );
    let next = engine.run_pass().unwrap();
    assert!(next.capture_pending, "{next:?}");
    assert_eq!(next.bytes_read, 0);
    assert_eq!(next.bytes_stored, 0);
    assert_eq!(sdk.pending_captures().unwrap(), pending);
}

#[test]
fn a_failed_scope_reset_reports_its_original_error_before_reply_without_a_commit_cue() {
    use polylogue_store::primary::{Confirmed, ConfirmedChange, ReplayControl};
    let home = ClaudeHome::new();
    let store = home.dir.path().join("sdk");
    let sdk = Arc::new(client(&store));
    let (archive, _lock) = Archive::open_primary(&home.root, &store).unwrap();
    let writer = polylogue_store::Writer::start_on_sinex(&archive, sdk).unwrap();
    let commits = writer.sender().unwrap().completion_commits();
    let reset = |scope: &str| {
        vec![Confirmed {
            native: None,
            position: 0,
            output_index: 0,
            event: String::new(),
            change: ConfirmedChange::Control(Box::new(ReplayControl::Reset {
                scope: scope.into(),
            })),
        }]
    };
    writer.apply_confirmed(reset("original")).unwrap();
    while commits.try_recv().is_ok() {}
    let index = rusqlite::Connection::open(archive.index_db()).unwrap();
    index.execute_batch("CREATE TRIGGER fail_scope BEFORE INSERT ON meta WHEN new.key='confirmed_scope' AND new.value='changed' BEGIN SELECT raise(ABORT,'original scope reset failure'); END").unwrap();
    let error = writer.apply_confirmed(reset("changed")).unwrap_err();
    assert!(
        error.to_string().contains("original scope reset failure"),
        "{error}"
    );
    assert!(matches!(
        archive.scope_token(),
        Err(polylogue_store::StoreError::ScopeChanged)
    ));
    let report = writer.flush_rows().unwrap();
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.starts_with("confirmed stream:")
                && error.contains("original scope reset failure")),
        "{report:?}"
    );
    assert!(matches!(
        commits.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    index.execute_batch("DROP TRIGGER fail_scope").unwrap();
    writer.apply_confirmed(reset("changed")).unwrap();
    archive.scope_token().unwrap().check().unwrap();
    assert_eq!(
        ReadPool::open(&archive)
            .unwrap()
            .reader()
            .unwrap()
            .confirmed_scope()
            .unwrap()
            .as_deref(),
        Some("changed")
    );
}
