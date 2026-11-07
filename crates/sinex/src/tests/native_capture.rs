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
    let engine = Engine::on_sinex(home.config(CHUNK), archive.clone(), sdk.clone()).unwrap();
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
    let mut engine = Engine::on_sinex(config, archive, sdk).unwrap();
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
