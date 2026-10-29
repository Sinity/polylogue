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
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(pass.capture_pending);
    assert_eq!(pass.bytes_read, text.len() as u64);
    let count = assert_complete(&sdk, &archive, &path, text.as_bytes());
    assert!(count > 2);
    drop(engine);
    drop(lock);
    drop(sdk);

    let (mut engine, archive, _lock, sdk) = open(&home);
    let quiet = engine.run_pass().unwrap();
    assert!(quiet.errors.is_empty(), "{quiet:?}");
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
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pass.bytes_read, text.len() as u64 - CHUNK);
    assert!(assert_complete(&sdk, &archive, &path, text.as_bytes()) > 2);
}

#[test]
fn offline_empty_native_file_has_an_actual_body_and_eof_with_no_confirmed_source() {
    let home = ClaudeHome::new();
    home.append("empty.jsonl", "");
    let path = home.file("empty.jsonl");
    let (mut engine, archive, _lock, sdk) = open(&home);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(pass.capture_pending);
    assert_eq!(assert_complete(&sdk, &archive, &path, b""), 1);
    let quiet = engine.run_pass().unwrap();
    assert!(quiet.errors.is_empty(), "{quiet:?}");
    assert_eq!(assert_complete(&sdk, &archive, &path, b""), 1);
}
