//! The consumer half of a Sinex backup and restore, against a real daemon.
use super::super::*;
use crate::{Mode, Sinex};
use polylogue_model::BlobHash;
use polylogue_model::LockExt;
use polylogue_store::primary::{Material, Product, ProductWrite, SourceLocation};
use polylogue_store::{Archive, ReadPool, Writer};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

#[path = "../../../tests/support/real_sinex.rs"]
mod real_sinex;

fn config(real_sinex: &real_sinex::RealSinex) -> Config {
    Config {
        mode: Mode::Primary,
        nats: real_sinex.nats.url().into(),
        password_file: None,
        subject_root: "rebuild".into(),
        producer: "polylogue".into(),
        store: real_sinex.store.clone(),
        upload_root: None,
    }
}

fn wait_until(stage: &str, cache: &Path, sinex: &Sinex, mut predicate: impl FnMut() -> bool) {
    let started = Instant::now();
    while !predicate() {
        if started.elapsed() >= Duration::from_secs(90) {
            panic!(
                "Sinex fixture {} did not finish {stage}: {:?}",
                cache.display(),
                sinex.status()
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn connect(config: &Config) -> Arc<Sinex> {
    let started = Instant::now();
    loop {
        match Sinex::connect(config, &Authentication::Anonymous, "fixture-host") {
            Ok(connected) => return Arc::new(connected),
            Err(error @ Error::Unreachable(_)) => assert!(
                started.elapsed() < Duration::from_secs(90),
                "Sinex producer did not become reachable: {error}"
            ),
            Err(error) => panic!("Sinex producer connection failed: {error}"),
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn prepare(sinex: &Sinex, bytes: &[u8]) -> String {
    sinex
        .prepare_blob(
            BlobHash::of(bytes),
            bytes.len() as u64,
            crate::tests::body(&sinex.config.store, bytes),
        )
        .unwrap()
        .identity
}

fn encoding(
    sinex: &Sinex,
    identity: &str,
) -> (hashed_blobs::EncodedBlob, hashed_blobs::EncodedBlob) {
    let queue = sinex.spool.locked();
    (
        queue.prepared_frame(identity).unwrap().descriptor(),
        queue.prepared_body(identity).unwrap()[0].descriptor(),
    )
}

/// Material acknowledgment, the dependent confirmed group, its application to the
/// confirmed cache, then the consumer's confirmed removal. No test manufactures an
/// acknowledgment or position.
fn send_and_apply(sinex: &Arc<Sinex>, cache: &Path, items: &[(&str, &[u8])]) {
    sinex.send_pending();
    wait_until("acknowledging the pending frames", cache, sinex, || {
        items.iter().all(|(identity, _)| {
            sinex
                .spool
                .locked()
                .acknowledgment(identity)
                .unwrap()
                .is_some()
        })
    });
    for (identity, bytes) in items {
        let acknowledgment = sinex
            .spool
            .locked()
            .acknowledgment(identity)
            .unwrap()
            .unwrap();
        let sinex_protocol::FrameResult::Admitted {
            material: Some(id),
            material_extents,
            ..
        } = acknowledgment.result
        else {
            panic!("material admitted: {acknowledgment:?}")
        };
        let material = Material {
            id: id.to_string(),
            hash: BlobHash::of(bytes),
            length: bytes.len() as u64,
            extents: material_extents,
        };
        sinex
            .complete_captures(
                &[(*identity).into()],
                &[ProductWrite {
                    location: Some(SourceLocation {
                        material: material.id.clone(),
                        anchor: 0,
                    }),
                    product: Product::Blob(material),
                    parents: Vec::new(),
                    machine: Some("fixture-host".into()),
                    output_index: 0,
                }],
            )
            .unwrap();
    }
    let (archive, lock) = Archive::open_primary(cache, &sinex.config.store).unwrap();
    let writer = Writer::start_on_sinex(&archive, sinex.clone()).unwrap();
    writer.synchronize_primary().unwrap_or_else(|error| {
        panic!(
            "synchronize confirmed frames for {}: {error}",
            cache.display()
        )
    });
    wait_until("applying and retiring the frames", cache, sinex, || {
        let pool = match ReadPool::open(&archive) {
            Ok(pool) => pool,
            Err(polylogue_store::StoreError::ScopeChanged) => return false,
            Err(error) => panic!("open Sinex primary cache: {error}"),
        };
        let reader = match pool.reader() {
            Ok(reader) => reader,
            Err(polylogue_store::StoreError::ScopeChanged) => return false,
            Err(error) => panic!("read Sinex primary cache: {error}"),
        };
        items.iter().all(|(identity, bytes)| {
            if sinex.spool.locked().get(identity).unwrap().is_some() {
                return false;
            }
            let blobs = match reader.blob_reader() {
                Ok(blobs) => blobs,
                Err(polylogue_store::StoreError::ScopeChanged) => return false,
                Err(error) => panic!("open Sinex primary blob reader: {error}"),
            };
            let mut body = match blobs.read(&hashed_blobs::Hash(BlobHash::of(bytes).0)) {
                Ok(body) => body,
                Err(
                    polylogue_store::StoreError::NotFound(_)
                    | polylogue_store::StoreError::ScopeChanged,
                ) => return false,
                Err(error) => panic!("read confirmed Sinex primary bytes: {error}"),
            };
            let mut found = Vec::new();
            body.read_to_end(&mut found).unwrap();
            found == *bytes
        })
    });
    drop(writer);
    drop(lock);
}

fn frame_count(real_sinex: &real_sinex::RealSinex, identity: &str) -> i64 {
    real_sinex.sql(&format!("SELECT count(*) FROM core.ingress_frames WHERE producer='polylogue' AND frame='{}' AND refusal IS NULL",identity.replace('\'',"''"))).trim().parse().unwrap()
}

enum BackupEvent {
    Capturing,
    Finished,
}

fn start_backup(
    progress: mpsc::SyncSender<BackupEvent>,
    prepare: impl FnOnce() -> std::result::Result<serde_json::Value, String> + Send + 'static,
) -> std::thread::JoinHandle<std::result::Result<serde_json::Value, String>> {
    std::thread::spawn(move || {
        let result = prepare();
        let _ = progress.send(BackupEvent::Finished);
        result
    })
}

fn prepare_backup(
    socket: &Path,
    token: &Path,
    output: &Path,
) -> std::result::Result<serde_json::Value, String> {
    let daemon = std::env::var_os("SINEXD").ok_or("missing matching SINEXD")?;
    let client = PathBuf::from(daemon).with_file_name("sinexctl");
    let result = std::process::Command::new(client)
        .arg("--endpoint")
        .arg(format!("unix:{}", socket.display()))
        .arg("--token-file")
        .arg(token)
        .args(["--json", "backup", "prepare", "--deadline", "none", "--to"])
        .arg(output)
        .output()
        .map_err(|error| format!("run sinexctl backup prepare: {error}"))?;
    if !result.status.success() {
        return Err(format!(
            "sinexctl backup prepare {}: {} {}",
            result.status,
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr),
        ));
    }
    serde_json::from_slice(&result.stdout).map_err(|error| format!("read backup result: {error}"))
}

#[test]
#[ignore = "needs matching public Sinex daemon plus pg_restore; set SINEXD"]
fn paired_backup_restores_polylogue_pending_frames_after_live_confirmed_removal() {
    let root = tempfile::tempdir().unwrap();
    let mut running = real_sinex::RealSinex::start(
        &root.path().join("real_sinex"),
        vec![serde_json::to_value(crate::kinds::source_declaration()).unwrap()],
        false, // cross-domain evidence producer
    );
    let initial = connect(&config(&running));
    let baseline = b"synthetic baseline bytes";
    let baseline_id = prepare(&initial, baseline);
    send_and_apply(
        &initial,
        &root.path().join("baseline-index"),
        &[(&baseline_id, baseline)],
    );
    drop(initial);

    // Pausing the Sinex callback before its queue lock lets the SDK prepare
    // new bytes while Sinex's backup publication gate is closed.
    let mut sinex = Arc::new(super::synchronous(
        &config(&running),
        &Authentication::Anonymous,
    ));
    let (entered, wait_capture) = mpsc::sync_channel(1);
    let response_progress = entered.clone();
    let (resume, wait_resume) = mpsc::sync_channel(1);
    let wait_resume = Mutex::new(wait_resume);
    let spool = sinex.spool.clone();
    let callback = move |cancelled: Arc<std::sync::atomic::AtomicBool>| {
        entered
            .send(BackupEvent::Capturing)
            .map_err(|error| error.to_string())?;
        // No production mutex is held while the fixture waits.
        wait_resume
            .locked()
            .recv_timeout(Duration::from_secs(90))
            .map_err(|error| error.to_string())?;
        capture_backup(&spool, &cancelled)
    };
    let client = sinex
        .runtime
        .block_on(Client::connect_with_context(
            &sinex.config,
            &sinex.authentication,
            sinex.spool.clone(),
            sinex.health.clone(),
            callback,
            sinex.publication.clone(),
            true,
        ))
        .unwrap();
    *sinex.client.locked() = Some(client);
    let before = b"synthetic pending before the core backup";
    let before_id = prepare(&sinex, before);
    let before_encoding = encoding(&sinex, &before_id);
    assert_eq!(frame_count(&running, &before_id), 0);
    let artifact = root.path().join("artifact");
    let socket = running.socket.clone();
    let token = root.path().join("real_sinex/tokens/operator.token");
    let destination = artifact.clone();
    let backup = start_backup(response_progress, move || {
        prepare_backup(&socket, &token, &destination)
    });
    match wait_capture.recv_timeout(Duration::from_secs(90)).unwrap() {
        BackupEvent::Capturing => (),
        BackupEvent::Finished => panic!(
            "backup returned before capturing the producer: {:?}",
            backup.join().unwrap()
        ),
    }
    let during = b"synthetic capture while the publication gate is closed";
    let during_id = prepare(&sinex, during);
    let during_encoding = encoding(&sinex, &during_id);
    // Exercise the normal sender while Sinex backup holds its publication
    // gate. Its separate connection leaves confirmed reads available, and the
    // capture above finished before the backup callback resumed.
    Arc::get_mut(&mut sinex).unwrap().start_sender().unwrap();
    assert!(
        sinex.client.try_lock().is_ok(),
        "sending does not hold the confirmed-reader lock"
    );
    sinex.reachable().unwrap();
    assert_eq!(frame_count(&running, &before_id), 0);
    assert_eq!(frame_count(&running, &during_id), 0);
    resume.send(()).unwrap();
    let reply = backup
        .join()
        .unwrap()
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(reply["output"].as_str(), artifact.to_str());
    assert_eq!(reply["hold_open"].as_bool(), Some(true));
    let manifest: sinex_model::backup::Manifest =
        serde_json::from_slice(&std::fs::read(artifact.join("manifest.json")).unwrap()).unwrap();
    assert!(manifest.producers_complete);
    assert!(
        manifest
            .required_producers
            .as_ref()
            .unwrap()
            .iter()
            .any(|producer| producer == "polylogue")
    );
    let snapshot = manifest
        .producer_snapshots
        .iter()
        .find(|snapshot| snapshot.producer == "polylogue")
        .unwrap();
    assert!(snapshot.run_id.is_some());
    assert!(snapshot.unavailable.is_none());
    assert!(!snapshot.files.is_empty());
    for file in &snapshot.files {
        let bytes = std::fs::read(artifact.join(&file.path)).unwrap();
        assert_eq!(bytes.len() as u64, file.bytes);
        assert_eq!(blake3::hash(&bytes).to_hex().as_str(), file.blake3);
    }
    send_and_apply(
        &sinex,
        &root.path().join("live-index"),
        &[(&before_id, before), (&during_id, during)],
    );
    for identity in [&before_id, &during_id] {
        assert_eq!(frame_count(&running, identity), 1);
        assert!(sinex.spool.locked().get(identity).unwrap().is_none());
    }
    drop(sinex);
    let restored_store = root.path().join("restored-store");
    running.restore(&artifact, &restored_store);
    // The core backup excludes both untransmitted frames, although the later live core
    // admitted and removed them. Only the producer snapshot in the backup can resend them.
    assert_eq!(frame_count(&running, &baseline_id), 1);
    assert_eq!(frame_count(&running, &before_id), 0);
    assert_eq!(frame_count(&running, &during_id), 0);
    running.restart();
    let restored = connect(&config(&running));
    assert_eq!(encoding(&restored, &before_id), before_encoding);
    assert_eq!(encoding(&restored, &during_id), during_encoding);
    send_and_apply(
        &restored,
        &root.path().join("restored-index"),
        &[(&before_id, before), (&during_id, during)],
    );
    for identity in [&baseline_id, &before_id, &during_id] {
        assert_eq!(frame_count(&running, identity), 1);
    }
    let archive = Archive::on_sinex(&root.path().join("restored-index"), &restored_store);
    let mut body = archive
        .blob_reader()
        .unwrap()
        .read(&hashed_blobs::Hash(BlobHash::of(baseline).0))
        .unwrap();
    let mut found = Vec::new();
    body.read_to_end(&mut found).unwrap();
    assert_eq!(
        found, baseline,
        "fresh confirmed stream also rebuilt the pre-backup baseline"
    );
}

#[test]
fn backup_http_refusal_reaches_the_waiter_without_a_capture_callback() {
    use std::io::{BufRead, BufReader, Read, Write};
    let root = tempfile::tempdir().unwrap();
    let socket = root.path().join("backup.sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let destination = root.path().join("artifact");
    let expected = destination.clone();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(&mut stream);
        let mut length = 0;
        loop {
            let mut line = String::new();
            assert_ne!(
                reader.read_line(&mut line).unwrap(),
                0,
                "HTTP request ended early"
            );
            if line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse::<usize>().unwrap();
            }
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).unwrap();
        let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(request["method"], "backup");
        assert_eq!(request["action"], "prepare");
        assert_eq!(request["to"], expected.display().to_string());
        let body = "synthetic backup refusal";
        write!(stream, "HTTP/1.1 422 Unprocessable Entity\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
    });
    let (progress, wait) = mpsc::sync_channel(1);
    let backup = start_backup(progress, move || {
        let request = sinex_protocol::Request::Backup {
            action: "prepare".into(),
            hold: None,
            to: Some(destination.display().to_string()),
            page_size: None,
            cursor: None,
        };
        real_sinex::request(
            &socket,
            "operator",
            "/v1",
            serde_json::to_value(request).unwrap(),
        )
    });
    assert!(matches!(
        wait.recv_timeout(Duration::from_secs(5)).unwrap(),
        BackupEvent::Finished
    ));
    let error = backup.join().unwrap().unwrap_err();
    assert!(error.contains("422"), "{error}");
    assert!(error.contains("synthetic backup refusal"), "{error}");
    server.join().unwrap();
}
