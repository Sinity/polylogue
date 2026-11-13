//! The consumer-specific half of native backup custody, against an actual daemon.
use super::super::*;
use crate::{Mode, Sinex};
use polylogue_model::BlobHash;
use polylogue_store::primary::{Material, Product, ProductWrite, SourceLocation};
use polylogue_store::{Archive, ReadPool, Writer};
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

#[path = "../../../tests/support/native.rs"]
mod native;

fn config(native: &native::Native) -> Config {
    Config {
        mode: Mode::Primary,
        nats: native.nats.url().into(),
        password_file: None,
        subject_root: "rebuild".into(),
        producer: "polylogue".into(),
        store: native.store.clone(),
        upload_root: None,
    }
}

fn wait_until(mut predicate: impl FnMut() -> bool) {
    let started = Instant::now();
    while !predicate() {
        assert!(
            started.elapsed() < Duration::from_secs(90),
            "native primary fixture did not settle"
        );
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
                "native producer did not become reachable: {error}"
            ),
            Err(error) => panic!("native producer connection failed: {error}"),
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
    let queue = sinex.spool.lock().unwrap();
    (
        queue.prepared_frame(identity).unwrap().descriptor(),
        queue.prepared_body(identity).unwrap()[0].descriptor(),
    )
}

/// Material ACK, dependent canonical group, actual confirmed-cache application,
/// then ordinary consumer confirmed removal. No test manufactures an acknowledgment or position.
fn publish_and_project(sinex: &Arc<Sinex>, cache: &Path, items: &[(&str, &[u8])]) {
    sinex.send_pending();
    wait_until(|| {
        items.iter().all(|(identity, _)| {
            sinex
                .spool
                .lock()
                .unwrap()
                .acknowledgment(identity)
                .unwrap()
                .is_some()
        })
    });
    for (identity, bytes) in items {
        let acknowledgment = sinex
            .spool
            .lock()
            .unwrap()
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
                    publish: true,
                }],
            )
            .unwrap();
    }
    let (archive, lock) = Archive::open_primary(cache, &sinex.config.store).unwrap();
    let writer = Writer::start_on_sinex(&archive, sinex.clone()).unwrap();
    wait_until(|| {
        let pool = match ReadPool::open(&archive) {
            Ok(pool) => pool,
            Err(polylogue_store::StoreError::ScopeChanged) => return false,
            Err(error) => panic!("open native primary cache: {error}"),
        };
        let reader = match pool.reader() {
            Ok(reader) => reader,
            Err(polylogue_store::StoreError::ScopeChanged) => return false,
            Err(error) => panic!("read native primary cache: {error}"),
        };
        items.iter().all(|(identity, bytes)| {
            if sinex.spool.lock().unwrap().get(identity).unwrap().is_some() {
                return false;
            }
            let blobs = match reader.blob_reader() {
                Ok(blobs) => blobs,
                Err(polylogue_store::StoreError::ScopeChanged) => return false,
                Err(error) => panic!("open native primary blob reader: {error}"),
            };
            let mut body = match blobs.read(&hashed_blobs::Hash(BlobHash::of(bytes).0)) {
                Ok(body) => body,
                Err(
                    polylogue_store::StoreError::NotFound(_)
                    | polylogue_store::StoreError::ScopeChanged,
                ) => return false,
                Err(error) => panic!("read confirmed native primary bytes: {error}"),
            };
            let mut found = Vec::new();
            body.read_to_end(&mut found).unwrap();
            found == *bytes
        })
    });
    drop(writer);
    drop(lock);
}

fn frame_count(native: &native::Native, identity: &str) -> i64 {
    native.sql(&format!("SELECT count(*) FROM core.ingress_frames WHERE producer='polylogue' AND frame='{}' AND refusal IS NULL",identity.replace('\'',"''"))).trim().parse().unwrap()
}

#[test]
#[ignore = "needs matching public Sinex daemon plus pg_restore; set SINEXD"]
fn paired_backup_restores_polylogue_pending_frames_after_live_confirmed_removal() {
    let root = tempfile::tempdir().unwrap();
    let mut running = native::Native::start(
        &root.path().join("native"),
        vec![serde_json::to_value(crate::kinds::source_declaration()).unwrap()],
    );
    let initial = connect(&config(&running));
    let baseline = b"synthetic baseline bytes";
    let baseline_id = prepare(&initial, baseline);
    publish_and_project(
        &initial,
        &root.path().join("baseline-index"),
        &[(&baseline_id, baseline)],
    );
    drop(initial);

    // Pausing the native callback before its queue lock lets the ordinary SDK
    // prepare new bytes while publication is paused.
    let mut sinex = Arc::new(super::synchronous(
        &config(&running),
        &Authentication::Anonymous,
    ));
    let (entered, wait_capture) = mpsc::sync_channel(1);
    let (resume, wait_resume) = mpsc::sync_channel(1);
    let wait_resume = Mutex::new(wait_resume);
    let spool = sinex.spool.clone();
    let callback = move |cancelled: Arc<std::sync::atomic::AtomicBool>| {
        entered.send(()).map_err(|error| error.to_string())?;
        // No production mutex is held while the fixture waits.
        wait_resume
            .lock()
            .unwrap()
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
    *sinex.client.lock().unwrap() = Some(client);
    let before = b"synthetic pending before the core backup";
    let before_id = prepare(&sinex, before);
    let before_encoding = encoding(&sinex, &before_id);
    assert_eq!(frame_count(&running, &before_id), 0);
    let artifact = root.path().join("artifact");
    let socket = running.socket.clone();
    let output = artifact.clone();
    let backup = std::thread::spawn(move || {
        native::request(
            &socket,
            "operator",
            serde_json::json!({
                "method":"backup","action":"prepare","output":output
            }),
        )
    });
    wait_capture.recv_timeout(Duration::from_secs(90)).unwrap();
    let during = b"synthetic acquisition while publication is frozen";
    let during_id = prepare(&sinex, during);
    let during_encoding = encoding(&sinex, &during_id);
    // Exercise the normal sender while native backup holds its publication
    // gate. Its independent connection leaves confirmed reads available, and
    // capture above completed before acquisition was released.
    Arc::get_mut(&mut sinex).unwrap().start_sender().unwrap();
    assert!(
        sinex.client.try_lock().is_ok(),
        "publication does not own the confirmed-reader lock"
    );
    sinex.reachable().unwrap();
    assert_eq!(frame_count(&running, &before_id), 0);
    assert_eq!(frame_count(&running, &during_id), 0);
    resume.send(()).unwrap();
    let reply = backup.join().unwrap();
    assert!(reply.get("result").is_some(), "{reply}");
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
    publish_and_project(
        &sinex,
        &root.path().join("live-index"),
        &[(&before_id, before), (&during_id, during)],
    );
    for identity in [&before_id, &during_id] {
        assert_eq!(frame_count(&running, identity), 1);
        assert!(sinex.spool.lock().unwrap().get(identity).unwrap().is_none());
    }
    drop(sinex);
    let restored_store = root.path().join("restored-store");
    running.restore(&artifact, &restored_store);
    // The core backup excludes both untransmitted frames, although the later live core
    // admitted and removed them. Only the acquired producer artifact can retry.
    assert_eq!(frame_count(&running, &baseline_id), 1);
    assert_eq!(frame_count(&running, &before_id), 0);
    assert_eq!(frame_count(&running, &during_id), 0);
    running.restart();
    let restored = connect(&config(&running));
    assert_eq!(encoding(&restored, &before_id), before_encoding);
    assert_eq!(encoding(&restored, &during_id), during_encoding);
    publish_and_project(
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
        "fresh confirmed stream also rebuilt the pre-cut baseline"
    );
}
