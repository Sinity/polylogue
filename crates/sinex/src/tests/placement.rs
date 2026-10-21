//! Exercise placement through real NATS without a database or product daemon.
use crate::*;
use polylogue_store::Archive;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[tokio::test]
async fn blob_placement_checks_each_ack_batch_before_requesting_more() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let config = Config {
        mode: Mode::Primary,
        nats: nats.url().into(),
        subjects: "placement".into(),
        producer: "polylogue".into(),
        store: dir.path().join("store"),
        archive: None,
    };
    let mut sinex = Sinex::connect(&config).await.unwrap();
    let mut requests = sinex
        .client
        .subscribe(sinex.subjects.have(&sinex.producer))
        .await
        .unwrap();
    sinex.client.flush().await.unwrap();
    let refused = Arc::new(AtomicBool::new(true));
    let queried = Arc::new(AtomicUsize::new(0));
    let worker = tokio::spawn({
        let client = sinex.client.clone();
        let refused = Arc::clone(&refused);
        let queried = Arc::clone(&queried);
        async move {
            while let Some(message) = requests.next().await {
                let request: Have = serde_json::from_slice(&message.payload).unwrap();
                assert!(!request.frames.is_empty() && request.frames.len() <= 500);
                queried.fetch_add(request.frames.len(), Ordering::SeqCst);
                let reject = refused.load(Ordering::SeqCst);
                let acks: Vec<_> = request
                    .frames
                    .into_iter()
                    .map(|frame| Acknowledgment {
                        frame,
                        material: Some(MaterialId::new()),
                        admitted: vec![],
                        outcome: serde_json::json!({}),
                        refusal: reject.then(|| "synthetic refusal".into()),
                        commit_seq: Some(1),
                    })
                    .collect();
                client
                    .publish(
                        message.reply.unwrap(),
                        serde_json::to_vec(&acks).unwrap().into(),
                    )
                    .await
                    .unwrap();
            }
        }
    });
    let root = dir.path().join("archive");
    let (_archive, _lock) = Archive::open_or_create(&root).unwrap();
    let rows = ArchiveRows::open(&root).unwrap();
    let hashes: Vec<_> = (0u64..1203)
        .map(|i| {
            let mut hash = [0; 32];
            hash[..8].copy_from_slice(&i.to_be_bytes());
            hash
        })
        .collect();
    let mut done = Published::default();
    let error = sinex
        .place_blobs(&rows, "synthetic", &hashes, &mut done)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Refused { .. }), "{error}");
    assert_eq!(
        queried.load(Ordering::SeqCst),
        500,
        "later acknowledgment groups must wait until earlier refusals are checked"
    );
    assert_eq!(done.blobs, 0);
    refused.store(false, Ordering::SeqCst);
    queried.store(0, Ordering::SeqCst);
    sinex
        .place_blobs(&rows, "synthetic", &hashes, &mut done)
        .await
        .unwrap();
    assert_eq!(
        queried.load(Ordering::SeqCst),
        hashes.len(),
        "retry must query every reference again"
    );
    assert_eq!(
        done.blobs, 0,
        "held references never open absent local blob bodies"
    );
    sinex
        .place_blobs(&rows, "synthetic", &[], &mut done)
        .await
        .unwrap();
    assert_eq!(queried.load(Ordering::SeqCst), hashes.len());
    let names: Vec<_> = hashes
        .iter()
        .map(|hash| format!("blake3:{}", sinex_rows::hex(hash)))
        .collect();
    let mut invalid = names.clone();
    invalid.push("not-a-content-address".into());
    queried.store(0, Ordering::SeqCst);
    let error = sinex
        .place_named_blobs(&rows, "synthetic", &invalid, &mut done)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Invalid(_)), "{error}");
    assert_eq!(
        queried.load(Ordering::SeqCst),
        0,
        "all names must validate before the first request"
    );
    refused.store(true, Ordering::SeqCst);
    let error = sinex
        .place_named_blobs(&rows, "synthetic", &names, &mut done)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Refused { .. }), "{error}");
    assert_eq!(queried.load(Ordering::SeqCst), 500);
    refused.store(false, Ordering::SeqCst);
    queried.store(0, Ordering::SeqCst);
    sinex
        .place_named_blobs(&rows, "synthetic", &names, &mut done)
        .await
        .unwrap();
    assert_eq!(queried.load(Ordering::SeqCst), names.len());
    assert_eq!(
        done.blobs, 0,
        "held named references must not open missing local bodies"
    );
    sinex
        .place_named_blobs(&rows, "synthetic", &[], &mut done)
        .await
        .unwrap();
    assert_eq!(queried.load(Ordering::SeqCst), names.len());
    worker.abort();
    assert!(worker.await.unwrap_err().is_cancelled());
}
