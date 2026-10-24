use super::*;
use std::io::{Seek, Write};
use tokio::sync::oneshot;

async fn setup(
    root: &Path,
    nats: &devkit::nats::Server,
) -> (Config, Sinex, async_nats::Subscriber) {
    let config = Config {
        mode: Mode::Primary,
        nats: nats.url().into(),
        subjects: "byte-proof".into(),
        producer: "polylogue".into(),
        store: root.to_path_buf(),
        archive: None,
    };
    let sinex = Sinex::connect(&config).await.unwrap();
    jetstream::new(sinex.client.clone())
        .create_stream(async_nats::jetstream::stream::Config {
            name: "BYTE_PROOF_INGRESS".into(),
            subjects: vec![sinex.subjects.ingress(&sinex.producer)],
            ..Default::default()
        })
        .await
        .unwrap();
    let ingress = sinex
        .client
        .subscribe(sinex.subjects.ingress(&sinex.producer))
        .await
        .unwrap();
    sinex.client.flush().await.unwrap();
    (config, sinex, ingress)
}

fn source(bytes: &[u8]) -> sinex_rows::BlobSnapshot {
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(bytes).unwrap();
    file.rewind().unwrap();
    sinex_rows::BlobSnapshot {
        file,
        length: bytes.len() as u64,
    }
}

fn acknowledgment(
    frame: String,
    material: Option<MaterialId>,
    outcome: serde_json::Value,
) -> Acknowledgment {
    Acknowledgment {
        frame,
        material,
        admitted: vec![],
        outcome,
        refusal: None,
        commit_seq: Some(1),
    }
}

async fn ack(client: &async_nats::Client, subject: &str, ack: &Acknowledgment) {
    client
        .publish(subject.to_owned(), serde_json::to_vec(ack).unwrap().into())
        .await
        .unwrap();
    client.flush().await.unwrap();
}

#[tokio::test]
async fn cancellation_after_reserve_reuses_frozen_identity_and_completed_ack_needs_no_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (config, mut sinex, mut ingress) = setup(dir.path(), &nats).await;
    // Force a large manifest through indirection. Material BODY and JSON
    // ENVELOPE share the original final-frame owner, with distinct roles.
    sinex.max_message = 8192;
    let client = sinex.client.clone();
    let subject = sinex.subjects.ack(&sinex.producer);
    let (observed, after_final) = oneshot::channel();
    let root = dir.path().to_owned();
    let worker = tokio::spawn(async move {
        let Ingress::Reserve {
            frame: first_attempt,
            publication,
        } = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("reserve first")
        };
        assert_eq!(publication.class, Class::Open);
        assert_eq!(publication.frame, "stable-material");
        assert!(publication.material.is_some() && publication.envelope.is_some());
        let frozen = spool::load(&root, "polylogue", &publication.frame)
            .unwrap()
            .unwrap();
        let id = publication.material.as_ref().unwrap().id;
        for (role, _) in publication.roles() {
            spool::snapshot(&root, "polylogue", &publication, role).unwrap();
        }
        // Nothing has reached public CAS before receiver reservation.
        let public = hashed_blobs::Store::open(root.clone(), vec![], 0o600).unwrap();
        let hash = parse_hash(&publication.material.as_ref().unwrap().bytes.blob).unwrap();
        assert!(!public.contains(&hash));
        ack(
            &client,
            &subject,
            &acknowledgment(
                first_attempt.clone(),
                None,
                serde_json::json!({"producer_bytes":1,"complete":null}),
            ),
        )
        .await;
        let wire: Ingress = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap();
        assert_eq!(wire, frozen.wire_frame());
        assert!(matches!(wire, Ingress::Indirect { .. }));
        observed.send((frozen.clone(), id)).unwrap();
        // Simulate committed admission with lost final ack: receiver cleaned
        // its roles, but cannot remove the sender's durable typed identity.
        spool::discard_bytes(&root, "polylogue", &publication).unwrap();
        let Ingress::Reserve {
            frame: retry_attempt,
            publication: retry,
        } = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("retry reserve")
        };
        assert_ne!(retry_attempt, first_attempt);
        assert_eq!(retry, publication);
        let done = acknowledgment(publication.frame.clone(), Some(id), serde_json::json!({}));
        ack(
            &client,
            &subject,
            &acknowledgment(
                retry_attempt,
                None,
                serde_json::json!({"producer_bytes":1,"complete":done}),
            ),
        )
        .await;
    });
    let bytes = b"Synthetic body";
    let hash = hashed_blobs::Hash(*blake3::hash(bytes).as_bytes());
    let call = tokio::spawn(async move {
        sinex
            .material(
                "stable-material".into(),
                None,
                &hash,
                || Ok(source(bytes)),
                serde_json::json!({"large":"x".repeat(16000)}),
                Class::Open,
            )
            .await
    });
    let (original, id) = after_final.await.unwrap();
    call.abort();
    assert!(call.await.unwrap_err().is_cancelled());
    assert_eq!(
        spool::load(dir.path(), "polylogue", "stable-material").unwrap(),
        Some(original)
    );
    let mut retry = Sinex::connect(&config).await.unwrap();
    retry.max_message = 8192;
    let got = retry
        .material(
            "stable-material".into(),
            None,
            &hash,
            || panic!("retry must reuse frozen bytes and UUIDv7 without reopening source"),
            serde_json::json!({"changed":"must not replace frozen manifest"}),
            Class::Open,
        )
        .await
        .unwrap();
    assert_eq!(got, id);
    worker.await.unwrap();
    assert!(
        spool::load(dir.path(), "polylogue", "stable-material")
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn sealed_material_refuses_before_source_or_plaintext_spool() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (_, mut sinex, _) = setup(dir.path(), &nats).await;
    let result = sinex
        .material(
            "sealed".into(),
            None,
            &hashed_blobs::Hash([0; 32]),
            || panic!("sealed source must not be read"),
            serde_json::json!({}),
            Class::Sealed,
        )
        .await;
    assert!(matches!(result, Err(Error::Invalid(message)) if message.contains("sealed")));
    assert!(!dir.path().join("incoming").exists());
}

#[tokio::test]
async fn missing_reservation_version_refuses_and_preserves_private_retry_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (_, mut sinex, mut ingress) = setup(dir.path(), &nats).await;
    let client = sinex.client.clone();
    let subject = sinex.subjects.ack(&sinex.producer);
    let worker = tokio::spawn(async move {
        let frame: Ingress =
            serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap();
        assert!(matches!(frame, Ingress::Reserve { .. }));
        ack(
            &client,
            &subject,
            &acknowledgment(frame.frame().into(), None, serde_json::json!({})),
        )
        .await;
        assert!(
            tokio::time::timeout(Duration::from_millis(100), ingress.next())
                .await
                .is_err()
        );
    });
    let bytes = b"synthetic";
    let hash = hashed_blobs::Hash(*blake3::hash(bytes).as_bytes());
    let result = sinex
        .material(
            "unsupported".into(),
            None,
            &hash,
            || Ok(source(bytes)),
            serde_json::json!({}),
            Class::Open,
        )
        .await;
    assert!(matches!(result, Err(Error::Invalid(message)) if message.contains("does not support")));
    assert!(
        spool::load(dir.path(), "polylogue", "unsupported")
            .unwrap()
            .is_some()
    );
    let public = hashed_blobs::Store::open(dir.path().to_owned(), vec![], 0o600).unwrap();
    assert!(!public.contains(&hash));
    worker.await.unwrap();
}

#[tokio::test]
async fn cancellation_with_incomplete_reservation_reuses_exact_material_and_private_body() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (config, mut sinex, mut ingress) = setup(dir.path(), &nats).await;
    sinex.store_root = dir.path().join("unavailable-restore-reader");
    let client = sinex.client.clone();
    let subject = sinex.subjects.ack(&sinex.producer);
    let (reserved, wait_reservation) = oneshot::channel();
    let root = dir.path().to_owned();
    let worker = tokio::spawn(async move {
        let Ingress::Reserve { publication, .. } =
            serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("reserve first")
        };
        assert!(publication.envelope.is_none());
        let frozen = spool::load(&root, "polylogue", &publication.frame)
            .unwrap()
            .unwrap();
        reserved.send(frozen.clone()).unwrap();
        // The reservation was committed, but its acknowledgment was lost.
        let Ingress::Reserve {
            frame,
            publication: retry,
        } = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("retry reserve")
        };
        assert_eq!(retry, publication);
        ack(
            &client,
            &subject,
            &acknowledgment(
                frame,
                None,
                serde_json::json!({"producer_bytes":1,"complete":null}),
            ),
        )
        .await;
        let sent: Ingress = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap();
        assert_eq!(sent, frozen.frame);
        let mut body = spool::snapshot(
            &root,
            "polylogue",
            &publication,
            sinex_protocol::producer_bytes::Role::Material,
        )
        .unwrap();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut body, &mut bytes).unwrap();
        assert_eq!(bytes, b"original private bytes");
        let id = publication.material.as_ref().unwrap().id;
        ack(
            &client,
            &subject,
            &acknowledgment(publication.frame, Some(id), serde_json::json!({})),
        )
        .await;
        id
    });
    let body = b"original private bytes";
    let hash = hashed_blobs::Hash(*blake3::hash(body).as_bytes());
    let call = tokio::spawn(async move {
        sinex
            .material(
                "incomplete".into(),
                None,
                &hash,
                || Ok(source(body)),
                serde_json::json!({"manifest":"original"}),
                Class::Open,
            )
            .await
    });
    let frozen = wait_reservation.await.unwrap();
    call.abort();
    assert!(call.await.unwrap_err().is_cancelled());
    let mut retry = Sinex::connect(&config).await.unwrap();
    let actual = retry
        .material(
            "incomplete".into(),
            None,
            &hash,
            || panic!("must not reopen changed source"),
            serde_json::json!({"manifest":"changed"}),
            Class::Open,
        )
        .await
        .unwrap();
    assert_eq!(Some(actual), frozen.publication.material.map(|m| m.id));
    assert_eq!(actual, worker.await.unwrap());
    assert!(
        spool::load(dir.path(), "polylogue", "incomplete")
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn missing_frozen_body_refuses_without_reopening_source_or_sending_final_frame() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (_, mut sinex, mut ingress) = setup(dir.path(), &nats).await;
    let body = b"private bytes";
    let hash = hashed_blobs::Hash(*blake3::hash(body).as_bytes());
    let frozen = sinex
        .freeze_bytes(
            Ingress::Material {
                frame: "missing".into(),
                material: Box::new(MaterialFrame {
                    id: MaterialId::new(),
                    input: InputRef {
                        kind: "polylogue".into(),
                        locator: "missing".into(),
                        host: String::new(),
                    },
                    blob: format!("blake3:{}", hash.hex()),
                    length: body.len() as u64,
                    parser: "polylogue".into(),
                    parser_version: 1,
                    part_index: 0,
                    previous_part: None,
                    manifest: serde_json::json!({}),
                }),
            },
            Class::Open,
            Some(&mut source(body).file),
        )
        .unwrap();
    spool::discard_bytes(dir.path(), "polylogue", &frozen.publication).unwrap();
    let client = sinex.client.clone();
    let subject = sinex.subjects.ack(&sinex.producer);
    let worker = tokio::spawn(async move {
        let Ingress::Reserve { frame, .. } =
            serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("reserve first")
        };
        ack(
            &client,
            &subject,
            &acknowledgment(
                frame,
                None,
                serde_json::json!({"producer_bytes":1,"complete":null}),
            ),
        )
        .await;
        assert!(
            tokio::time::timeout(Duration::from_millis(100), ingress.next())
                .await
                .is_err()
        );
    });
    let result = sinex
        .material(
            "missing".into(),
            None,
            &hash,
            || panic!("lost frozen bytes must not be replaced"),
            serde_json::json!({}),
            Class::Open,
        )
        .await;
    assert!(matches!(result,Err(Error::Invalid(message)) if message.contains("spool unavailable")));
    assert_eq!(
        spool::load(dir.path(), "polylogue", "missing").unwrap(),
        Some(frozen)
    );
    worker.await.unwrap();
}

#[tokio::test]
async fn oversized_row_event_uses_private_envelope_under_original_final_frame() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (_, mut sinex, mut ingress) = setup(dir.path(), &nats).await;
    sinex.max_message = 8192;
    let frame = Ingress::Events {
        frame: "rows:archive:1".into(),
        failures: vec![],
        events: vec![ProducerEvent {
            event_type: "polylogue.rows".into(),
            payload: serde_json::from_str(&format!(r#"{{"z":"{}","a":1}}"#, "x".repeat(16000)))
                .unwrap(),
            location: None,
            ordinal: 0,
            parents: vec![],
            reading_order: 1,
            class: Class::Guarded,
        }],
    };
    let expected = frame.clone();
    assert_eq!(encode_frame(&frame), serde_json::to_vec(&frame).unwrap());
    let root = dir.path().to_owned();
    let client = sinex.client.clone();
    let subject = sinex.subjects.ack(&sinex.producer);
    let worker = tokio::spawn(async move {
        let Ingress::Reserve {
            frame: attempt,
            publication,
        } = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("reserve first")
        };
        assert_eq!(publication.class, Class::Guarded);
        assert_eq!(publication.frame, expected.frame());
        assert!(publication.material.is_none() && publication.envelope.is_some());
        let mut body = spool::snapshot(
            &root,
            "polylogue",
            &publication,
            sinex_protocol::producer_bytes::Role::Envelope,
        )
        .unwrap();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut body, &mut bytes).unwrap();
        assert_eq!(bytes, encode_frame(&expected));
        let loaded = spool::load(&root, "polylogue", expected.frame())
            .unwrap()
            .unwrap();
        assert_eq!(encode_frame(&loaded.frame), bytes);
        ack(
            &client,
            &subject,
            &acknowledgment(
                attempt,
                None,
                serde_json::json!({"producer_bytes":1,"complete":null}),
            ),
        )
        .await;
        let wire: Ingress = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap();
        assert_eq!(wire.frame(), expected.frame());
        assert!(matches!(wire, Ingress::Indirect { .. }));
        ack(
            &client,
            &subject,
            &acknowledgment(publication.frame, None, serde_json::json!({})),
        )
        .await;
    });
    assert_eq!(sinex.send(&frame).await.unwrap().frame, frame.frame());
    worker.await.unwrap();
    assert!(
        spool::load(dir.path(), "polylogue", frame.frame())
            .unwrap()
            .is_none()
    );
}

fn frozen_material(sinex: &Sinex, name: &str) -> Frozen {
    let body = b"synthetic pending body";
    let mut bytes = source(body);
    sinex
        .freeze_bytes(
            Ingress::Material {
                frame: name.into(),
                material: Box::new(MaterialFrame {
                    id: MaterialId::new(),
                    input: InputRef {
                        kind: "polylogue".into(),
                        locator: name.into(),
                        host: String::new(),
                    },
                    blob: format!("blake3:{}", blake3::hash(body)),
                    length: body.len() as u64,
                    parser: "polylogue".into(),
                    parser_version: 1,
                    part_index: 0,
                    previous_part: None,
                    manifest: serde_json::json!({}),
                }),
            },
            Class::Open,
            Some(&mut bytes.file),
        )
        .unwrap()
}

#[tokio::test]
async fn pending_cleanup_checks_only_own_artifacts_and_never_resends_unconfirmed_frames() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (_, mut sinex, mut ingress) = setup(dir.path(), &nats).await;
    let done = frozen_material(&sinex, "done");
    let waiting = frozen_material(&sinex, "waiting");
    let mut unrelated = waiting.clone();
    unrelated.publication.frame = "another-producer".into();
    if let Ingress::Material { frame, .. } = &mut unrelated.frame {
        *frame = "another-producer".into();
    }
    spool::freeze(dir.path(), "other-producer", &unrelated).unwrap();
    // A terminal ACK must permit cleanup even after receiver byte cleanup.
    spool::discard_bytes(dir.path(), "polylogue", &done.publication).unwrap();
    let mut requests = sinex
        .client
        .subscribe(sinex.subjects.have(&sinex.producer))
        .await
        .unwrap();
    sinex.client.flush().await.unwrap();
    let client = sinex.client.clone();
    let id = done.publication.material.as_ref().unwrap().id;
    let worker = tokio::spawn(async move {
        let message = requests.next().await.unwrap();
        let mut request: Have = serde_json::from_slice(&message.payload).unwrap();
        request.frames.sort();
        assert_eq!(request.frames, ["done", "waiting"]);
        let held = vec![acknowledgment(
            "done".into(),
            Some(id),
            serde_json::json!({}),
        )];
        client
            .publish(
                message.reply.unwrap(),
                serde_json::to_vec(&held).unwrap().into(),
            )
            .await
            .unwrap();
        client.flush().await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(100), ingress.next())
                .await
                .is_err()
        );
    });
    sinex.cleanup_completed().await.unwrap();
    worker.await.unwrap();
    assert!(
        spool::load(dir.path(), "polylogue", "done")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        spool::load(dir.path(), "polylogue", "waiting").unwrap(),
        Some(waiting)
    );
    assert_eq!(
        spool::load(dir.path(), "other-producer", "another-producer").unwrap(),
        Some(unrelated)
    );
}

#[tokio::test]
async fn pending_cleanup_preserves_identity_on_refused_or_different_final_ack() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (_, mut sinex, _) = setup(dir.path(), &nats).await;
    let frozen = frozen_material(&sinex, "pending");
    let mut requests = sinex
        .client
        .subscribe(sinex.subjects.have(&sinex.producer))
        .await
        .unwrap();
    sinex.client.flush().await.unwrap();
    let client = sinex.client.clone();
    let worker = tokio::spawn(async move {
        for refused in [true, false] {
            let message = requests.next().await.unwrap();
            let request: Have = serde_json::from_slice(&message.payload).unwrap();
            assert_eq!(request.frames, ["pending"]);
            let mut result = acknowledgment(
                "pending".into(),
                Some(MaterialId::new()),
                serde_json::json!({}),
            );
            result.refusal = refused.then(|| "synthetic refusal".into());
            client
                .publish(
                    message.reply.unwrap(),
                    serde_json::to_vec(&vec![result]).unwrap().into(),
                )
                .await
                .unwrap();
        }
    });
    assert!(matches!(
        sinex.cleanup_completed().await,
        Err(Error::Refused { .. })
    ));
    assert_eq!(
        spool::load(dir.path(), "polylogue", "pending").unwrap(),
        Some(frozen.clone())
    );
    assert!(
        matches!(sinex.cleanup_completed().await,Err(Error::Invalid(message)) if message.contains("differs from frozen"))
    );
    assert_eq!(
        spool::load(dir.path(), "polylogue", "pending").unwrap(),
        Some(frozen)
    );
    worker.await.unwrap();
}
