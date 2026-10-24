use super::*;
use crate::{Config, Mode, parse_content_name};
use async_nats::jetstream;
use futures_util::StreamExt;
use sinex_protocol::{Have, ProducerEvent};
use std::path::Path;
use std::time::Duration;
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

/// A source that must not be read: the spooled request is reused.
struct Unread(&'static str);

impl std::io::Read for Unread {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        panic!("{}", self.0)
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
async fn a_cancelled_send_is_resent_from_the_spool_and_a_completed_request_needs_no_bytes() {
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
            publication: request,
        } = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("the request comes first")
        };
        assert_eq!(request.class, Class::Open);
        assert_eq!(request.frame, "stable-material");
        assert!(request.material.is_some() && request.envelope.is_some());
        let spooled = spool::load(&root, "polylogue", &request.frame)
            .unwrap()
            .unwrap();
        let id = request.material.as_ref().unwrap().id;
        for (role, _) in request.roles() {
            spool::snapshot(&root, "polylogue", &request, role).unwrap();
        }
        // Nothing reaches Sinex's content store before Sinex takes the request.
        let public = hashed_blobs::Store::open(root.clone(), vec![], 0o600).unwrap();
        let hash = parse_content_name(&request.material.as_ref().unwrap().bytes.blob).unwrap();
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
        assert_eq!(wire, spooled.wire_frame());
        assert!(matches!(wire, Ingress::Indirect { .. }));
        observed.send((spooled.clone(), id)).unwrap();
        // Sinex committed the frame but its final acknowledgment was lost: it removed
        // its roles, but cannot remove the sender's durable typed identity.
        spool::discard_bytes(&root, "polylogue", &request).unwrap();
        let Ingress::Reserve {
            frame: retry_attempt,
            publication: retry,
        } = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("the retried request comes first")
        };
        assert_ne!(retry_attempt, first_attempt);
        assert_eq!(retry, request);
        let done = acknowledgment(request.frame.clone(), Some(id), serde_json::json!({}));
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
                &hash,
                bytes.len() as u64,
                &mut &bytes[..],
                serde_json::json!({"large":"x".repeat(16000)}),
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
            &hash,
            0,
            &mut Unread("a retry reuses the spooled bytes and material ID"),
            serde_json::json!({"changed":"must not replace the spooled manifest"}),
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
async fn a_sinex_without_spooled_bytes_refuses_and_the_request_stays_spooled() {
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
            &hash,
            bytes.len() as u64,
            &mut &bytes[..],
            serde_json::json!({}),
        )
        .await;
    assert!(matches!(result, Err(Error::Invalid(message)) if message.contains("does not accept")));
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
async fn a_cancelled_incomplete_request_resends_the_same_material_and_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (config, mut sinex, mut ingress) = setup(dir.path(), &nats).await;
    let client = sinex.client.clone();
    let subject = sinex.subjects.ack(&sinex.producer);
    let (requested, wait_request) = oneshot::channel();
    let root = dir.path().to_owned();
    let worker = tokio::spawn(async move {
        let Ingress::Reserve {
            publication: request,
            ..
        } = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("the request comes first")
        };
        assert!(request.envelope.is_none());
        let spooled = spool::load(&root, "polylogue", &request.frame)
            .unwrap()
            .unwrap();
        requested.send(spooled.clone()).unwrap();
        // Sinex committed the request, but its acknowledgment was lost.
        let Ingress::Reserve {
            frame,
            publication: retry,
        } = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("the retried request comes first")
        };
        assert_eq!(retry, request);
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
        assert_eq!(sent, spooled.frame);
        let mut body = spool::snapshot(
            &root,
            "polylogue",
            &request,
            sinex_protocol::producer_bytes::Role::Material,
        )
        .unwrap();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut body, &mut bytes).unwrap();
        assert_eq!(bytes, b"original private bytes");
        let id = request.material.as_ref().unwrap().id;
        ack(
            &client,
            &subject,
            &acknowledgment(request.frame, Some(id), serde_json::json!({})),
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
                &hash,
                body.len() as u64,
                &mut &body[..],
                serde_json::json!({"manifest":"original"}),
            )
            .await
    });
    let spooled = wait_request.await.unwrap();
    call.abort();
    assert!(call.await.unwrap_err().is_cancelled());
    let mut retry = Sinex::connect(&config).await.unwrap();
    let actual = retry
        .material(
            "incomplete".into(),
            &hash,
            0,
            &mut Unread("must not reopen changed source"),
            serde_json::json!({"manifest":"changed"}),
        )
        .await
        .unwrap();
    assert_eq!(Some(actual), spooled.publication.material.map(|m| m.id));
    assert_eq!(actual, worker.await.unwrap());
    assert!(
        spool::load(dir.path(), "polylogue", "incomplete")
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn missing_spooled_bytes_refuse_without_reading_the_source_again() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (_, mut sinex, mut ingress) = setup(dir.path(), &nats).await;
    let body = b"private bytes";
    let hash = hashed_blobs::Hash(*blake3::hash(body).as_bytes());
    let spooled = sinex
        .spool(
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
            Some(&mut &body[..]),
        )
        .unwrap();
    spool::discard_bytes(dir.path(), "polylogue", &spooled.publication).unwrap();
    let client = sinex.client.clone();
    let subject = sinex.subjects.ack(&sinex.producer);
    let worker = tokio::spawn(async move {
        let Ingress::Reserve { frame, .. } =
            serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("the request comes first")
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
            &hash,
            0,
            &mut Unread("lost spooled bytes must not be replaced from the source"),
            serde_json::json!({}),
        )
        .await;
    assert!(matches!(result,Err(Error::Invalid(message)) if message.contains("spool unavailable")));
    assert_eq!(
        spool::load(dir.path(), "polylogue", "missing").unwrap(),
        Some(spooled)
    );
    worker.await.unwrap();
}

#[tokio::test]
async fn an_oversized_events_frame_travels_as_a_spooled_envelope() {
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
            class: Class::Open,
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
            publication: request,
        } = serde_json::from_slice(&ingress.next().await.unwrap().payload).unwrap()
        else {
            panic!("the request comes first")
        };
        assert_eq!(request.class, Class::Open);
        assert_eq!(request.frame, expected.frame());
        assert!(request.material.is_none() && request.envelope.is_some());
        let mut body = spool::snapshot(
            &root,
            "polylogue",
            &request,
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
            &acknowledgment(request.frame, None, serde_json::json!({})),
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

fn spooled_material(sinex: &Sinex, name: &str) -> Spooled {
    let body = b"synthetic pending body";
    sinex
        .spool(
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
            Some(&mut &body[..]),
        )
        .unwrap()
}

#[tokio::test]
async fn finishing_the_spool_asks_only_about_its_own_requests_and_resends_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (_, mut sinex, mut ingress) = setup(dir.path(), &nats).await;
    let done = spooled_material(&sinex, "done");
    let waiting = spooled_material(&sinex, "waiting");
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
    sinex.finish_staged().await.unwrap();
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
async fn finishing_the_spool_keeps_a_request_on_a_refused_or_different_acknowledgment() {
    let dir = tempfile::tempdir().unwrap();
    let nats = devkit::nats::Server::start().unwrap();
    let (_, mut sinex, _) = setup(dir.path(), &nats).await;
    let spooled = spooled_material(&sinex, "pending");
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
        sinex.finish_staged().await,
        Err(Error::Refused { .. })
    ));
    assert_eq!(
        spool::load(dir.path(), "polylogue", "pending").unwrap(),
        Some(spooled.clone())
    );
    assert!(
        matches!(sinex.finish_staged().await,Err(Error::Invalid(message)) if message.contains("differs from the spooled"))
    );
    assert_eq!(
        spool::load(dir.path(), "polylogue", "pending").unwrap(),
        Some(spooled)
    );
    worker.await.unwrap();
}
