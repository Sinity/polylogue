use super::*;
use serde_json::json;

struct Fixture {
    _root: tempfile::TempDir,
    client: crate::Sinex,
    entry: crate::spool::Prepared,
    material: MaterialFrame,
    request: RecordedContinuationRequest,
    binding: RecordedMaterialBinding,
}
fn prepared() -> Fixture {
    let root = tempfile::tempdir().unwrap();
    let client = crate::tests::client(root.path());
    let encoded: serde_json::Value =
        serde_json::from_str(include_str!("../../producer/requests/wire/oracle.json")).unwrap();
    let case = &encoded["cases"][0];
    let frame: Ingress = serde_json::from_value(case["material"].clone()).unwrap();
    let Ingress::Material { material, .. } = &frame else {
        panic!("material")
    };
    let material = material.as_ref().clone();
    let body = case["body"].as_str().unwrap().as_bytes();
    let (entry, request) = {
        let queue = client.queue().unwrap();
        let entry = queue.prepare_material(frame, body).unwrap();
        let request = selector(&queue, &entry, Vec::new()).unwrap();
        (entry, request)
    };
    let receipt=serde_json::from_value(json!({"number":11,"frame":entry.identity,"material":material.id,"admitted":[],"outcome":{"frame_content_hash":entry.frame.blob},"refusal":null,"commit_seq":null})).unwrap();
    let binding = RecordedMaterialBinding {
        receipt,
        material: material.id,
        blob: material.blob.clone(),
        length: material.length,
        source: material.source.clone(),
        source_fingerprint: crate::client::confirmed::source3::source_fingerprint().into(),
        family: "polylogue".into(),
        write_kind_prefixes: Vec::new(),
        producer_host: None,
    };
    Fixture {
        _root: root,
        client,
        entry,
        material,
        request,
        binding,
    }
}

#[test]
fn retained_binding_survives_lost_current_permission_without_becoming_publishable() {
    let Fixture {
        _root,
        client: _client,
        entry,
        material,
        request,
        binding,
    } = prepared();
    let reply = RecordedContinuation::Retained {
        request: request.clone(),
        binding: Box::new(binding),
        headers: RecordedHeaders::NotPermitted {
            reason: "current write grant removed".into(),
        },
    };
    let observed =
        observation("polylogue-test", &entry, &material, &request, reply.clone()).unwrap();
    assert_eq!(observed.custody.frame, entry.identity);
    assert!(matches!(
        observed.headers,
        RecordedHeaders::NotPermitted { .. }
    ));
    assert!(permitted(&entry, &material, &request, reply).is_err());
}

#[test]
fn unknown_cross_binding_and_excised_receipts_never_enter_queued_projection() {
    let Fixture {
        _root,
        client: _client,
        entry,
        material,
        request,
        binding,
    } = prepared();
    let unknown = RecordedContinuation::Unknown {
        request: request.clone(),
        reason: "legacy binding absent".into(),
    };
    assert!(observation("polylogue-test", &entry, &material, &request, unknown).is_err());
    for field in [
        "source",
        "fingerprint",
        "material",
        "blob",
        "length",
        "receipt",
        "hash",
        "refused",
    ] {
        let mut changed = binding.clone();
        match field {
            "source" => changed.source.version = 4,
            "fingerprint" => changed.source_fingerprint = "other".into(),
            "material" => changed.material = crate::material_id("other", "frame"),
            "blob" => changed.blob = format!("blake3:{}", BlobHash::of(b"other")),
            "length" => changed.length += 1,
            "receipt" => changed.receipt.number = None,
            "hash" => {
                changed.receipt.outcome =
                    json!({"frame_content_hash":format!("blake3:{}",BlobHash::of(b"other"))})
            }
            _ => {
                changed.receipt.refusal = Some("excised".into());
                changed.receipt.outcome = json!({});
            }
        }
        let reply = RecordedContinuation::Retained {
            request: request.clone(),
            binding: Box::new(changed),
            headers: RecordedHeaders::PermittedHeaders,
        };
        assert!(
            observation("polylogue-test", &entry, &material, &request, reply).is_err(),
            "{field}"
        );
    }
}

#[test]
fn permitted_headers_are_bound_to_the_exact_request_and_not_historical_prefixes() {
    let Fixture {
        _root,
        client: _client,
        entry,
        material,
        mut request,
        binding,
    } = prepared();
    request.uses.push(
        crate::client::confirmed::source3::header(&sinex_model::KindRef {
            name: "polylogue.note".into(),
            version: 4,
        })
        .unwrap(),
    );
    let reply = RecordedContinuation::Retained {
        request: request.clone(),
        binding: Box::new(binding.clone()),
        headers: RecordedHeaders::PermittedHeaders,
    };
    // A known empty historical prefix array remains provenance only.
    assert!(binding.write_kind_prefixes.is_empty());
    permitted(&entry, &material, &request, reply).unwrap();
    let mut echoed = request.clone();
    echoed.uses[0].kind_definition_hash = "changed".into();
    let reply = RecordedContinuation::Retained {
        request: echoed,
        binding: Box::new(binding),
        headers: RecordedHeaders::PermittedHeaders,
    };
    assert!(permitted(&entry, &material, &request, reply).is_err());
}

#[test]
fn own_recorded_observer_uses_authenticated_read_acl_without_registration_or_current_family() {
    let Fixture {
        _root: root,
        client,
        entry,
        material: _,
        request: expected,
        binding,
    } = prepared();
    drop(client);
    let subjects = sinex_protocol::Subjects::new("retained");
    let authorization = subjects.authorization(
        ("daemon", "synthetic-daemon-password"),
        &[("polylogue-test", "synthetic-producer-password")],
        &[("polylogue-test", "synthetic-scope".into())],
    );
    let server = devkit::nats::Server::start_with(&devkit::nats::Options {
        config: authorization,
    })
    .unwrap();
    let url = server.url().to_string();
    let owner_url = url.clone();
    let (ready, wait) = std::sync::mpsc::channel();
    let owner =
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                use futures_util::StreamExt;
                let client = async_nats::ConnectOptions::with_user_and_password(
                    "daemon".into(),
                    "synthetic-daemon-password".into(),
                )
                .connect(owner_url)
                .await
                .unwrap();
                let mut hello = client
                    .subscribe(subjects.connect("polylogue-test"))
                    .await
                    .unwrap();
                let mut recorded = client
                    .subscribe(subjects.recorded("polylogue-test"))
                    .await
                    .unwrap();
                let mut ingress = client
                    .subscribe(subjects.ingress("polylogue-test"))
                    .await
                    .unwrap();
                client.flush().await.unwrap();
                ready.send(()).unwrap();
                for mode in 0..3 {
                    let request = hello.next().await.unwrap();
                    sinex_transport::replies::publish_json(
                        &client,
                        request.reply.unwrap().to_string(),
                        sinex_protocol::ProducerWelcome {
                            refusal: None,
                            family: Some("polylogue-test".into()),
                            canonical_publisher: false,
                        },
                        std::time::Duration::from_secs(5),
                    )
                    .await
                    .unwrap();
                    let request = tokio::select! {
                        request=recorded.next()=>request.unwrap(),
                        _=ingress.next()=>panic!("a retained observer must not register or submit"),
                    };
                    let query: RecordedQuery = serde_json::from_slice(&request.payload).unwrap();
                    assert_eq!(query.after, i64::MAX);
                    assert_eq!(query.continuations.as_slice(), std::slice::from_ref(&expected));
                    assert!(query.frames.is_empty());
                    assert!(query.numbers.is_empty());
                    let continuations = match mode {
                        0 => vec![RecordedContinuation::Retained {
                            request: expected.clone(),
                            binding: Box::new(binding.clone()),
                            headers: RecordedHeaders::NotPermitted {
                                reason: "current write grant removed".into(),
                            },
                        }],
                        1 => vec![RecordedContinuation::Unknown {
                            request: expected.clone(),
                            reason: "original binding unavailable".into(),
                        }],
                        _ => Vec::new(),
                    };
                    sinex_transport::replies::publish_json(
                        &client,
                        request.reply.unwrap().to_string(),
                        RecordedPage {
                            continuations,
                            ..Default::default()
                        },
                        std::time::Duration::from_secs(5),
                    )
                    .await
                    .unwrap();
                }
            });
        });
    wait.recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    let config = crate::Config {
        mode: crate::Mode::Primary,
        nats: url,
        password_file: None,
        subject_root: "retained".into(),
        producer: "polylogue-test".into(),
        store: root.path().into(),
    };
    let client = crate::Sinex::open(
        &config,
        &crate::Authentication::Password("synthetic-producer-password".into()),
        "current-host",
    )
    .unwrap();
    let mut count = 0;
    client
        .visit_pending_requests(&mut |pending| {
            let polylogue_model::authored::PendingAuthored::ConservedQueued { custody, .. } =
                pending
            else {
                panic!("queued only")
            };
            assert_eq!(custody.frame, entry.identity);
            assert_eq!(custody.receipt_number, 11);
            count += 1;
            Ok(())
        })
        .unwrap();
    for _ in 0..2 {
        assert!(
            client
                .visit_pending_requests(&mut |_| {
                    count += 1;
                    Ok(())
                })
                .is_err()
        );
    }
    assert_eq!(count, 1);
    assert!(
        client
            .queue()
            .unwrap()
            .get(&entry.identity)
            .unwrap()
            .unwrap()
            .phase
            .is_pending(),
        "observation creates no ACK"
    );
    drop(client);
    owner.join().unwrap();
}
