use super::*;
use futures_util::StreamExt;
use sinex_protocol::{FetchMaterial, FetchMaterialReply, Ingress, MaterialFrame};
use std::os::unix::fs::MetadataExt;
use std::sync::{Arc, Mutex};

fn material(bytes: &[u8]) -> MaterialFrame {
    MaterialFrame {
        source: crate::source_reference(),
        id: sinex_model::MaterialId::new(),
        input: sinex_model::InputRef {
            kind: "file".into(),
            locator: "/synthetic/input".into(),
            host: "fixture".into(),
        },
        blob: format!("blake3:{}", blake3::hash(bytes)),
        length: bytes.len() as u64,
        chunks: vec![],
        parser: "synthetic".into(),
        parser_version: 1,
        part_index: 0,
        previous_part: None,
        manifest: sinex_model::Manifest::from_json(serde_json::json!({})),
    }
}

fn delivery(remote: bool) {
    let server = server();
    let root = tempfile::tempdir().unwrap();
    let remote_root = tempfile::tempdir().unwrap();
    let mut config = config(&server, root.path(), None);
    if remote {
        config.upload_root = Some(root.path().join("upload"));
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut owner = ProducerOwner::start(&server, "MATERIAL_INGRESS").await;
        let spool = Arc::new(Mutex::new(
            crate::spool::Spool::open(root.path(), "polylogue").unwrap(),
        ));
        let mut bytes = vec![0; 192 * 1024];
        blake3::Hasher::new()
            .update(b"synthetic material handoff")
            .finalize_xof()
            .fill(&mut bytes);
        let metadata = material(&bytes);
        let frame = Ingress::Material {
            frame: "material-frame".into(),
            material: Box::new(metadata.clone()),
        };
        let prepared = spool
            .lock()
            .unwrap()
            .prepare_material(frame.clone(), bytes.as_slice())
            .unwrap();
        let source = spool
            .lock()
            .unwrap()
            .encoded_body(&prepared.identity)
            .unwrap()[0]
            .0
            .clone();
        let source_inode = std::fs::metadata(&source).unwrap().ino();
        let authentication = Authentication::Password(PASSWORD.into());
        let (client, ()) = tokio::join!(
            super::super::Client::connect_shared(
                &config,
                &authentication,
                spool,
                Default::default(),
                Default::default(),
                false
            ),
            owner.welcome(),
        );
        let client = client.unwrap();
        let receive = async {
            let mut frames = 0;
            loop {
                let message = owner.ingress.next().await.unwrap();
                let received: Ingress = serde_json::from_slice(&message.payload).unwrap();
                frames += 1;
                let mut acknowledgment = crate::tests::no_event_acknowledgment(&received);
                acknowledgment.number = frames;
                match &received {
                    Ingress::Chunk {
                        for_frame,
                        blob,
                        descriptor,
                        offset,
                        data,
                        ..
                    } => {
                        assert!(
                            remote,
                            "same-host material must arrive as one logical frame"
                        );
                        assert_eq!(for_frame, "material-frame");
                        assert_eq!(blob, &metadata.blob);
                        use base64::Engine;
                        let descriptor = hashed_blobs::EncodedBlob::decode(descriptor).unwrap();
                        sinex_transport::frame_bytes::receive(
                            remote_root.path(),
                            "polylogue",
                            for_frame,
                            &descriptor,
                            *offset,
                            &base64::engine::general_purpose::STANDARD
                                .decode(data)
                                .unwrap(),
                        )
                        .unwrap();
                    }
                    Ingress::Material { material, .. } => {
                        assert_eq!(material.id, metadata.id);
                        let handoff = if remote {
                            remote_root.path()
                        } else {
                            root.path()
                        };
                        let (_, pinned) = sinex_transport::frame_bytes::encoded(
                            handoff,
                            "polylogue",
                            "material-frame",
                            &metadata.blob,
                        )
                        .unwrap();
                        if !remote {
                            assert_eq!(pinned.metadata().unwrap().ino(), source_inode);
                        }
                        let mut actual = Vec::new();
                        use std::io::Read;
                        sinex_transport::frame_bytes::reader(
                            handoff,
                            "polylogue",
                            "material-frame",
                            &metadata.blob,
                            metadata.length,
                        )
                        .unwrap()
                        .read_to_end(&mut actual)
                        .unwrap();
                        assert_eq!(actual, bytes);
                        acknowledgment.result = sinex_protocol::FrameResult::Admitted {
                            material: Some(metadata.id),
                            material_extents: vec![crate::tests::extent(handoff, &bytes).encode()],
                            admitted: vec![],
                            counts: Default::default(),
                            commit_seq: None,
                        };
                        owner.acknowledge(acknowledgment).await;
                        assert_eq!(frames == 1, !remote);
                        break;
                    }
                    other => panic!("unexpected frame: {other:?}"),
                }
                owner.acknowledge(acknowledgment).await;
            }
        };
        let (result, ()) = tokio::join!(client.publish_prepared(prepared), receive);
        assert_eq!(result.unwrap().refusal(), None);
    });
}

#[test]
fn local_material_is_one_frame_and_keeps_the_frozen_inode() {
    delivery(false);
}
#[test]
fn remote_material_uploads_bytes_before_its_logical_frame() {
    delivery(true);
}

#[test]
fn fetch_material_returns_bytes_and_ranges_without_a_shared_store() {
    let server = server();
    let root = tempfile::tempdir().unwrap();
    let config = config(&server, &root.path().join("absent-shared-store"), None);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut owner = ProducerOwner::start(&server, "FETCH_ONLY").await;
        let subjects = Subjects::new("auth");
        let mut fetches = owner
            .client
            .subscribe(subjects.fetch_material("polylogue"))
            .await
            .unwrap();
        owner.client.flush().await.unwrap();
        let spool = Arc::new(Mutex::new(
            crate::spool::Spool::open(root.path(), "polylogue").unwrap(),
        ));
        let authentication = Authentication::Password(PASSWORD.into());
        let (client, ()) = tokio::join!(
            super::super::Client::connect_shared(
                &config,
                &authentication,
                spool,
                Default::default(),
                Default::default(),
                false
            ),
            owner.welcome(),
        );
        let client = client.unwrap();
        let metadata = material(b"0123456789");
        let request = FetchMaterial {
            material: metadata.id,
            offset: 2,
            end: Some(9),
            byte_budget: 3,
        };
        for invalid_continuation in [false, true] {
            let respond = async {
                let message = fetches.next().await.unwrap();
                assert_eq!(
                    serde_json::from_slice::<FetchMaterial>(&message.payload).unwrap(),
                    request
                );
                sinex_transport::replies::publish_json(
                    &owner.client,
                    message.reply.unwrap().to_string(),
                    FetchMaterialReply::Material {
                        material: Box::new(metadata.clone()),
                        offset: 2,
                        bytes: b"234".to_vec(),
                        next_offset: (!invalid_continuation).then_some(5),
                    },
                    Duration::from_secs(5),
                )
                .await
                .unwrap();
            };
            let (page, ()) = tokio::join!(client.fetch_material_page(request.clone()), respond);
            if invalid_continuation {
                assert!(
                    matches!(page, Err(Error::Invalid(reason)) if reason.contains("byte range"))
                );
            } else {
                let page = page.unwrap();
                assert_eq!(page.bytes, b"234");
                assert_eq!(page.offset, 2);
                assert_eq!(page.next_offset, Some(5));
            }
        }
        assert!(!config.store.exists());
    });
}

#[test]
fn non_loopback_plaintext_nats_is_refused_before_producer_hello() {
    let server = server();
    let root = tempfile::tempdir().unwrap();
    let mut config = config(&server, root.path(), None);
    let address: async_nats::ServerAddr = server.url().parse().unwrap();
    // Linux routes an unspecified destination to the local listener. It is
    // intentionally outside the client's loopback exception, without DNS.
    config.nats = format!("nats://0.0.0.0:{}", address.port());
    config.upload_root = Some(root.path().join("upload"));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let plain =
            async_nats::ConnectOptions::with_user_and_password("polylogue".into(), PASSWORD.into())
                .connect(config.nats.as_str())
                .await
                .unwrap();
        plain.flush().await.unwrap();
        drop(plain);
        let mut owner = ProducerOwner::start(&server, "TLS_INGRESS").await;
        // Without the TLS requirement this responder completes the same
        // production connect immediately, making the refusal assertion fail.
        let welcome = tokio::spawn(async move {
            owner.welcome().await;
        });
        let spool = Arc::new(Mutex::new(
            crate::spool::Spool::open(root.path(), "polylogue").unwrap(),
        ));
        let result = super::super::Client::connect_shared(
            &config,
            &Authentication::Password(PASSWORD.into()),
            spool,
            Default::default(),
            Default::default(),
            false,
        )
        .await;
        assert!(
            matches!(result, Err(Error::Unreachable(_))),
            "plaintext route must not establish the producer client"
        );
        assert!(
            !welcome.is_finished(),
            "no producer hello reaches the plaintext broker"
        );
        welcome.abort();
        let _ = welcome.await;
    });
}
