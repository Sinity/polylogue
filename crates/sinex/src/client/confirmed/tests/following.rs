use super::*;
use sinex_protocol::{ConfirmedConsumerReply, FetchEventReply, Subjects};
use std::sync::Arc;

async fn reply(
    daemon: &async_nats::Client,
    request: async_nats::Message,
    body: impl serde::Serialize + Send + 'static,
) {
    sinex_transport::replies::publish_json(
        daemon,
        request.reply.unwrap().to_string(),
        body,
        Duration::from_secs(5),
    )
    .await
    .unwrap();
}

async fn connections(
    server: &devkit::nats::Server,
    root: &std::path::Path,
    spool: Arc<std::sync::Mutex<crate::spool::Spool>>,
) -> (async_nats::Client, Client) {
    let daemon = async_nats::ConnectOptions::with_user_and_password(
        "daemon".into(),
        "synthetic-daemon-password".into(),
    )
    .connect(server.url())
    .await
    .unwrap();
    let subjects = Subjects::new("auth");
    let stream = jetstream::new(daemon.clone())
        .get_or_create_stream(jetstream::stream::Config {
            name: subjects.confirmed_stream(),
            subjects: vec![subjects.confirmed_control()],
            ..Default::default()
        })
        .await
        .unwrap();
    stream
        .create_consumer(jetstream::consumer::pull::Config {
            durable_name: Some(subjects.consumer_name("test-scope")),
            ack_policy: jetstream::consumer::AckPolicy::Explicit,
            ..Default::default()
        })
        .await
        .unwrap();
    let client = Client {
        client: async_nats::ConnectOptions::with_user_and_password(
            "polylogue".into(),
            "synthetic-password".into(),
        )
        .custom_inbox_prefix(subjects.inbox_prefix("polylogue"))
        .connect(server.url())
        .await
        .unwrap(),
        subjects,
        producer: "polylogue".into(),
        family: "polylogue".into(),
        max_message: 1 << 20,
        frame_root: root.into(),
        remote_upload: false,
        consumer_cache: None,
        consumer_instance: None,
        health: Default::default(),
        consumer: None,
        pending: Vec::new(),
        spool,
        publication: Default::default(),
        _backup: None,
    };
    (daemon, client)
}

async fn publish(daemon: &async_nats::Client, change: sinex_protocol::Confirmed) {
    jetstream::new(daemon.clone())
        .publish(
            Subjects::new("auth").confirmed_control(),
            serde_json::to_vec(&change).unwrap().into(),
        )
        .await
        .unwrap()
        .await
        .unwrap();
}

fn provisioned() -> ConfirmedConsumerReply {
    let subjects = Subjects::new("auth");
    ConfirmedConsumerReply::Provisioned {
        position: 17,
        stream: subjects.confirmed_stream(),
        stream_instance: Some("native-instance".into()),
        consumer: subjects.consumer_name("test-scope"),
        scope: "test-scope".into(),
    }
}

#[test]
fn background_follow_drains_split_position_after_durable_staging_then_waits_when_current() {
    let server = crate::client::tests::server();
    let root = tempfile::tempdir().unwrap();
    let config = crate::Config {
        mode: crate::Mode::Primary,
        nats: server.url().into(),
        password_file: None,
        subject_root: "auth".into(),
        producer: "polylogue".into(),
        store: root.path().join("spool"),
        upload_root: None,
    };
    let sinex = crate::client::tests::synchronous(
        &config,
        &crate::Authentication::Password("synthetic-password".into()),
    );
    let (archive, _lock) = polylogue_store::Archive::open_primary(
        &root.path().join("index"),
        &root.path().join("bytes"),
    )
    .unwrap();
    let (observed, observations) = std::sync::mpsc::channel();
    let index = archive.clone();
    let (daemon, mut consumers) = sinex.runtime.block_on(async {
        let (daemon, client) = connections(&server, &config.store, sinex.spool.clone()).await;
        *sinex.client.lock().unwrap() = Some(client);
        for number in 1..=300 {
            publish(
                &daemon,
                sinex_protocol::Confirmed::Tombstoned {
                    event: EventId(uuid::Uuid::from_u128(number)),
                    commit_seq: Some(17),
                    original_commit_seq: Some(1),
                    operation: sinex_model::OperationId(uuid::Uuid::from_u128(999)),
                },
            )
            .await;
        }
        publish(
            &daemon,
            sinex_protocol::Confirmed::PositionComplete { commit_seq: 17 },
        )
        .await;
        let consumers = daemon
            .subscribe(Subjects::new("auth").confirmed_consumer("polylogue"))
            .await
            .unwrap();
        daemon.flush().await.unwrap();
        (daemon, consumers)
    });
    let admin = daemon.clone();
    let owner = sinex.runtime.spawn(async move {
        let mut number = 0;
        while let Some(request) = consumers.next().await {
            number += 1;
            let query: sinex_protocol::ConfirmedConsumer =
                serde_json::from_slice(&request.payload).unwrap();
            let staged = if number == 3 {
                let pool = polylogue_store::ReadPool::open(&index).unwrap();
                let reader = pool.reader().unwrap();
                Some((
                    reader.confirmed_checkpoint().unwrap(),
                    reader
                        .query("SELECT count(*) AS n FROM primary_pending", &[])
                        .unwrap(),
                    query.after,
                ))
            } else {
                None
            };
            let mut status = provisioned();
            if number <= 2 {
                // Delivery may include a commit newer than the sampled head.
                let ConfirmedConsumerReply::Provisioned { position, .. } = &mut status else {
                    unreachable!()
                };
                *position = 0;
            }
            reply(&daemon, request, status).await;
            if observed
                .send((number, std::time::Instant::now(), staged))
                .is_err()
            {
                break;
            }
        }
    });
    let sinex = Arc::new(sinex);
    let mut writer = polylogue_store::Writer::start_on_sinex(&archive, sinex.clone()).unwrap();
    // No synchronize request: the production follower itself must drain backlog.
    let first = observations.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(first.0, 1);
    let second = observations.recv_timeout(Duration::from_secs(1)).unwrap();
    let third = observations.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!((second.0, third.0), (2, 3));
    assert_eq!(
        third.2,
        Some((None, vec![json!({"n":256})], 0)),
        "a split commit is staged before it becomes visible"
    );
    assert!(third.1.duration_since(second.1) < Duration::from_secs(1));
    let empty = observations.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(
        empty.0, 4,
        "a final empty probe observes that delivery has drained"
    );
    let pool = polylogue_store::ReadPool::open(&archive).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while pool.reader().unwrap().confirmed_checkpoint().unwrap() != Some(17) {
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        pool.reader()
            .unwrap()
            .query("SELECT count(*) AS n FROM primary_fact", &[])
            .unwrap(),
        [json!({"n":300})]
    );
    assert!(
        observations
            .recv_timeout(Duration::from_millis(150))
            .is_err(),
        "a current stream must idle instead of spinning"
    );
    let stopped = std::time::Instant::now();
    writer.stop();
    assert!(stopped.elapsed() < Duration::from_secs(1));
    // Reprovision the native consumer from its beginning while retaining the
    // applied cache. Old deliveries still need draining and durable ACKs.
    sinex.runtime.block_on(async {
        let subjects = Subjects::new("auth");
        let stream = jetstream::new(admin)
            .get_stream(subjects.confirmed_stream())
            .await
            .unwrap();
        stream
            .delete_consumer(&subjects.consumer_name("test-scope"))
            .await
            .unwrap();
        stream
            .create_consumer(jetstream::consumer::pull::Config {
                durable_name: Some(subjects.consumer_name("test-scope")),
                ack_policy: jetstream::consumer::AckPolicy::Explicit,
                ..Default::default()
            })
            .await
            .unwrap();
        let mut connection = sinex.client.lock().unwrap();
        let client = connection.as_mut().unwrap();
        assert!(client.pending.is_empty());
        client.consumer = None;
        client.consumer_cache = None;
        client.consumer_instance = None;
    });
    let mut resumed = polylogue_store::Writer::start_on_sinex(&archive, sinex.clone()).unwrap();
    assert_eq!(
        observations.recv_timeout(Duration::from_secs(5)).unwrap().0,
        5
    );
    assert_eq!(
        observations.recv_timeout(Duration::from_secs(1)).unwrap().0,
        6
    );
    assert_eq!(
        observations.recv_timeout(Duration::from_secs(1)).unwrap().0,
        7
    );
    assert_eq!(
        pool.reader().unwrap().confirmed_checkpoint().unwrap(),
        Some(17)
    );
    assert_eq!(
        pool.reader()
            .unwrap()
            .query("SELECT count(*) AS n FROM primary_fact", &[])
            .unwrap(),
        [json!({"n":300})]
    );
    assert!(
        observations
            .recv_timeout(Duration::from_millis(150))
            .is_err(),
        "an exhausted replay must idle"
    );
    resumed.stop();
    owner.abort();
}

async fn answer_fetch(
    daemon: &async_nats::Client,
    request: async_nats::Message,
    template: &ConfirmedEvent,
) {
    let query: sinex_protocol::FetchEvent = serde_json::from_slice(&request.payload).unwrap();
    let number = query.event.0.as_u128();
    let result = if number == 9 {
        FetchEventReply::Refused(sinex_protocol::FetchRefusal {
            kind: sinex_protocol::FetchRefusalKind::Denied,
            reason: "exact grant refusal".into(),
        })
    } else if number == 36 {
        FetchEventReply::Erased(sinex_protocol::ErasedEvent {
            event: query.event,
            original_commit_seq: 17,
            commit_seq: 18,
        })
    } else {
        let mut fetched = template.clone();
        fetched.id = query.event;
        match number {
            18 => fetched.commit_seq = 18,
            19 => fetched.payload = json!({"changed":"payload"}),
            20 => fetched.kind_definition_hash = "0".repeat(64),
            _ => (),
        }
        FetchEventReply::Event(Box::new(fetched))
    };
    reply(daemon, request, result).await;
}

#[test]
fn confirmed_fetches_overlap_with_a_bound_and_keep_order_refusals_and_erasure() {
    let server = crate::client::tests::server();
    let root = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(10), async {
            let spool = Arc::new(std::sync::Mutex::new(crate::spool::Spool::open(root.path(), "polylogue").unwrap()));
            let (daemon, mut client) = connections(&server, root.path(), spool).await;
            let subjects = Subjects::new("auth");
            let mut consumers = daemon.subscribe(subjects.confirmed_consumer("polylogue")).await.unwrap();
            let mut fetches = daemon.subscribe(subjects.fetch_event("polylogue")).await.unwrap();
            daemon.flush().await.unwrap();
            for number in 1..=36 {
                publish(&daemon, sinex_protocol::Confirmed::EventReference {
                    commit_seq: 17, event: EventId(uuid::Uuid::from_u128(number)),
                }).await;
            }
            publish(&daemon, sinex_protocol::Confirmed::PositionComplete { commit_seq: 17 }).await;
            let (inflight, first_fetches) = tokio::sync::oneshot::channel();
            let (resume, resumed) = tokio::sync::oneshot::channel();
            let template = current_event(&document(polylogue_model::batch::StepProgress::AtEnd));
            let owner = async move {
                let request = consumers.next().await.unwrap();
                reply(&daemon, request, provisioned()).await;
                let mut interrupted = Vec::new();
                for _ in 0..16 { interrupted.push(fetches.next().await.unwrap()); }
                inflight.send(()).unwrap();
                resumed.await.unwrap();
                // Their readers were cancelled; the framed reply has nobody
                // to acknowledge it. The resumed call owns fresh reply inboxes.
                drop(interrupted);
                let request = consumers.next().await.unwrap();
                reply(&daemon, request, provisioned()).await;
                let mut count = 0;
                for size in [16, 16, 4] {
                    let mut requests = Vec::new();
                    for _ in 0..size { requests.push(fetches.next().await.unwrap()); }
                    assert!(tokio::time::timeout(Duration::from_millis(20), fetches.next()).await.is_err(), "payload fetch concurrency exceeded its bound");
                    requests.sort_by_key(|request| {
                        serde_json::from_slice::<sinex_protocol::FetchEvent>(&request.payload).unwrap().event.0.as_u128()
                    });
                    let first = requests.remove(0);
                    // Complete later events before the first one, without
                    // serializing every framed response and its acknowledgements.
                    futures_util::stream::iter(requests.into_iter().rev().map(|request| answer_fetch(&daemon, request, &template)))
                        .buffer_unordered(16)
                        .collect::<Vec<_>>().await;
                    answer_fetch(&daemon, first, &template).await;
                    count += size;
                }
                assert_eq!(count, 36);
            };
            // Both sides and all response futures belong to this join. A
            // failure or timeout drops the whole exchange, not detached workers.
            let reader = async {
                let state = ReplayState { scope: Some("test-scope".into()), ..Default::default() };
                {
                    let attempt = client.confirmed(&state, "cache");
                    tokio::pin!(attempt);
                    tokio::select! {
                        ready = first_fetches => ready.unwrap(),
                        result = &mut attempt => panic!("held fetch returned before cancellation: {result:?}"),
                    }
                }
                assert_eq!(client.pending.len(), 37, "cancelling payload reads must retain unacknowledged delivery");
                resume.send(()).unwrap();
                client.release_pending().await;
                tokio::time::sleep(Duration::from_millis(1100)).await;
                let page = client.confirmed(&state, "cache").await.unwrap();
                assert_eq!(page.len(), 39);
                for (number, item) in (1..=36).zip(&page[1..37]) {
                    assert_eq!(item.position, 17);
                    assert_eq!(item.event, EventId(uuid::Uuid::from_u128(number)).to_string());
                    match number {
                        9 => assert!(matches!(&item.change, ConfirmedChange::Refused { reason, .. } if reason.contains("exact grant refusal"))),
                        18..=20 => assert!(matches!(&item.change, ConfirmedChange::Refused { .. })),
                        36 => assert!(matches!(&item.change, ConfirmedChange::Tombstoned)),
                        _ => assert!(matches!(&item.change, ConfirmedChange::Live(_))),
                    }
                }
                assert!(matches!(&page[37].change, ConfirmedChange::Control(c) if matches!(c.as_ref(), ReplayControl::Completion { position: 17, delivery: Some(receipt) } if receipt.stream_sequence == 37 && receipt.fingerprint.len() == 64)));
                assert!(matches!(&page[38].change, ConfirmedChange::Control(c) if matches!(c.as_ref(), ReplayControl::Delivered { delivery } if delivery.stream_sequence == 37)));
                // Transport delivery alone cannot authorize ACK or retire producer custody.
                assert!(client.acknowledge(&state, &|_| Ok(false)).await.unwrap_err().to_string().contains("not been durably staged"));
                assert_eq!(client.pending.len(), 37);
            };
            tokio::join!(owner, reader);
        }).await.expect("bounded confirmed transport fixture");
    });
}

#[test]
fn reconnect_requests_the_completed_receipt_and_delivers_the_head_marker_without_new_data() {
    let server = crate::client::tests::server();
    let root = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(10), async {
            let spool = Arc::new(std::sync::Mutex::new(crate::spool::Spool::open(root.path(), "polylogue").unwrap()));
            let (daemon, mut client) = connections(&server, root.path(), spool).await;
            let subjects = Subjects::new("auth");
            for number in 1..=36 {
                publish(&daemon, sinex_protocol::Confirmed::EventReference {
                    commit_seq: 17, event: EventId(uuid::Uuid::from_u128(number)),
                }).await;
            }
            publish(&daemon, sinex_protocol::Confirmed::PositionComplete { commit_seq: 17 }).await;
            let stream = jetstream::new(daemon.clone()).get_stream(subjects.confirmed_stream()).await.unwrap();
            let raw = stream.get_raw_message(37).await.unwrap();
            let receipt = local::CompletedDelivery {
                stream_instance: "native-instance".into(), stream_sequence: 37,
                fingerprint: sinex_protocol::DeliveredCompletion::fingerprint(37, raw.time.unix_timestamp_nanos(), raw.subject.as_str(), &raw.payload),
            };
            let name = subjects.consumer_name("test-scope");
            stream.delete_consumer(&name).await.unwrap();
            let consumer = stream.create_consumer(jetstream::consumer::pull::Config {
                durable_name: Some(name), ack_policy: jetstream::consumer::AckPolicy::Explicit,
                deliver_policy: jetstream::consumer::DeliverPolicy::ByStartSequence { start_sequence: 37 },
                ..Default::default()
            }).await.unwrap();
            assert_eq!(consumer.cached_info().delivered.consumer_sequence, 0);
            let mut requests = daemon.subscribe(subjects.confirmed_consumer("polylogue")).await.unwrap();
            daemon.flush().await.unwrap();
            let state = ReplayState {
                applied_position: Some(17), scope: Some("test-scope".into()),
                completed_delivery: Some(receipt.clone()),
                staged_delivery: Some(local::NativeDeliveryCursor { stream_instance: "native-instance".into(), stream_sequence: 99 }),
                ..Default::default()
            };
            let owner = async {
                let request = requests.next().await.unwrap();
                let body: sinex_protocol::ConfirmedConsumer = serde_json::from_slice(&request.payload).unwrap();
                assert_eq!(body.after, 17);
                assert_eq!(body.completion.as_ref().unwrap().stream_sequence, 37, "a newer partial cursor cannot replace completion");
                assert_eq!(body.completion.as_ref().unwrap().fingerprint, receipt.fingerprint);
                reply(&daemon, request, provisioned()).await;
            };
            let read = async {
                let page = client.confirmed(&state, "cache").await.unwrap();
                assert_eq!(client.pending.len(), 1, "native delivery excludes all original data");
                assert_eq!(page.len(), 3);
                assert!(matches!(&page[1].change, ConfirmedChange::Control(control)
                    if matches!(control.as_ref(), ReplayControl::Completion { position: 17, delivery: Some(actual) } if actual == &receipt)));
            };
            tokio::join!(owner, read);
        }).await.expect("bounded reconnect fixture");
    });
}
