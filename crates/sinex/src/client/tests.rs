use crate::{Authentication, Config, Error, Mode, Sinex};
use sinex_protocol::{ProducerWelcome, Subjects};
use std::{path::Path, time::Duration};
const PASSWORD: &str = "synthetic-password";

pub(super) fn server() -> devkit::nats::Server {
    let config = Subjects::new("auth").authorization(
        ("daemon", "synthetic-daemon-password"),
        &[("polylogue", PASSWORD)],
        &[("polylogue", "test-scope".into())],
    );
    devkit::nats::Server::start_with(&devkit::nats::Options { config }).unwrap()
}
fn config(server: &devkit::nats::Server, store: &Path, password_file: Option<&Path>) -> Config {
    Config {
        mode: Mode::Primary,
        nats: server.url().into(),
        password_file: password_file.map(Path::to_path_buf),
        subject_root: "auth".into(),
        producer: "polylogue".into(),
        store: store.into(),
        upload_root: None,
    }
}

// Low-level transport controls use one connection and explicit draining. Native
// sender tests exercise the independently running worker.
pub(super) fn synchronous(config: &Config, authentication: &Authentication) -> Sinex {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    Sinex {
        config: config.clone(),
        authentication: authentication.clone(),
        health: Default::default(),
        spool: std::sync::Arc::new(std::sync::Mutex::new(
            crate::spool::Spool::open(&config.store, &config.producer).unwrap(),
        )),
        runtime: std::sync::Arc::new(runtime),
        publication: Default::default(),
        sender: None,
        stopping: Some(Default::default()),
        client: std::sync::Mutex::new(None),
        host: "fixture-host".into(),
    }
}

#[test]
fn authenticated_private_reply_and_complete_registration_acknowledgment_establish_family() {
    let server = server();
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("password");
    std::fs::write(&file, format!("{PASSWORD}\n")).unwrap();
    let url = server.url().to_string();
    let (ready, wait) = std::sync::mpsc::channel();
    let owner = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            use futures_util::StreamExt;
            let client=async_nats::ConnectOptions::with_user_and_password("daemon".into(),"synthetic-daemon-password".into()).connect(url).await.unwrap();
            async_nats::jetstream::new(client.clone()).get_or_create_stream(async_nats::jetstream::stream::Config{name:"AUTH_INGRESS".into(),subjects:vec!["auth.ingress.*".into()],..Default::default()}).await.unwrap();
            let mut hello=client.subscribe("auth.connect.polylogue").await.unwrap();
            let mut ingress=client.subscribe("auth.ingress.polylogue").await.unwrap();
            let mut recorded=client.subscribe("auth.recorded.polylogue").await.unwrap();
            let mut consumers=client.subscribe("auth.confirmed_consumer.polylogue").await.unwrap();
            let mut contexts=client.subscribe("auth.public_context.polylogue").await.unwrap();
            client.flush().await.unwrap();ready.send(()).unwrap();
            let request=hello.next().await.unwrap();let inbox=request.reply.unwrap();
            assert!(inbox.starts_with("_INBOX.polylogue."));
            sinex_transport::replies::publish_json(&client,inbox.to_string(),ProducerWelcome::Welcome{family:Some("polylogue".into()),canonical_publisher:true},Duration::from_secs(5)).await.unwrap();
            let frame=ingress.next().await.unwrap();
            let sinex_protocol::Ingress::RegisterKinds{frame,declarations}=serde_json::from_slice(&frame.payload).unwrap() else {panic!("kind registration expected")};
            let hashes:Vec<_>=declarations.iter().map(|declaration|(declaration.kind.clone(),declaration.definition_hash())).collect();
            let mut acknowledgment=crate::tests::no_event_acknowledgment(&sinex_protocol::Ingress::RegisterKinds{frame:frame.clone(),declarations:declarations.clone()});
            acknowledgment.result=sinex_protocol::FrameResult::KindsRegistered{kinds:hashes};
            client.publish("auth.ack.polylogue",serde_json::to_vec(&sinex_protocol::ReceiptHint{number:1,frame_hash:blake3::hash(frame.as_bytes()).to_hex().to_string()}).unwrap().into()).await.unwrap();
            let request=recorded.next().await.unwrap();
            let query:sinex_protocol::RecordedQuery=serde_json::from_slice(&request.payload).unwrap();assert_eq!(query.numbers,[1]);
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),sinex_protocol::RecordedPage{acknowledgments:vec![acknowledgment],next:None,..Default::default()},Duration::from_secs(5)).await.unwrap();
            let request=contexts.next().await.unwrap();
            let query:sinex_protocol::PublicContextRequest=serde_json::from_slice(&request.payload).unwrap();assert!(query.coverage.is_none());
            assert!(request.reply.as_ref().unwrap().starts_with("_INBOX.polylogue."));
            let context:sinex_protocol::PublicContext=serde_json::from_value(serde_json::json!({"state":"available","context_token":"synthetic-context-token","settings_identity":"owner-settings","owner_revision":7,"day":{"zone":"Europe/Warsaw","start":"04:00:00"},"quiet_hours":{"from":1320,"to":420},"interrupt_budget":{"per_hour":3,"used":1,"remaining":2,"window_ms":3600000},"observed_at":"2026-01-01T00:00:00Z","coverage":{"state":"not_requested"}})).unwrap();
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),context,Duration::from_secs(5)).await.unwrap();
            // Exercise real transmission/recorded-acknowledgment lookup for two
            // accepted frames which have no admitted-event position.
            for (number, expected) in [(5, "empty-events"), (6, "empty-kinds")] {
                let message=ingress.next().await.unwrap();
                let frame:sinex_protocol::Ingress=serde_json::from_slice(&message.payload).unwrap();
                assert_eq!(frame.frame(), expected);
                match &frame {
                    sinex_protocol::Ingress::Events{events,failures,..} => assert!(events.is_empty() && failures.is_empty()),
                    sinex_protocol::Ingress::RegisterKinds{declarations,..} => assert!(declarations.is_empty()),
                    _ => panic!("unexpected no-event frame"),
                }
                let mut acknowledgment=crate::tests::no_event_acknowledgment(&frame);
                acknowledgment.number=number;
                client.publish("auth.ack.polylogue",serde_json::to_vec(&sinex_protocol::ReceiptHint{number,frame_hash:blake3::hash(expected.as_bytes()).to_hex().to_string()}).unwrap().into()).await.unwrap();
                let request=recorded.next().await.unwrap();
                let query:sinex_protocol::RecordedQuery=serde_json::from_slice(&request.payload).unwrap();
                assert_eq!(query.numbers,[number]);
                sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),sinex_protocol::RecordedPage{acknowledgments:vec![acknowledgment],next:None,..Default::default()},Duration::from_secs(5)).await.unwrap();
            }
            let subjects=Subjects::new("auth");
            let request=consumers.next().await.unwrap();
            let query:sinex_protocol::ConfirmedConsumer=serde_json::from_slice(&request.payload).unwrap();
            assert_eq!(query,sinex_protocol::ConfirmedConsumer{basis:None,completion:None,after:42,cache:"cache-test".into(),status_only:false});
            let reply=sinex_protocol::ConfirmedConsumerReply::Provisioned{position:50,stream:subjects.confirmed_stream(),stream_instance:Some("instance".into()),consumer:"consumer".into(),scope:"test-scope".into()};
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),reply,Duration::from_secs(5)).await.unwrap();
            let request=consumers.next().await.unwrap();
            let query:sinex_protocol::ConfirmedConsumer=serde_json::from_slice(&request.payload).unwrap();
            assert_eq!(query.after,0,"an index that applied nothing reads the stream from its start");
            let reply=sinex_protocol::ConfirmedConsumerReply::Refused{reason:"synthetic refusal".into()};
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),reply,Duration::from_secs(5)).await.unwrap();
        });
    });
    wait.recv().unwrap();
    let sinex = synchronous(
        &config(&server, root.path(), Some(&file)),
        &Authentication::Password(PASSWORD.into()),
    );
    sinex.read_client(async |_| Ok(())).unwrap();
    assert_eq!(sinex.status().unwrap().family.as_deref(), Some("polylogue"));
    sinex
        .client_operation(async |client| {
            let frame = sinex_protocol::Ingress::RegisterKinds {
                frame: "refused-registration".into(),
                declarations: Vec::new(),
            };
            let hash = sinex_transport::producer::content_hash(&frame).unwrap();
            let acknowledgment = sinex_protocol::Acknowledgment {
                number: 2,
                frame: frame.frame().into(),
                content_hash: Some(hash),
                result: sinex_protocol::FrameResult::Refused {
                    kind: sinex_model::FrameRefusal::Content,
                    reason: "exact owner declaration refusal".into(),
                },
            };
            {
                let spool = client.spool.lock().unwrap();
                let prepared = spool.prepare(frame.clone()).unwrap();
                let mut mismatch = acknowledgment.clone();
                mismatch.content_hash = Some("different payload".into());
                assert!(matches!(
                    super::verified_acknowledgment(&prepared, mismatch),
                    Err(Error::AcknowledgmentConflict { .. })
                ));
                spool.acknowledge(acknowledgment).unwrap();
            }
            for _ in 0..2 {
                assert!(matches!(client.send(frame.clone()).await,
                    Err(Error::Refused { name, reason })
                    if name == frame.frame() && reason == "exact owner declaration refusal"));
            }

            let conflict = sinex_protocol::Ingress::RegisterKinds {
                frame: "conflicting-registration".into(),
                declarations: Vec::new(),
            };
            let offered_hash = sinex_transport::producer::content_hash(&conflict).unwrap();
            let acknowledgment = sinex_protocol::Acknowledgment {
                content_hash: Some("different payload".into()),
                number: 3,
                ..crate::tests::no_event_acknowledgment(&conflict)
            };
            {
                let spool = client.spool.lock().unwrap();
                spool.prepare(conflict.clone()).unwrap();
                spool
                    .record_conflict(conflict.frame(), &offered_hash, &acknowledgment)
                    .unwrap();
            }
            assert!(matches!(client.send(conflict).await,
                Err(Error::Refused { reason, .. })
                if reason == "the original operation is not publishable"));
            Ok(())
        })
        .unwrap();
    let context = sinex
        .context(|zone| jiff::tz::TimeZone::get(zone).map_err(|error| error.to_string()))
        .unwrap();
    assert_eq!(context.settings_identity, "owner-settings");
    assert_eq!(context.owner_revision, 7);
    assert_eq!(context.day.zone_name(), "Europe/Warsaw");
    assert_eq!(context.interrupt_budget.remaining, 2);
    assert!(
        !serde_json::to_string(&context)
            .unwrap()
            .contains("synthetic-context-token")
    );
    assert!(!format!("{context:?}").contains("synthetic-context-token"));
    sinex
        .client_operation(async |client| {
            for frame in [
                sinex_protocol::Ingress::Events {
                    frame: "empty-events".into(),
                    source: None,
                    coverage: None,
                    events: vec![],
                    failures: vec![],
                },
                sinex_protocol::Ingress::RegisterKinds {
                    frame: "empty-kinds".into(),
                    declarations: vec![],
                },
            ] {
                let acknowledgment = client.send(frame.clone()).await?;
                assert_eq!(acknowledgment.commit_seq(), None);
                assert_eq!(
                    client.spool.lock().unwrap().acknowledgment(frame.frame())?,
                    Some(acknowledgment)
                );
            }
            Ok(())
        })
        .unwrap();
    use polylogue_store::primary::{self, ConfirmedChange, ReplayControl};
    let state = primary::ReplayState {
        applied_position: Some(42),
        scope: Some("old-scope".into()),
        head: None,
        staged_delivery: None,
        completed_delivery: None,
        basis: None,
        projection: None,
    };
    // A committed local frame can fail before reaching ingress. Its retry
    // error belongs to publication, while confirmed reads remain available.
    let pending = sinex_protocol::Ingress::Events {
        source: None,
        coverage: None,
        frame: "recoverable-publication".into(),
        events: Vec::new(),
        failures: Vec::new(),
    };
    sinex.queue().unwrap().prepare(pending.clone()).unwrap();
    let frame_path = root
        .path()
        .join("producer-spool")
        .join(blake3::hash(b"polylogue").to_hex().to_string())
        .join("operations")
        .join(
            blake3::hash(pending.frame().as_bytes())
                .to_hex()
                .to_string(),
        )
        .join("frame.zst");
    let original_frame = std::fs::read(&frame_path).unwrap();
    let mut corrupt_frame = original_frame.clone();
    corrupt_frame[0] ^= 1;
    std::fs::write(&frame_path, corrupt_frame).unwrap();
    let _ = sinex.drain_pending();
    let publication_error = sinex.status().unwrap().error.unwrap();
    assert!(
        publication_error.starts_with("publication:"),
        "{publication_error}"
    );
    let control = primary::Sinex::confirmed(&sinex, &state, "cache-test").unwrap();
    assert_eq!(
        sinex.status().unwrap().error.as_deref(),
        Some(publication_error.as_str()),
        "a successful confirmed read cannot clear a failed publication"
    );
    assert!(
        matches!(control.as_slice(),[primary::Confirmed{change:ConfirmedChange::Control(control),..}]
        if matches!(control.as_ref(),ReplayControl::Reset{scope} if scope=="test-scope"))
    );
    let currency = sinex.currency(Some(42), Some("old-scope".into())).unwrap();
    assert_eq!(currency.scope(), Some("old-scope"));
    assert!(currency.degraded());
    let state = primary::ReplayState {
        applied_position: None,
        scope: Some("test-scope".into()),
        head: None,
        staged_delivery: None,
        completed_delivery: None,
        basis: None,
        projection: None,
    };
    let refused = primary::Sinex::confirmed(&sinex, &state, "cache-test").unwrap_err();
    assert!(
        refused.to_string().contains("synthetic refusal"),
        "{refused}"
    );
    let both = sinex.status().unwrap().error.unwrap();
    assert!(
        both.contains(&publication_error) && both.contains("synthetic refusal"),
        "{both}"
    );
    // Restore the original encoded frame and install its exact owner acknowledgment.
    // The corresponding retry succeeds without changing bytes or identity.
    std::fs::write(&frame_path, original_frame).unwrap();
    let mut acknowledgment = crate::tests::no_event_acknowledgment(&pending);
    acknowledgment.number = 4;
    sinex.queue().unwrap().acknowledge(acknowledgment).unwrap();
    let _ = sinex.drain_pending();
    let read_error = sinex.status().unwrap().error.unwrap();
    assert!(
        read_error.contains("synthetic refusal"),
        "a successful publication retry must preserve the independent read error: {read_error}"
    );
    assert!(!read_error.contains(&publication_error));
    sinex
        .read_client(async |client| {
            client
                .client
                .flush()
                .await
                .map_err(|error| Error::Unreachable(error.to_string()))
        })
        .unwrap();
    assert!(sinex.status().unwrap().error.is_none());
    drop(sinex);
    owner.join().unwrap();
}

#[test]
fn missing_and_wrong_passwords_are_refused_before_publication() {
    let server = server();
    let root = tempfile::tempdir().unwrap();
    assert!(matches!(
        Sinex::connect(
            &config(&server, &root.path().join("missing"), None),
            &Authentication::Anonymous,
            "fixture-host"
        ),
        Err(Error::Unreachable(_))
    ));
    let file = root.path().join("password");
    std::fs::write(&file, "incorrect").unwrap();
    assert!(matches!(
        Sinex::connect(
            &config(&server, &root.path().join("wrong"), Some(&file)),
            &Authentication::Password("incorrect".into()),
            "fixture-host"
        ),
        Err(Error::Unreachable(_))
    ));
}

#[test]
fn unavailable_authentication_is_named_and_offline_capture_remains_openable() {
    let server = server();
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("absent");
    let config = config(&server, root.path(), Some(&file));
    let authentication = Authentication::Unavailable {
        reason: "password file absent".into(),
    };
    assert!(Sinex::open(&config, &authentication, "fixture-host").is_ok());
    assert!(
        matches!(Sinex::connect(&config, &authentication, "fixture-host"),Err(Error::Invalid(message)) if message.contains("password file") && message.contains("absent"))
    );
}

#[test]
fn authentication_and_host_are_explicit_and_password_debug_is_redacted() {
    let root = tempfile::tempdir().unwrap();
    let config = Config {
        mode: Mode::Primary,
        nats: "nats://127.0.0.1:1".into(),
        password_file: Some(root.path().join("not-read")),
        subject_root: "fixture".into(),
        producer: "polylogue".into(),
        store: root.path().into(),
        upload_root: None,
    };
    assert!(Sinex::open(&config, &Authentication::Anonymous, "").is_err());
    let secret = Authentication::Password("synthetic-sensitive-value".into());
    assert!(!format!("{secret:?}").contains("synthetic-sensitive-value"));
    let sinex = Sinex::open(&config, &secret, "supplied-host").unwrap();
    assert_eq!(sinex.host, "supplied-host");
}

#[test]
fn an_unreadable_event_is_refused_at_its_position_but_a_lost_connection_stops_the_page() {
    use polylogue_store::primary::ConfirmedChange;
    let refused = crate::client::confirmed::readable(
        9,
        "event-x",
        "polylogue.future@7".into(),
        None,
        Err(Error::Invalid(
            "unsupported Polylogue kind polylogue.future@7".into(),
        )),
    )
    .unwrap();
    assert_eq!(refused.position, 9);
    assert_eq!(refused.event, "event-x");
    assert_eq!(
        refused.change,
        ConfirmedChange::Refused {
            kind: "polylogue.future@7".into(),
            reason: "unsupported Polylogue kind polylogue.future@7".into(),
            unreadable_input: true,
        }
    );
    let session = crate::kinds::kind_reference("polylogue.session").unwrap();
    let current_session = format!("{}@{}", session.name, session.version);
    let input = crate::kinds::kind_reference("polylogue.input_observation").unwrap();
    let current_input = format!("{}@{}", input.name, input.version);
    for (kind, unreadable_input) in [
        (current_input.as_str(), true),
        (current_session.as_str(), false),
        ("polylogue.session@6", true),
        ("polylogue.session@999", true),
        ("", true),
    ] {
        let definition = kind.rsplit_once('@').and_then(|(name, _)| {
            crate::kinds::kind_reference(name)
                .ok()
                .and_then(|kind| crate::kinds::declared_definition(&kind))
        });
        let refused = crate::client::confirmed::readable(
            10,
            "unreadable-fact",
            kind.into(),
            definition,
            Err(Error::Invalid("synthetic unreadable payload".into())),
        )
        .unwrap();
        assert!(
            matches!(refused.change,
            ConfirmedChange::Refused { unreadable_input: actual, .. }
            if actual == unreadable_input),
            "{kind}"
        );
    }
    let unknown_definition = crate::client::confirmed::readable(
        11,
        "unknown-definition",
        current_session,
        Some("unrecognized definition"),
        Err(Error::Invalid("synthetic unreadable payload".into())),
    )
    .unwrap();
    assert!(matches!(
        unknown_definition.change,
        ConfirmedChange::Refused {
            unreadable_input: true,
            ..
        }
    ));
    assert!(matches!(
        crate::client::confirmed::readable(
            9,
            "event-x",
            String::new(),
            None,
            Err(Error::Unreachable("connection lost".into()))
        ),
        Err(Error::Unreachable(_))
    ));
}

mod backup_restore;
mod material_bytes;

#[test]
fn reordered_duplicate_acknowledgments_keep_concurrent_frames_independent_after_refusal() {
    use futures_util::StreamExt;
    use std::sync::{Arc, Mutex};
    let server = server();
    let root = tempfile::tempdir().unwrap();
    let config = config(&server, root.path(), None);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let daemon = async_nats::ConnectOptions::with_user_and_password("daemon".into(), "synthetic-daemon-password".into()).connect(server.url()).await.unwrap();
        async_nats::jetstream::new(daemon.clone()).get_or_create_stream(async_nats::jetstream::stream::Config {
            name: "CONCURRENT_INGRESS".into(), subjects: vec!["auth.ingress.*".into()], ..Default::default()
        }).await.unwrap();
        let mut hello = daemon.subscribe("auth.connect.polylogue").await.unwrap();
        let mut ingress = daemon.subscribe("auth.ingress.polylogue").await.unwrap();
        let mut recorded = daemon.subscribe("auth.recorded.polylogue").await.unwrap();
        daemon.flush().await.unwrap();
        let owner = tokio::spawn(async move {
            let request = hello.next().await.unwrap();
            sinex_transport::replies::publish_json(&daemon, request.reply.unwrap().to_string(), ProducerWelcome::Welcome { family: Some("polylogue".into()), canonical_publisher: true }, Duration::from_secs(5)).await.unwrap();
            let mut acknowledgments = Vec::new();
            for count in [4, 1] {
                let mut batch = Vec::new();
                for _ in 0..count {
                    let message = ingress.next().await.unwrap();
                    let frame: sinex_protocol::Ingress = serde_json::from_slice(&message.payload).unwrap();
                    let mut acknowledgment = crate::tests::no_event_acknowledgment(&frame);
                    acknowledgment.number = frame.frame().strip_prefix("concurrent-").unwrap().parse::<i64>().unwrap() + 1;
                    if acknowledgment.number == 1 {
                        acknowledgment.result = sinex_protocol::FrameResult::Refused { kind: sinex_model::FrameRefusal::Content, reason: "synthetic refusal".into() };
                    }
                    batch.push(acknowledgment);
                }
                // All four frames must reach ingress before any acknowledgment. Plain
                // subscriptions each see these reordered, duplicated hints.
                for acknowledgment in batch.iter().rev() {
                    let hint = sinex_protocol::ReceiptHint { number: acknowledgment.number, frame_hash: blake3::hash(acknowledgment.frame.as_bytes()).to_hex().to_string() };
                    for _ in 0..2 {
                        daemon.publish("auth.ack.polylogue", serde_json::to_vec(&hint).unwrap().into()).await.unwrap();
                    }
                }
                for _ in 0..count {
                    let request = recorded.next().await.unwrap();
                    let query: sinex_protocol::RecordedQuery = serde_json::from_slice(&request.payload).unwrap();
                    let selected = batch.iter().filter(|acknowledgment| query.numbers.contains(&acknowledgment.number)).cloned().collect();
                    sinex_transport::replies::publish_json(&daemon, request.reply.unwrap().to_string(), sinex_protocol::RecordedPage { acknowledgments: selected, next: None, ..Default::default() }, Duration::from_secs(5)).await.unwrap();
                }
                acknowledgments.extend(batch);
            }
            acknowledgments
        });
        let spool = Arc::new(Mutex::new(crate::spool::Spool::open(root.path(), "polylogue").unwrap()));
        for number in 0..5 {
            spool.lock().unwrap().prepare(sinex_protocol::Ingress::Events { frame: format!("concurrent-{number}"), source: None, coverage: None, events: vec![], failures: vec![] }).unwrap();
        }
        let mut client = super::Client::connect_shared(&config, &Authentication::Password(PASSWORD.into()), spool.clone(), Default::default(), Default::default(), false).await.unwrap();
        let result = tokio::time::timeout(Duration::from_secs(10), client.retry_prepared()).await.expect("all independent frames complete without waiting for an earlier refusal");
        assert!(matches!(result, Err(Error::Refused { name, reason }) if name == "concurrent-0" && reason == "synthetic refusal"));
        let acknowledgments = owner.await.unwrap();
        assert_eq!(acknowledgments.len(), 5);
        for acknowledgment in acknowledgments {
            assert_eq!(spool.lock().unwrap().acknowledgment(&acknowledgment.frame).unwrap(), Some(acknowledgment));
        }
        client.retry_prepared().await.unwrap();
    });
}

#[test]
fn dropping_sender_waiting_for_acknowledgment_preserves_the_original_frame() {
    use futures_util::StreamExt;
    let server = server();
    let root = tempfile::tempdir().unwrap();
    let config = config(&server, root.path(), None);
    let url = server.url().to_string();
    let (ready, waiting) = std::sync::mpsc::channel();
    let (published, wait_published) = std::sync::mpsc::channel();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let owner = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let client = async_nats::ConnectOptions::with_user_and_password(
                "daemon".into(),
                "synthetic-daemon-password".into(),
            )
            .connect(url)
            .await
            .unwrap();
            async_nats::jetstream::new(client.clone())
                .get_or_create_stream(async_nats::jetstream::stream::Config {
                    name: "CANCEL_INGRESS".into(),
                    subjects: vec!["auth.ingress.*".into()],
                    ..Default::default()
                })
                .await
                .unwrap();
            let mut hello = client.subscribe("auth.connect.polylogue").await.unwrap();
            let mut ingress = client.subscribe("auth.ingress.polylogue").await.unwrap();
            let mut recorded = client.subscribe("auth.recorded.polylogue").await.unwrap();
            client.flush().await.unwrap();
            ready.send(()).unwrap();
            let request = hello.next().await.unwrap();
            sinex_transport::replies::publish_json(
                &client,
                request.reply.unwrap().to_string(),
                ProducerWelcome::Welcome {
                    family: Some("polylogue".into()),
                    canonical_publisher: true,
                },
                Duration::from_secs(5),
            )
            .await
            .unwrap();
            let message = ingress.next().await.unwrap();
            let frame: sinex_protocol::Ingress = serde_json::from_slice(&message.payload).unwrap();
            let sinex_protocol::Ingress::RegisterKinds { declarations, .. } = &frame else {
                panic!("register before sending")
            };
            let mut acknowledgment = crate::tests::no_event_acknowledgment(&frame);
            acknowledgment.result = sinex_protocol::FrameResult::KindsRegistered {
                kinds: declarations
                    .iter()
                    .map(|kind| (kind.kind.clone(), kind.definition_hash()))
                    .collect(),
            };
            client
                .publish(
                    "auth.ack.polylogue",
                    serde_json::to_vec(&sinex_protocol::ReceiptHint {
                        number: acknowledgment.number,
                        frame_hash: blake3::hash(frame.frame().as_bytes()).to_hex().to_string(),
                    })
                    .unwrap()
                    .into(),
                )
                .await
                .unwrap();
            let request = recorded.next().await.unwrap();
            sinex_transport::replies::publish_json(
                &client,
                request.reply.unwrap().to_string(),
                sinex_protocol::RecordedPage {
                    acknowledgments: vec![acknowledgment],
                    next: None,
                    ..Default::default()
                },
                Duration::from_secs(5),
            )
            .await
            .unwrap();
            let message = ingress.next().await.unwrap();
            let frame: sinex_protocol::Ingress = serde_json::from_slice(&message.payload).unwrap();
            assert_eq!(frame.frame(), "cancelled-frame");
            published.send(()).unwrap();
            stopped.await.unwrap();
        });
    });
    waiting.recv_timeout(Duration::from_secs(5)).unwrap();
    let sinex = Sinex::open(
        &config,
        &Authentication::Password(PASSWORD.into()),
        "fixture-host",
    )
    .unwrap();
    let frame = sinex_protocol::Ingress::Events {
        frame: "cancelled-frame".into(),
        source: None,
        coverage: None,
        events: vec![],
        failures: vec![],
    };
    let prepared = sinex.queue().unwrap().prepare(frame.clone()).unwrap();
    wait_published
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    drop(sinex);
    let restored = crate::spool::Spool::open(root.path(), "polylogue").unwrap();
    assert_eq!(
        restored.get("cancelled-frame").unwrap().unwrap().frame,
        prepared.frame
    );
    assert_eq!(restored.decode_frame("cancelled-frame").unwrap(), frame);
    assert!(
        restored
            .acknowledgment("cancelled-frame")
            .unwrap()
            .is_none()
    );
    stop.send(()).unwrap();
    owner.join().unwrap();
}

#[test]
fn frame_storage_failure_keeps_its_retryable_type() {
    assert!(matches!(
        super::frame_error(sinex_transport::frame_bytes::Error::Storage(
            std::io::Error::from_raw_os_error(5)
        )),
        Error::Io(_)
    ));
    assert!(matches!(
        super::frame_error(sinex_transport::frame_bytes::Error::Mismatch(
            "synthetic corrupt encoding".into()
        )),
        Error::Invalid(_)
    ));
}

struct ProducerOwner {
    client: async_nats::Client,
    hello: async_nats::Subscriber,
    ingress: async_nats::Subscriber,
    recorded: async_nats::Subscriber,
}
impl ProducerOwner {
    async fn start(server: &devkit::nats::Server, name: &str) -> Self {
        let client = async_nats::ConnectOptions::with_user_and_password(
            "daemon".into(),
            "synthetic-daemon-password".into(),
        )
        .connect(server.url())
        .await
        .unwrap();
        async_nats::jetstream::new(client.clone())
            .get_or_create_stream(async_nats::jetstream::stream::Config {
                name: name.into(),
                subjects: vec!["auth.ingress.*".into()],
                ..Default::default()
            })
            .await
            .unwrap();
        let hello = client.subscribe("auth.connect.polylogue").await.unwrap();
        let ingress = client.subscribe("auth.ingress.polylogue").await.unwrap();
        let recorded = client.subscribe("auth.recorded.polylogue").await.unwrap();
        client.flush().await.unwrap();
        Self {
            client,
            hello,
            ingress,
            recorded,
        }
    }
    async fn welcome(&mut self) {
        use futures_util::StreamExt;
        let request = self.hello.next().await.unwrap();
        sinex_transport::replies::publish_json(
            &self.client,
            request.reply.unwrap().to_string(),
            ProducerWelcome::Welcome {
                family: Some("polylogue".into()),
                canonical_publisher: true,
            },
            Duration::from_secs(5),
        )
        .await
        .unwrap();
    }
    async fn acknowledge(&mut self, acknowledgment: sinex_protocol::Acknowledgment) {
        use futures_util::StreamExt;
        self.client
            .publish(
                "auth.ack.polylogue",
                serde_json::to_vec(&sinex_protocol::ReceiptHint {
                    number: acknowledgment.number,
                    frame_hash: blake3::hash(acknowledgment.frame.as_bytes())
                        .to_hex()
                        .to_string(),
                })
                .unwrap()
                .into(),
            )
            .await
            .unwrap();
        let request = self.recorded.next().await.unwrap();
        let query: sinex_protocol::RecordedQuery =
            serde_json::from_slice(&request.payload).unwrap();
        assert_eq!(query.numbers, [acknowledgment.number]);
        sinex_transport::replies::publish_json(
            &self.client,
            request.reply.unwrap().to_string(),
            sinex_protocol::RecordedPage {
                acknowledgments: vec![acknowledgment],
                next: None,
                ..Default::default()
            },
            Duration::from_secs(5),
        )
        .await
        .unwrap();
    }
    async fn register(&mut self) {
        use futures_util::StreamExt;
        let message = self.ingress.next().await.unwrap();
        let frame: sinex_protocol::Ingress = serde_json::from_slice(&message.payload).unwrap();
        let sinex_protocol::Ingress::RegisterKinds { declarations, .. } = &frame else {
            panic!("register before sending")
        };
        let mut acknowledgment = crate::tests::no_event_acknowledgment(&frame);
        acknowledgment.result = sinex_protocol::FrameResult::KindsRegistered {
            kinds: declarations
                .iter()
                .map(|kind| (kind.kind.clone(), kind.definition_hash()))
                .collect(),
        };
        self.acknowledge(acknowledgment).await;
    }
}

fn eventually(mut predicate: impl FnMut() -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !predicate() {
        assert!(
            std::time::Instant::now() < deadline,
            "producer did not report the expected durable acknowledgment"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn current_frame_reaches_durable_acknowledgment_with_5000_original_frames_waiting() {
    use futures_util::StreamExt;
    let server = server();
    let root = tempfile::tempdir().unwrap();
    let config = config(&server, root.path(), None);
    let mut sinex = synchronous(&config, &Authentication::Password(PASSWORD.into()));
    for number in 0..5000 {
        sinex
            .queue()
            .unwrap()
            .prepare(sinex_protocol::Ingress::Events {
                frame: format!("original-{number}"),
                source: None,
                coverage: None,
                events: vec![],
                failures: vec![],
            })
            .unwrap();
    }
    let current = sinex_protocol::Ingress::Events {
        frame: "current-frame".into(),
        source: None,
        coverage: None,
        events: vec![],
        failures: vec![],
    };
    sinex.queue().unwrap().prepare(current.clone()).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut owner = runtime.block_on(ProducerOwner::start(&server, "BACKLOG_INGRESS"));
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let serving = std::thread::spawn(move || {
        runtime.block_on(async move {
            owner.welcome().await;
            owner.register().await;
            let mut current = None;
            let mut oldest = None;
            for _ in 0..4 {
                let message = owner.ingress.next().await.unwrap();
                let frame: sinex_protocol::Ingress =
                    serde_json::from_slice(&message.payload).unwrap();
                match frame.frame() {
                    "current-frame" => current = Some(frame),
                    "original-0" => oldest = Some(frame),
                    _ => {} // These backlog frames deliberately receive no acknowledgment.
                }
            }
            let mut refusal = crate::tests::no_event_acknowledgment(
                &oldest.expect("oldest backlog gets a transmission slot"),
            );
            refusal.number = 2;
            refusal.result = sinex_protocol::FrameResult::Refused {
                kind: sinex_model::FrameRefusal::Content,
                reason: "synthetic old-frame refusal".into(),
            };
            owner.acknowledge(refusal).await;
            let mut acknowledgment = crate::tests::no_event_acknowledgment(
                &current.expect("current independent work gets a transmission slot"),
            );
            acknowledgment.number = 3;
            owner.acknowledge(acknowledgment.clone()).await;
            stopped.await.unwrap();
            acknowledgment
        })
    });
    sinex.start_sender().unwrap();
    eventually(|| {
        sinex
            .queue()
            .unwrap()
            .acknowledgment("current-frame")
            .unwrap()
            .is_some()
    });
    let current_ack = sinex
        .queue()
        .unwrap()
        .acknowledgment("current-frame")
        .unwrap()
        .unwrap();
    assert_eq!(
        current_ack.content_hash.as_deref(),
        Some(
            sinex_transport::producer::content_hash(&current)
                .unwrap()
                .as_str()
        )
    );
    assert!(
        sinex
            .queue()
            .unwrap()
            .get("original-2500")
            .unwrap()
            .unwrap()
            .phase
            .is_pending()
    );
    assert!(
        sinex
            .queue()
            .unwrap()
            .get("original-0")
            .unwrap()
            .unwrap()
            .phase
            .is_refused()
    );
    drop(sinex);
    stop.send(()).unwrap();
    assert_eq!(serving.join().unwrap(), current_ack);
}

#[test]
fn refused_attachment_retains_its_request_while_independent_completions_progress() {
    use polylogue_model::{
        BlobHash,
        authored::{AuthoredEffect, ChatChange, ExecutionEligibility, PreparedRequest},
        user::{ChatAction, ChatAttachment, UserOutcome, UserWrite},
    };
    use polylogue_store::primary::ProductWrite;
    let server = server();
    let root = tempfile::tempdir().unwrap();
    let config = config(&server, root.path(), None);
    let sinex = synchronous(&config, &Authentication::Password(PASSWORD.into()));
    let mut owner = sinex
        .runtime
        .block_on(ProducerOwner::start(&server, "REFUSED_DEPENDENCY"));
    let welcome = sinex.runtime.spawn(async move {
        owner.welcome().await;
    });
    let client = sinex
        .runtime
        .block_on(super::Client::connect_shared(
            &config,
            &sinex.authentication,
            sinex.spool.clone(),
            sinex.health.clone(),
            sinex.publication.clone(),
            false,
        ))
        .unwrap();
    sinex.runtime.block_on(welcome).unwrap();
    *sinex.client.lock().unwrap() = Some(client);
    let bytes = b"synthetic refused attachment";
    let hash = BlobHash::of(bytes);
    let blob = sinex
        .prepare_blob(
            hash,
            bytes.len() as u64,
            crate::tests::body(root.path(), bytes),
        )
        .unwrap();
    let frame = sinex.queue().unwrap().decode_frame(&blob.identity).unwrap();
    let mut refusal = crate::tests::no_event_acknowledgment(&frame);
    refusal.result = sinex_protocol::FrameResult::Refused {
        kind: sinex_model::FrameRefusal::Content,
        reason: "synthetic attachment refusal".into(),
    };
    sinex.queue().unwrap().acknowledge(refusal).unwrap();
    for (number, id, blocked) in [(2, "blocked", true), (3, "independent", false)] {
        let mut effect = AuthoredEffect::default();
        if blocked {
            effect.chats.push(ChatChange {
                action: ChatAction {
                    id: "synthetic-chat".into(),
                    request_key: "synthetic-key".into(),
                    request_hash: hash,
                    request: serde_json::json!({}),
                    stage_only: true,
                    needs_approval: false,
                    created_at: 0,
                    execution: ExecutionEligibility::Live,
                    attachments: vec![ChatAttachment {
                        id: "synthetic-attachment".into(),
                        blob: hash,
                        name: "synthetic.txt".into(),
                        media_type: "text/plain".into(),
                        size: bytes.len() as u64,
                    }],
                    events: vec![],
                },
                added_events: vec![],
            });
        }
        let request = PreparedRequest {
            request: polylogue_store::fixture::authored::operator_request(
                id.into(),
                UserWrite::Assert { assertions: vec![] },
            ),
            outcome: UserOutcome {
                id: id.into(),
                repeat: false,
                pending: false,
            },
            effect,
        };
        sinex.prepare_request(&request).unwrap();
        let identity = format!("request:{id}");
        let physical = crate::tests::extent(
            &root.path().join(id),
            &serde_json::to_vec(&request).unwrap(),
        );
        sinex.queue().unwrap().acknowledge(serde_json::from_value(serde_json::json!({ "number":number, "frame":identity, "material":crate::material_id("polylogue", &identity), "material_extents":[physical.encode()], "admitted":[], "content_hash":null, "result":"admitted", "counts":crate::tests::zero_counts(), "commit_seq":number })).unwrap()).unwrap();
    }
    let (_, product) = crate::tests::named::phases().remove(0);
    sinex
        .prepare_named_products(
            "independent-observation",
            &[ProductWrite {
                product,
                location: None,
                parents: vec![],
                machine: None,
                output_index: 0,
                publish: true,
            }],
        )
        .unwrap();
    let observation = "independent-observation:observation";
    let mut body = Vec::new();
    std::io::Read::read_to_end(
        &mut sinex.queue().unwrap().body(observation).unwrap(),
        &mut body,
    )
    .unwrap();
    let physical = crate::tests::extent(&root.path().join("observation"), &body);
    sinex.queue().unwrap().acknowledge(serde_json::from_value(serde_json::json!({ "number":4, "frame":observation, "material":crate::material_id("polylogue", observation), "material_extents":[physical.encode()], "admitted":[], "content_hash":null, "result":"admitted", "counts":crate::tests::zero_counts(), "commit_seq":4 })).unwrap()).unwrap();
    let result = sinex.drain_pending();
    assert!(
        matches!(&result, Err(Error::Refused { reason, .. }) if reason == "synthetic attachment refusal"),
        "{result:?}"
    );
    let queue = sinex.queue().unwrap();
    assert!(queue.capture_group("request:blocked").unwrap().is_none());
    assert!(
        queue
            .capture_group("request:independent")
            .unwrap()
            .is_some()
    );
    assert!(queue.get("independent-observation").unwrap().is_some());
    drop(queue);
    assert!(
        sinex
            .status()
            .unwrap()
            .error
            .unwrap()
            .contains("synthetic attachment refusal")
    );
}
