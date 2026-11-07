use crate::{Authentication, Config, Error, Mode, Sinex};
use sinex_protocol::{ProducerWelcome, Subjects};
use std::{path::Path, time::Duration};
const PASSWORD: &str = "synthetic-password";

fn server() -> devkit::nats::Server {
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
    }
}

#[test]
fn authenticated_private_reply_and_complete_registration_receipt_establish_family() {
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
            sinex_transport::replies::publish_json(&client,inbox.to_string(),ProducerWelcome{refusal:None,family:Some("polylogue".into()),canonical_publisher:true},Duration::from_secs(5)).await.unwrap();
            let frame=ingress.next().await.unwrap();
            let sinex_protocol::Ingress::RegisterKinds{frame,declarations}=serde_json::from_slice(&frame.payload).unwrap() else {panic!("kind registration expected")};
            let hashes:Vec<_>=declarations.iter().map(|declaration|(declaration.kind.clone(),declaration.definition_hash())).collect();
            let receipt:sinex_protocol::Acknowledgment=serde_json::from_value(serde_json::json!({"number":1,"frame":frame,"material":null,"admitted":[],"outcome":{"registered_kinds":hashes,"frame_content_hash":sinex_transport::producer::content_hash(&sinex_protocol::Ingress::RegisterKinds{frame:frame.clone(),declarations:declarations.clone()}).unwrap()},"refusal":null,"commit_seq":null})).unwrap();
            client.publish("auth.ack.polylogue",serde_json::to_vec(&sinex_protocol::ReceiptHint{number:1,frame_hash:blake3::hash(frame.as_bytes()).to_hex().to_string()}).unwrap().into()).await.unwrap();
            let request=recorded.next().await.unwrap();
            let query:sinex_protocol::RecordedQuery=serde_json::from_slice(&request.payload).unwrap();assert_eq!(query.numbers,[1]);
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),sinex_protocol::RecordedPage{acknowledgments:vec![receipt],next:None,..Default::default()},Duration::from_secs(5)).await.unwrap();
            let request=contexts.next().await.unwrap();
            let query:sinex_protocol::PublicContextRequest=serde_json::from_slice(&request.payload).unwrap();assert!(query.coverage.is_none());
            assert!(request.reply.as_ref().unwrap().starts_with("_INBOX.polylogue."));
            let context:sinex_protocol::PublicContext=serde_json::from_value(serde_json::json!({"state":"available","context_token":"synthetic-context-token","settings_identity":"owner-settings","owner_revision":7,"day":{"zone":"Europe/Warsaw","start":"04:00:00"},"quiet_hours":{"from":1320,"to":420},"interrupt_budget":{"per_hour":3,"used":1,"remaining":2,"window_ms":3600000},"observed_at":"2026-01-01T00:00:00Z","coverage":{"state":"not_requested"}})).unwrap();
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),context,Duration::from_secs(5)).await.unwrap();
            let subjects=Subjects::new("auth");
            let request=consumers.next().await.unwrap();
            let query:sinex_protocol::ConfirmedConsumer=serde_json::from_slice(&request.payload).unwrap();
            assert_eq!(query,sinex_protocol::ConfirmedConsumer{basis:None,after:42,cache:"cache-test".into(),status_only:false});
            let reply=sinex_protocol::ConfirmedConsumerReply{position:Some(50),stream:subjects.confirmed_stream(),stream_instance:Some("instance".into()),consumer:"consumer".into(),scope:"test-scope".into(),refusal:None,gap:None};
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),reply,Duration::from_secs(5)).await.unwrap();
            let request=consumers.next().await.unwrap();
            let query:sinex_protocol::ConfirmedConsumer=serde_json::from_slice(&request.payload).unwrap();
            assert_eq!(query.after,0,"an index that applied nothing reads the stream from its start");
            let reply=sinex_protocol::ConfirmedConsumerReply{refusal:Some("synthetic refusal".into()),..Default::default()};
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),reply,Duration::from_secs(5)).await.unwrap();
        });
    });
    wait.recv().unwrap();
    let sinex = Sinex::connect(
        &config(&server, root.path(), Some(&file)),
        &Authentication::Password(PASSWORD.into()),
        "fixture-host",
    )
    .unwrap();
    assert_eq!(sinex.status().unwrap().family.as_deref(), Some("polylogue"));
    sinex
        .with(async |client| {
            let frame = sinex_protocol::Ingress::RegisterKinds {
                frame: "refused-registration".into(),
                declarations: Vec::new(),
            };
            let hash = sinex_transport::producer::content_hash(&frame).unwrap();
            let receipt: sinex_protocol::Acknowledgment = serde_json::from_value(
                serde_json::json!({"number":2,"frame":frame.frame(),"material":null,
                    "admitted":[],"outcome":{"frame_content_hash":hash},
                    "refusal":"exact owner declaration refusal","commit_seq":null}),
            )
            .unwrap();
            {
                let spool = client.spool.lock().unwrap();
                let prepared = spool.prepare(frame.clone()).unwrap();
                let mut mismatch = receipt.clone();
                mismatch.outcome["frame_content_hash"] = "different payload".into();
                assert!(matches!(
                    super::verified_receipt(&prepared, mismatch),
                    Err(Error::ReceiptConflict { .. })
                ));
                spool.acknowledge(receipt).unwrap();
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
            let receipt: sinex_protocol::Acknowledgment = serde_json::from_value(
                serde_json::json!({"number":3,"frame":conflict.frame(),"material":null,
                    "admitted":[],"outcome":{"frame_content_hash":"different payload"},
                    "refusal":null,"commit_seq":null}),
            )
            .unwrap();
            {
                let spool = client.spool.lock().unwrap();
                spool.prepare(conflict.clone()).unwrap();
                spool
                    .record_conflict(conflict.frame(), &offered_hash, &receipt)
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
    use polylogue_store::primary::{self, ConfirmedChange, ReplayControl};
    let state = primary::ReplayState {
        applied_position: Some(42),
        scope: Some("old-scope".into()),
        head: None,
        staged_delivery: None,
    };
    let control = primary::Sinex::confirmed(&sinex, &state, "cache-test").unwrap();
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
    };
    let refused = primary::Sinex::confirmed(&sinex, &state, "cache-test").unwrap_err();
    assert!(
        refused.to_string().contains("synthetic refusal"),
        "{refused}"
    );
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
    for (kind, unreadable_input) in [
        ("polylogue.input_observation@4", true),
        ("polylogue.session@5", false),
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
        "polylogue.session@5".into(),
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
