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
            client.flush().await.unwrap();ready.send(()).unwrap();
            let request=hello.next().await.unwrap();let inbox=request.reply.unwrap();
            assert!(inbox.starts_with("_INBOX.polylogue."));
            sinex_transport::replies::publish_json(&client,inbox.to_string(),ProducerWelcome{refusal:None,family:Some("polylogue".into()),canonical_publisher:true},Duration::from_secs(5)).await.unwrap();
            let frame=ingress.next().await.unwrap();
            let sinex_protocol::Ingress::RegisterKinds{frame,declarations}=serde_json::from_slice(&frame.payload).unwrap() else {panic!("kind registration expected")};
            let hashes:Vec<_>=declarations.iter().map(|declaration|(declaration.kind.clone(),declaration.definition_hash())).collect();
            let receipt:sinex_protocol::Acknowledgment=serde_json::from_value(serde_json::json!({"number":1,"frame":frame,"material":null,"admitted":[],"outcome":{"registered_kinds":hashes},"refusal":null,"commit_seq":null})).unwrap();
            client.publish("auth.ack.polylogue",serde_json::to_vec(&sinex_protocol::ReceiptHint{number:1,frame_hash:blake3::hash(frame.as_bytes()).to_hex().to_string()}).unwrap().into()).await.unwrap();
            let request=recorded.next().await.unwrap();
            let query:sinex_protocol::RecordedQuery=serde_json::from_slice(&request.payload).unwrap();assert_eq!(query.numbers,[1]);
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),sinex_protocol::RecordedPage{acknowledgments:vec![receipt],next:None},Duration::from_secs(5)).await.unwrap();
            let subjects=Subjects::new("auth");
            let response=sinex_protocol::ConfirmedConsumerReply{stream:subjects.confirmed_stream(),consumer:subjects.consumer_name("test-scope"),scope:"test-scope".into(),position:Some(42),refusal:None,gap:None};
            let request=consumers.next().await.unwrap();
            let query:sinex_protocol::ConfirmedConsumer=serde_json::from_slice(&request.payload).unwrap();
            assert_eq!(query.after,42);
            assert!(!query.status_only);
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),response.clone(),Duration::from_secs(5)).await.unwrap();
            let request=consumers.next().await.unwrap();
            let query:sinex_protocol::ConfirmedConsumer=serde_json::from_slice(&request.payload).unwrap();
            assert_eq!(query.after,0,"a new scope must replay from the cleared checkpoint");
            let js=async_nats::jetstream::new(client.clone());
            let stream=js.get_or_create_stream(async_nats::jetstream::stream::Config{name:subjects.confirmed_stream(),subjects:vec![subjects.confirmed_all()],..Default::default()}).await.unwrap();
            stream.get_or_create_consumer(&response.consumer,async_nats::jetstream::consumer::pull::Config{durable_name:Some(response.consumer.clone()),..Default::default()}).await.unwrap();
            js.publish(subjects.confirmed_control(),serde_json::to_vec(&sinex_protocol::Confirmed::PositionComplete{commit_seq:0}).unwrap().into()).await.unwrap().await.unwrap();
            sinex_transport::replies::publish_json(&client,request.reply.unwrap().to_string(),response,Duration::from_secs(5)).await.unwrap();
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
    assert!(sinex.status().unwrap().canonical_publisher);
    let control =
        polylogue_store::primary::Sinex::confirmed(&sinex, 42, "cache-test", Some("old-scope"))
            .unwrap();
    assert!(
        matches!(control.as_slice(),[polylogue_store::primary::Confirmed{position:0,change:polylogue_store::primary::ConfirmedChange::ScopeReset{scope},..}] if scope=="test-scope")
    );
    let currency = sinex.currency(Some(42), Some("old-scope".into())).unwrap();
    assert_eq!(currency.scope(), Some("old-scope"));
    assert!(currency.degraded());
    polylogue_store::primary::Sinex::acknowledge(&sinex, 0).unwrap();
    let page =
        polylogue_store::primary::Sinex::confirmed(&sinex, 0, "cache-test", Some("test-scope"))
            .unwrap();
    assert!(matches!(
        page.as_slice(),
        [polylogue_store::primary::Confirmed {
            position: 0,
            change: polylogue_store::primary::ConfirmedChange::PositionComplete,
            ..
        }]
    ));
    polylogue_store::primary::Sinex::acknowledge(&sinex, 0).unwrap();
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
