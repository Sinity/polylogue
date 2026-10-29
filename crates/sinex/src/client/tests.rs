use crate::{Config, Error, Mode, Sinex};
use sinex_protocol::{ProducerWelcome, Subjects};
use std::{path::Path, time::Duration};
const PASSWORD: &str = "synthetic-password";

fn server() -> devkit::nats::Server {
    let config = Subjects::new("auth").authorization(
        ("daemon", "synthetic-daemon-password"),
        &[("polylogue", PASSWORD)],
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
        });
    });
    wait.recv().unwrap();
    let sinex = Sinex::connect(&config(&server, root.path(), Some(&file))).unwrap();
    assert_eq!(sinex.status().unwrap().family.as_deref(), Some("polylogue"));
    assert!(sinex.status().unwrap().canonical_publisher);
    drop(sinex);
    owner.join().unwrap();
}

#[test]
fn missing_and_wrong_passwords_are_refused_before_publication() {
    let server = server();
    let root = tempfile::tempdir().unwrap();
    assert!(matches!(
        Sinex::connect(&config(&server, &root.path().join("missing"), None)),
        Err(Error::Unreachable(_))
    ));
    let file = root.path().join("password");
    std::fs::write(&file, "incorrect").unwrap();
    assert!(matches!(
        Sinex::connect(&config(&server, &root.path().join("wrong"), Some(&file))),
        Err(Error::Unreachable(_))
    ));
}

#[test]
fn unreadable_secret_is_named_and_offline_capture_remains_openable() {
    let server = server();
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("absent");
    let config = config(&server, root.path(), Some(&file));
    assert!(Sinex::open(&config).is_ok());
    assert!(
        matches!(Sinex::connect(&config),Err(Error::Invalid(message)) if message.contains("password file") && message.contains("absent"))
    );
}
