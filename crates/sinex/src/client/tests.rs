use crate::{Config, Error, Mode, Sinex};
use sinex_protocol::Subjects;
use std::path::Path;

const PASSWORD: &str = "synthetic-password";

/// A NATS server that admits the daemon and one producer, `polylogue`, as
/// Sinex declares its producers.
fn server() -> devkit::nats::Server {
    let config = Subjects::new("auth").authorization(
        ("daemon", "synthetic-daemon-password"),
        &[("polylogue", PASSWORD)],
    );
    devkit::nats::Server::start_with(&devkit::nats::Options { config }).unwrap()
}

fn config(server: &devkit::nats::Server, password_file: Option<&Path>) -> Config {
    Config {
        mode: Mode::Primary,
        nats: server.url().into(),
        password_file: password_file.map(Path::to_path_buf),
        subject_root: "auth".into(),
        producer: "polylogue".into(),
        store: "/unused".into(),
    }
}

fn connect(config: &Config) -> crate::Result<()> {
    let dir = tempfile::tempdir().unwrap();
    Sinex::connect(config, dir.path()).map(drop)
}

#[test]
fn the_producer_connects_with_the_password_from_its_file() {
    let server = server();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("password");
    // Secret files usually end in a newline; it is not part of the password.
    std::fs::write(&file, format!("{PASSWORD}\n")).unwrap();
    let url = server.url().to_string();
    let (ready, started) = std::sync::mpsc::channel();
    let responder = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            use futures_util::StreamExt;
            let client = async_nats::ConnectOptions::with_user_and_password(
                "daemon".into(),
                "synthetic-daemon-password".into(),
            )
            .connect(&url)
            .await
            .unwrap();
            async_nats::jetstream::new(client.clone())
                .get_or_create_stream(async_nats::jetstream::stream::Config {
                    name: "AUTH_INGRESS".into(),
                    subjects: vec!["auth.ingress.*".into()],
                    ..Default::default()
                })
                .await
                .unwrap();
            let mut incoming = client.subscribe("auth.ingress.polylogue").await.unwrap();
            client.flush().await.unwrap();
            ready.send(()).unwrap();
            let message = incoming.next().await.unwrap();
            let sinex_protocol::Ingress::RegisterKinds {
                frame,
                declarations,
            } = serde_json::from_slice(&message.payload).unwrap()
            else {
                panic!("registration first")
            };
            let registered: Vec<_> = declarations
                .iter()
                .map(|d| (d.kind.clone(), d.definition_hash()))
                .collect();
            let ack = sinex_protocol::Acknowledgment {
                frame,
                material: None,
                admitted: vec![],
                outcome: serde_json::json!({"registered_kinds":registered}),
                refusal: None,
                commit_seq: None,
            };
            client
                .publish(
                    "auth.ack.polylogue",
                    serde_json::to_vec(&ack).unwrap().into(),
                )
                .await
                .unwrap();
            client.flush().await.unwrap();
        });
    });
    started.recv().unwrap();
    let mut config = config(&server, Some(&file));
    config.store = dir.path().join("store");
    std::fs::create_dir(&config.store).unwrap();
    let sinex = Sinex::connect(&config, dir.path()).unwrap();
    let replay = sinex.replay().unwrap();
    assert!(matches!(
        sinex.client.try_lock(),
        Err(std::sync::TryLockError::WouldBlock)
    ));
    drop(replay);
    assert!(sinex.client.try_lock().is_ok());
    drop(sinex);
    responder.join().unwrap();
}

#[test]
fn a_missing_or_wrong_password_is_refused_by_nats() {
    let server = server();
    let error = connect(&config(&server, None)).unwrap_err();
    assert!(
        matches!(&error, Error::Unreachable(m) if m.contains("authorization")),
        "{error}"
    );
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("password");
    std::fs::write(&file, "not-the-password").unwrap();
    let error = connect(&config(&server, Some(&file))).unwrap_err();
    assert!(
        matches!(&error, Error::Unreachable(m) if m.contains("authorization")),
        "{error}"
    );
}

#[test]
fn an_unreadable_password_file_is_named() {
    let server = server();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("absent");
    let error = connect(&config(&server, Some(&file))).unwrap_err();
    assert!(
        matches!(&error, Error::Invalid(m) if m.contains("password file") && m.contains("absent")),
        "{error}"
    );
}
