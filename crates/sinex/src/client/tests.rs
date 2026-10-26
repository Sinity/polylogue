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
    connect(&config(&server, Some(&file))).unwrap();
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
