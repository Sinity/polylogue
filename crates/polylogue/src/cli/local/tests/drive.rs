use crate::{auth, config::Startup};
use polylogue_sources::ai_studio::drive::{AuthorizedUser, token_path};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::Path,
    process::ExitCode,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

#[derive(Clone, Copy)]
enum Consent {
    Allow,
    Deny,
    WrongState,
    Wait,
}
#[derive(Clone, Debug)]
struct Request {
    path: String,
    query: BTreeMap<String, String>,
    form: BTreeMap<String, String>,
}
struct Provider {
    base: String,
    requests: Arc<Mutex<Vec<Request>>>,
    worker: Option<JoinHandle<()>>,
    publish: Arc<Mutex<Option<std::path::PathBuf>>>,
}
fn pairs(text: &str) -> BTreeMap<String, String> {
    reqwest::Url::parse(&format!("http://localhost/?{text}"))
        .unwrap()
        .query_pairs()
        .into_owned()
        .collect()
}
impl Provider {
    fn new(consent: Consent, token_status: u16, token: Value, revoke_status: u16) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(vec![]));
        let seen = requests.clone();
        let publish = Arc::new(Mutex::new(None::<std::path::PathBuf>));
        let publication = publish.clone();
        let worker = std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let target = first.split_whitespace().nth(1).unwrap();
                let (path, query) = target.split_once('?').unwrap_or((target, ""));
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse().unwrap();
                    }
                }
                let mut bytes = vec![0; length];
                reader.read_exact(&mut bytes).unwrap();
                let request = Request {
                    path: path.into(),
                    query: pairs(query),
                    form: pairs(&String::from_utf8(bytes).unwrap()),
                };
                let stop = path == "/stop";
                if !stop {
                    seen.lock().unwrap().push(request.clone());
                }
                if path == "/token"
                    && request.form.get("grant_type").map(String::as_str)
                        == Some("authorization_code")
                    && let Some(path) = publication.lock().unwrap().take()
                {
                    std::fs::write(path, br#"{"type":"authorized_user","client_id":"winner","refresh_token":"winner-token"}"#).unwrap();
                }
                let mut location = None;
                let (status, body) = match path {
                    "/authorize" => match consent {
                        Consent::Wait => (200, "{}".into()),
                        _ => {
                            let mut redirect =
                                reqwest::Url::parse(&request.query["redirect_uri"]).unwrap();
                            let mut query = redirect.query_pairs_mut();
                            query.append_pair(
                                "state",
                                if matches!(consent, Consent::WrongState) {
                                    "wrong"
                                } else {
                                    &request.query["state"]
                                },
                            );
                            if matches!(consent, Consent::Deny) {
                                query.append_pair("error", "access_denied");
                            } else {
                                query.append_pair("code", &request.query["code_challenge"]);
                            }
                            drop(query);
                            location = Some(redirect.to_string());
                            (302, "{}".into())
                        }
                    },
                    "/token"
                        if request.form.get("grant_type").map(String::as_str)
                            == Some("refresh_token") =>
                    {
                        (200, json!({"access_token":"short-lived"}).to_string())
                    }
                    "/token" => (token_status, token.to_string()),
                    "/revoke" => (
                        revoke_status,
                        if revoke_status == 400 {
                            json!({"error":"invalid_token"}).to_string()
                        } else {
                            "{}".into()
                        },
                    ),
                    "/drive/files"
                        if request
                            .query
                            .get("q")
                            .is_some_and(|query| query.contains("apps.folder")) =>
                    {
                        (200, json!({"files":[{"id":"folder"}]}).to_string())
                    }
                    "/drive/files" => (200, json!({"files":[]}).to_string()),
                    "/stop" => (200, "{}".into()),
                    _ => (404, "{}".into()),
                };
                write!(stream, "HTTP/1.1 {status} Reply\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n", body.len()).unwrap();
                if let Some(location) = location {
                    write!(stream, "Location: {location}\r\n").unwrap();
                }
                write!(stream, "\r\n{body}").unwrap();
                if stop {
                    break;
                }
            }
        });
        Self {
            base,
            requests,
            worker: Some(worker),
            publish,
        }
    }
    fn good() -> Self {
        Self::new(
            Consent::Allow,
            200,
            json!({"refresh_token":"retained-refresh","scope":auth::SCOPE}),
            200,
        )
    }
    fn driver(&self) -> auth::Drive {
        auth::Drive {
            authorization: format!("{}/authorize", self.base),
            token: format!("{}/token", self.base),
            revoke: format!("{}/revoke", self.base),
            browser,
        }
    }
    fn stop(&mut self) {
        let _ = ureq::get(&format!("{}/stop", self.base)).call();
    }
    fn finish(mut self) -> Vec<Request> {
        self.stop();
        self.worker.take().unwrap().join().unwrap();
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Provider {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            self.stop();
            let _ = worker.join();
        }
    }
}
fn browser(url: &str) -> std::io::Result<()> {
    let agent = ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build(),
    );
    agent.get(url).call().map_err(std::io::Error::other)?;
    Ok(())
}
fn setup(
    id: Option<&str>,
) -> (
    tempfile::TempDir,
    polylogue_store::fixture::Fixture,
    Startup,
) {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = polylogue_store::fixture::Fixture::new(&dir.path().join("archive")).unwrap();
    if let Some(id) = id {
        fixture.setting("drive.client_id", id).unwrap();
    }
    let startup = Startup::for_test(
        crate::test_daemon::engine_config(dir.path().join("home")),
        &fixture.archive.root,
    );
    (dir, fixture, startup)
}
fn install(startup: &Startup, driver: &auth::Drive, extra: &[&str]) -> (ExitCode, Vec<u8>) {
    let mut args = vec!["polylogue", "install", "drive"];
    args.extend_from_slice(extra);
    let (_, parsed) = crate::cli::parse(&args).unwrap();
    let mut output = vec![];
    let exit = super::super::install_with_output(&parsed, startup, true, &mut output, driver);
    (exit, output)
}
fn valid(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, br#"{"type":"authorized_user","client_id":"synthetic-client","refresh_token":"retained-refresh"}"#).unwrap();
}

#[test]
fn install_drive_dry_run_has_scope_pkce_and_no_requests_or_files() {
    let (_dir, fixture, startup) = setup(Some("synthetic-client"));
    let server = Provider::good();
    let (exit, output) = install(&startup, &server.driver(), &["--dry-run"]);
    assert_eq!(exit, ExitCode::SUCCESS);
    let report: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(report["status"], "authorization_dry_run");
    assert_eq!(report["scope"], auth::SCOPE);
    let url = reqwest::Url::parse(report["url"].as_str().unwrap()).unwrap();
    let query = pairs(url.query().unwrap());
    assert_eq!(query["client_id"], "synthetic-client");
    assert_eq!(query["code_challenge_method"], "S256");
    assert_eq!(query["access_type"], "offline");
    assert!(!token_path(&fixture.archive.root).exists());
    assert!(server.finish().is_empty());
}

#[test]
fn install_drive_authorizes_reopens_collects_and_removes_through_one_token() {
    use std::os::unix::fs::PermissionsExt;
    let (_dir, fixture, startup) = setup(Some("synthetic-client"));
    let server = Provider::good();
    let driver = server.driver();
    assert_eq!(install(&startup, &driver, &[]).0, ExitCode::SUCCESS);
    let path = token_path(&fixture.archive.root);
    let bytes = std::fs::read(&path).unwrap();
    let user: AuthorizedUser = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(user.client_id, "synthetic-client");
    assert_eq!(user.refresh_token, "retained-refresh");
    assert!(user.client_secret.is_none());
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let restarted = Startup::for_test(
        crate::test_daemon::engine_config(startup.config.paths.home.clone()),
        &fixture.archive.root,
    );
    assert_eq!(install(&restarted, &driver, &[]).0, ExitCode::SUCCESS);
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let collector = polylogue_sources::ai_studio::drive::Drive {
        endpoints: polylogue_sources::ai_studio::drive::Endpoints {
            api: format!("{}/drive", server.base),
            token: driver.token.clone(),
        },
        token_path: path.clone(),
        spool: fixture.archive.root.join("spool"),
    };
    assert_eq!(collector.sync().unwrap().prompts, 0);
    assert_eq!(
        install(&restarted, &driver, &["--remove", "--dry-run"]).0,
        ExitCode::SUCCESS
    );
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    assert_eq!(
        install(&restarted, &driver, &["--remove"]).0,
        ExitCode::SUCCESS
    );
    assert!(!path.exists());
    assert_eq!(
        install(&restarted, &driver, &["--remove"]).0,
        ExitCode::SUCCESS
    );
    let seen = server.finish();
    let authorization = seen
        .iter()
        .find(|request| request.path == "/authorize")
        .unwrap();
    assert_eq!(authorization.query["scope"], auth::SCOPE);
    let exchanges: Vec<_> = seen
        .iter()
        .filter(|request| request.path == "/token")
        .collect();
    assert_eq!(exchanges.len(), 2);
    assert_eq!(exchanges[0].form["grant_type"], "authorization_code");
    assert_eq!(exchanges[0].form["client_id"], "synthetic-client");
    assert_eq!(
        exchanges[0].form["redirect_uri"],
        authorization.query["redirect_uri"]
    );
    assert_eq!(
        exchanges[0].form["code"],
        auth::pkce_challenge(&exchanges[0].form["code_verifier"])
    );
    assert_eq!(
        exchanges[0].form["code"],
        authorization.query["code_challenge"]
    );
    assert!(exchanges[0].form["code_verifier"].len() >= 43);
    assert_eq!(exchanges[1].form["grant_type"], "refresh_token");
    assert!(!exchanges[1].form.contains_key("client_secret"));
    assert_eq!(
        seen.iter()
            .filter(|request| request.path == "/revoke")
            .count(),
        1
    );
}

#[test]
fn install_drive_refused_callback_and_exchange_keep_credentials_absent() {
    for (consent, status, token) in [
        (Consent::Deny, 200, json!({"refresh_token":"unexpected"})),
        (
            Consent::WrongState,
            200,
            json!({"refresh_token":"unexpected"}),
        ),
        (Consent::Allow, 503, json!({"error":"unavailable"})),
        (
            Consent::Allow,
            200,
            json!({"access_token":"no-offline-token"}),
        ),
        (
            Consent::Allow,
            200,
            json!({"refresh_token":"other","scope":"unrelated"}),
        ),
    ] {
        let (_dir, fixture, startup) = setup(Some("synthetic-client"));
        let server = Provider::new(consent, status, token, 200);
        assert_eq!(
            install(&startup, &server.driver(), &[]).0,
            ExitCode::from(4)
        );
        assert!(!token_path(&fixture.archive.root).exists());
        let seen = server.finish();
        if matches!(consent, Consent::Deny | Consent::WrongState) {
            assert_eq!(
                seen.iter()
                    .filter(|request| request.path == "/token")
                    .count(),
                0
            );
        }
    }
}

#[test]
fn install_drive_failed_revocation_keeps_exact_credentials() {
    let (_dir, fixture, startup) = setup(None);
    let path = token_path(&fixture.archive.root);
    valid(&path);
    let original = std::fs::read(&path).unwrap();
    let server = Provider::new(Consent::Allow, 200, json!({}), 503);
    assert_eq!(
        install(&startup, &server.driver(), &["--remove"]).0,
        ExitCode::from(4)
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert_eq!(server.finish().len(), 1);
}

#[test]
fn install_drive_timeout_releases_listener_and_preserves_no_token() {
    let (_dir, fixture, startup) = setup(Some("synthetic-client"));
    let server = Provider::new(Consent::Wait, 200, json!({}), 200);
    assert_eq!(
        install(&startup, &server.driver(), &["--deadline", "10"]).0,
        ExitCode::from(4)
    );
    assert!(!token_path(&fixture.archive.root).exists());
    let seen = server.finish();
    let address = reqwest::Url::parse(&seen[0].query["redirect_uri"]).unwrap();
    let listener =
        TcpListener::bind((address.host_str().unwrap(), address.port().unwrap())).unwrap();
    drop(listener);
}

#[test]
fn install_drive_refuses_managed_credentials_and_missing_client_id_without_network() {
    let (dir, fixture, startup) = setup(None);
    let server = Provider::good();
    assert_eq!(
        install(&startup, &server.driver(), &[]).0,
        ExitCode::from(2)
    );
    let source = dir.path().join("managed.json");
    valid(&source);
    let original = std::fs::read(&source).unwrap();
    let path = token_path(&fixture.archive.root);
    std::os::unix::fs::symlink(&source, &path).unwrap();
    assert_eq!(
        install(&startup, &server.driver(), &["--remove"]).0,
        ExitCode::from(4)
    );
    assert!(path.is_symlink());
    assert_eq!(std::fs::read(&source).unwrap(), original);
    assert!(server.finish().is_empty());
}

#[test]
fn install_drive_already_revoked_token_can_be_removed() {
    let (_dir, fixture, startup) = setup(None);
    let path = token_path(&fixture.archive.root);
    valid(&path);
    let server = Provider::new(Consent::Allow, 200, json!({}), 400);
    assert_eq!(
        install(&startup, &server.driver(), &["--remove"]).0,
        ExitCode::SUCCESS
    );
    assert!(!path.exists());
    assert_eq!(server.finish().len(), 1);
}

#[test]
fn collector_keeps_legacy_client_secret_refresh_semantics() {
    let (_dir, fixture, _startup) = setup(None);
    let path = token_path(&fixture.archive.root);
    std::fs::write(&path, br#"{"type":"authorized_user","client_id":"old-client","client_secret":"old-secret","refresh_token":"old-refresh"}"#).unwrap();
    let original = std::fs::read(&path).unwrap();
    let server = Provider::good();
    let collector = polylogue_sources::ai_studio::drive::Drive {
        endpoints: polylogue_sources::ai_studio::drive::Endpoints {
            api: format!("{}/drive", server.base),
            token: format!("{}/token", server.base),
        },
        token_path: path.clone(),
        spool: fixture.archive.root.join("spool"),
    };
    assert_eq!(collector.sync().unwrap().prompts, 0);
    assert_eq!(std::fs::read(path).unwrap(), original);
    let seen = server.finish();
    let refresh = seen
        .iter()
        .find(|request| request.path == "/token")
        .unwrap();
    assert_eq!(refresh.form["client_secret"], "old-secret");
    assert_eq!(refresh.form["refresh_token"], "old-refresh");
}

#[test]
fn historical_drive_intent_survives_normal_reader_reopen_without_becoming_a_control() {
    let (dir, mut fixture, _) = setup(None);
    fixture.setting("drive.ai-studio", "off").unwrap();
    let archive = fixture.archive.clone();
    drop(fixture);
    for _ in 0..2 {
        let pool = polylogue_store::ReadPool::open(&archive).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader
                .setting_value("drive.ai-studio", "synthetic-host")
                .unwrap(),
            Some(polylogue_model::settings::Value::Boolean(false))
        );
        assert!(
            reader
                .settings_catalog("synthetic-host")
                .unwrap()
                .iter()
                .all(|current| current.key != "drive.ai-studio")
        );
    }
    assert!(!token_path(&dir.path().join("archive")).exists());
}

#[test]
fn install_drive_keeps_token_published_during_consent() {
    let (_dir, fixture, startup) = setup(Some("synthetic-client"));
    let path = token_path(&fixture.archive.root);
    let server = Provider::good();
    *server.publish.lock().unwrap() = Some(path.clone());
    assert_eq!(
        install(&startup, &server.driver(), &[]).0,
        ExitCode::from(4)
    );
    let user: AuthorizedUser = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(user.refresh_token, "winner-token");
    assert_eq!(
        std::fs::read_dir(&fixture.archive.root)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp"))
            .count(),
        0
    );
    let seen = server.finish();
    assert_eq!(
        seen.iter()
            .filter(|request| request.path == "/token")
            .count(),
        1
    );
}
