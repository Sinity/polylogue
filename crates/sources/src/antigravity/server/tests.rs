use super::*;
use serde_json::json;
use std::io::{BufRead, BufReader, Read};
use std::net::TcpListener;
use std::thread;

fn server(replies: Vec<(String, Json, u16, Json)>) -> (Server, thread::JoinHandle<()>) {
    server_with_change(replies, None)
}

fn server_with_change(
    replies: Vec<(String, Json, u16, Json)>,
    change: Option<PathBuf>,
) -> (Server, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = thread::spawn(move || {
        for (method, expected, status, reply) in replies {
            let (mut socket, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            assert!(
                line.starts_with(&format!("POST /{SERVICE}/{method} HTTP/1.1")),
                "{line}"
            );
            let mut length = None;
            let mut token = false;
            loop {
                line.clear();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                let lower = line.to_ascii_lowercase();
                if let Some(value) = lower.strip_prefix("content-length:") {
                    length = Some(value.trim().parse::<usize>().unwrap());
                }
                if lower.trim() == "x-codeium-csrf-token: synthetic-token" {
                    token = true;
                }
            }
            assert!(token);
            let mut body = vec![0; length.unwrap()];
            reader.read_exact(&mut body).unwrap();
            assert_eq!(serde_json::from_slice::<Json>(&body).unwrap(), expected);
            if method == "ConvertTrajectoryToMarkdown"
                && let Some(path) = &change
            {
                fs::write(path, b"changed during conversion").unwrap();
            }
            let body = serde_json::to_vec(&reply).unwrap();
            write!(socket,"HTTP/1.1 {status} Reply\r\nContent-Length: {}\r\nConnection: close\r\nContent-Type: application/json\r\n\r\n",body.len()).unwrap();
            socket.write_all(&body).unwrap();
        }
    });
    (Server::new(port, "synthetic-token".into()), task)
}

fn search(limit: usize) -> (String, Json, u16, Json) {
    (
        "SearchConversations".into(),
        json!({"query":"","limit":limit}),
        200,
        json!({"results":[{"cascadeId":"one","title":"Synthetic title"}]}),
    )
}
fn markdown(id: &str, value: Json) -> (String, Json, u16, Json) {
    (
        "ConvertTrajectoryToMarkdown".into(),
        json!({"conversationId":id}),
        200,
        value,
    )
}

#[test]
fn converts_vendor_markdown_and_keeps_it_for_offline_rebuild() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("one.pb");
    fs::write(&path, b"synthetic opaque bytes").unwrap();
    let spool = temp.path().join("spool");
    let (client, task) = server(vec![
        search(1),
        markdown(
            "one",
            json!({"markdown":"### User Input\nQuestion\n### Planner Response\nAnswer"}),
        ),
    ]);
    assert!(client.agent.config().proxy().is_none());
    assert_eq!(client.agent.config().max_redirects(), 0);
    assert!(sync_paths(std::slice::from_ref(&path), &spool, &[client]).is_empty());
    task.join().unwrap();
    let bytes = fs::read(destination(&path, &spool)).unwrap();
    let parsed = crate::parse::run_document(&super::super::Antigravity, &bytes).unwrap();
    assert_eq!(parsed.output.messages.len(), 2);
    assert_eq!(
        parsed.output.sessions[0]
            .title
            .as_ref()
            .map(|t| t.text.as_str()),
        Some("Synthetic title")
    );
    assert!(
        !String::from_utf8(bytes)
            .unwrap()
            .contains("synthetic-token")
    );
    assert!(sync_paths(std::slice::from_ref(&path), &spool, &[]).is_empty());
    let second = temp.path().join("two.pb");
    fs::write(&second, b"another trajectory").unwrap();
    let (client, task) = server(vec![
        search(2),
        markdown(
            "two",
            json!({"markdown":"### User Input\nAnother question"}),
        ),
    ]);
    assert!(sync_paths(&[path, second], &spool, &[client]).is_empty());
    task.join().unwrap();
}

#[test]
fn isolates_failed_and_partial_conversions() {
    let temp = tempfile::tempdir().unwrap();
    let paths: Vec<_> = ["bad", "empty", "good"]
        .iter()
        .map(|id| {
            let p = temp.path().join(format!("{id}.pb"));
            fs::write(&p, b"opaque").unwrap();
            p
        })
        .collect();
    let (server, task) = server(vec![
        search(3),
        (
            "ConvertTrajectoryToMarkdown".into(),
            json!({"conversationId":"bad"}),
            500,
            json!({}),
        ),
        markdown("empty", json!({"markdown":"### User Input\n"})),
        markdown("good", json!({"markdown":"### Planner Response\nAnswer"})),
    ]);
    let spool = temp.path().join("spool");
    let errors = sync_paths(&paths, &spool, &[server]);
    task.join().unwrap();
    assert_eq!(errors.len(), 2);
    assert!(errors.iter().all(|e| e.contains("not converted")));
    assert!(destination(&paths[2], &spool).exists());
    assert!(!destination(&paths[0], &spool).exists());
    assert!(!destination(&paths[1], &spool).exists());
}

#[test]
fn unavailable_server_is_a_session_event_written_once_until_it_changes() {
    let temp = tempfile::tempdir().unwrap();
    let spool = temp.path().join("spool");
    let path = temp.path().join("one.pb");
    fs::write(&path, b"opaque").unwrap();
    assert!(sync_paths(std::slice::from_ref(&path), &spool, &[]).is_empty());
    let status_path = destination(&path, &spool).with_extension("status.json");
    let before = fs::metadata(&status_path).unwrap().modified().unwrap();
    let bytes = fs::read(&status_path).unwrap();
    let parsed = crate::parse::run_document(&super::super::Antigravity, &bytes).unwrap();
    assert_eq!(parsed.output.events[0].kind, "antigravity_conversion");
    assert_eq!(parsed.output.events[0].data["status"], "server_unavailable");
    assert_eq!(
        parsed.output.events[0]
            .session
            .as_ref()
            .unwrap()
            .to_string(),
        "antigravity:one"
    );
    assert!(sync_paths(std::slice::from_ref(&path), &spool, &[]).is_empty());
    assert_eq!(
        before,
        fs::metadata(status_path).unwrap().modified().unwrap()
    );
    let (client, task) = server(vec![
        search(1),
        markdown("one", json!({"markdown":"### User Input\nQuestion"})),
    ]);
    assert!(sync_paths(&[path.clone()], &spool, &[client]).is_empty());
    task.join().unwrap();
    let status: Json = serde_json::from_slice(
        &fs::read(destination(&path, &spool).with_extension("status.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(status["status"], "converted");
}

#[test]
fn discovery_reads_only_the_named_language_server_process() {
    let temp = tempfile::tempdir().unwrap();
    let daemon = temp.path().join("daemon");
    fs::create_dir(&daemon).unwrap();
    let proc = temp.path().join("processes");
    fs::create_dir_all(proc.join("42")).unwrap();
    fs::write(
        daemon.join("ls_test.json"),
        br#"{"pid":42,"httpPort":12345}"#,
    )
    .unwrap();
    fs::write(
        proc.join("42/cmdline"),
        b"/vendor/language_server_linux_x64\0-csrf_token=synthetic-token\0",
    )
    .unwrap();
    let found = discover(temp.path(), &proc);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].port, 12345);
    assert_eq!(found[0].token, "synthetic-token");
    fs::write(
        proc.join("42/cmdline"),
        b"/other/process\0-csrf_token=synthetic-token\0",
    )
    .unwrap();
    assert!(discover(temp.path(), &proc).is_empty());
}

#[test]
fn duplicate_conversation_ids_are_reported_without_conversion() {
    let temp = tempfile::tempdir().unwrap();
    let paths = [temp.path().join("a/same.pb"), temp.path().join("b/same.pb")];
    for path in &paths {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"opaque").unwrap();
    }
    let errors = sync_paths(&paths, &temp.path().join("spool"), &[]);
    assert_eq!(errors.len(), 2);
    assert!(
        errors
            .iter()
            .all(|e| e.contains("duplicate conversation identity"))
    );
}

#[test]
fn refuses_an_export_if_the_trajectory_changes_during_conversion() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("one.pb");
    fs::write(&path, b"original trajectory").unwrap();
    let spool = temp.path().join("spool");
    let (client, task) = server_with_change(
        vec![
            search(1),
            markdown("one", json!({"markdown":"### Planner Response\nAnswer"})),
        ],
        Some(path.clone()),
    );
    let errors = sync_paths(std::slice::from_ref(&path), &spool, &[client]);
    task.join().unwrap();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("trajectory changed during conversion"));
    assert!(!destination(&path, &spool).exists());
}

#[test]
fn conversion_uses_size_and_nanosecond_mtime_without_reading_source_bytes() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("one.pb");
    fs::write(&path, b"opaque").unwrap();
    // Metadata stays accessible while the trajectory content cannot be read.
    fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();
    let spool = temp.path().join("spool");
    let (client, task) = server(vec![
        search(1),
        markdown("one", json!({"markdown":"### User Input\nQuestion"})),
    ]);
    assert!(sync_paths(std::slice::from_ref(&path), &spool, &[client]).is_empty());
    task.join().unwrap();
    assert!(sync_paths(std::slice::from_ref(&path), &spool, &[]).is_empty());
    let first = FileState::read(&path).unwrap();
    assert!(converted(&path, &spool, first));
    assert!(!converted(
        &path,
        &spool,
        FileState {
            size: first.size + 1,
            ..first
        }
    ));
    assert!(!converted(
        &path,
        &spool,
        FileState {
            modified_nanoseconds: first.modified_nanoseconds + 1,
            ..first
        }
    ));
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_times(
        std::fs::FileTimes::new().set_modified(std::time::UNIX_EPOCH + Duration::from_secs(123)),
    )
    .unwrap();
    assert!(!converted(&path, &spool, FileState::read(&path).unwrap()));
    let (client, task) = server(vec![
        search(1),
        markdown(
            "one",
            json!({"markdown":"### User Input\nChanged question"}),
        ),
    ]);
    assert!(sync_paths(std::slice::from_ref(&path), &spool, &[client]).is_empty());
    task.join().unwrap();
    assert!(converted(&path, &spool, FileState::read(&path).unwrap()));
}

#[test]
fn an_unresponsive_server_has_a_short_request_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (stop, wait) = std::sync::mpsc::channel::<()>();
    let task = thread::spawn(move || {
        let (_socket, _) = listener.accept().unwrap();
        let _ = wait.recv();
    });
    let mut client = Server::new(port, "synthetic-token".into());
    assert_eq!(
        client.agent.config().timeout_global(),
        Some(Duration::from_secs(5))
    );
    client.agent = ureq::Agent::config_builder()
        .proxy(None)
        .max_redirects(0)
        .timeout_global(Some(Duration::from_millis(50)))
        .build()
        .into();
    let started = std::time::Instant::now();
    assert!(client.post("SearchConversations", json!({})).is_err());
    stop.send(()).unwrap();
    task.join().unwrap();
    assert!(started.elapsed() < Duration::from_secs(2));
}
