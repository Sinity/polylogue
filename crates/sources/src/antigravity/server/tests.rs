use super::*;
use serde_json::json;
use std::io::{BufRead, BufReader, Read};
use std::net::TcpListener;
use std::thread;

fn server(replies: Vec<(String, Json, u16, Json)>) -> (Server, thread::JoinHandle<()>) {
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
            let body = serde_json::to_vec(&reply).unwrap();
            write!(socket,"HTTP/1.1 {status} Reply\r\nContent-Length: {}\r\nConnection: close\r\nContent-Type: application/json\r\n\r\n",body.len()).unwrap();
            socket.write_all(&body).unwrap();
        }
    });
    (Server::new(port, "synthetic-token".into()), task)
}

fn search() -> (String, Json, u16, Json) {
    (
        "SearchConversations".into(),
        json!({"query":""}),
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
    let (server, task) = server(vec![
        search(),
        markdown(
            "one",
            json!({"markdown":"### User Input\nQuestion\n### Planner Response\nAnswer"}),
        ),
    ]);
    assert!(server.agent.config().proxy().is_none());
    assert_eq!(server.agent.config().max_redirects(), 0);
    assert!(sync_paths(std::slice::from_ref(&path), &spool, &[server]).is_empty());
    task.join().unwrap();
    let file = fs::read_dir(&spool)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let bytes = fs::read(file).unwrap();
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
    assert!(sync_paths(&[path], &spool, &[]).is_empty());
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
        search(),
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
    assert_eq!(fs::read_dir(spool).unwrap().count(), 1);
}

#[test]
fn reports_each_unconverted_trajectory_without_a_server() {
    let errors = sync_paths(
        &[PathBuf::from("one.pb"), PathBuf::from("two.pb")],
        Path::new("unused"),
        &[],
    );
    assert_eq!(errors.len(), 2);
    assert!(errors[0].contains("one.pb not converted"));
    assert!(errors[1].contains("two.pb not converted"));
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
    let paths = [PathBuf::from("a/same.pb"), PathBuf::from("b/same.pb")];
    let errors = sync_paths(&paths, Path::new("unused"), &[]);
    assert_eq!(errors.len(), 2);
    assert!(
        errors
            .iter()
            .all(|e| e.contains("duplicate conversation identity"))
    );
}
