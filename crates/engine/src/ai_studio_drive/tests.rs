use super::*;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

/// A fake Drive: answers token refresh, folder lookup, the listing, the
/// revisions of one prompt and each revision's bytes. Records each path.
fn serve(revisions: Vec<(&'static str, &'static str)>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let target = line.split_whitespace().nth(1).unwrap_or("").to_owned();
            let mut len = 0usize;
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h == "\r\n" || h.is_empty() {
                    break;
                }
                if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; len];
            std::io::Read::read_exact(&mut reader, &mut body).unwrap();
            log.lock().unwrap().push(target.clone());
            let path = target.split('?').next().unwrap();
            let reply = if path == "/token" {
                r#"{"access_token":"t","expires_in":3600}"#.to_owned()
            } else if path == "/drive/files" && target.contains("apps.folder") {
                r#"{"files":[{"id":"folder1"}]}"#.to_owned()
            } else if path == "/drive/files" {
                r#"{"files":[{"id":"1AbC","name":"Electrons","createdTime":"2026-05-17T12:20:00Z"}]}"#
                    .to_owned()
            } else if path == "/drive/files/1AbC/revisions" {
                let items: Vec<String> = revisions
                    .iter()
                    .map(|(id, at)| format!(r#"{{"id":"{id}","modifiedTime":"{at}"}}"#))
                    .collect();
                format!(r#"{{"revisions":[{}]}}"#, items.join(","))
            } else if let Some(rev) = path.strip_prefix("/drive/files/1AbC/revisions/") {
                format!(
                    r#"{{"runSettings":{{}},"chunkedPrompt":{{"chunks":[{{"role":"user","text":"{rev}"}}]}}}}"#
                )
            } else {
                "{}".to_owned()
            };
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                reply.len()
            );
        }
    });
    (base, seen)
}

fn collector(base: &str, dir: &Path) -> Collector {
    let token_path = dir.join("token.json");
    std::fs::write(
        &token_path,
        r#"{"client_id":"c","client_secret":"s","refresh_token":"r","token":"old"}"#,
    )
    .unwrap();
    Collector {
        endpoints: Endpoints {
            api: format!("{base}/drive"),
            token: format!("{base}/token"),
        },
        token_path,
        spool: dir.join("spool"),
    }
}

#[test]
fn each_sync_writes_the_next_revision_until_all_are_written() {
    let (base, _) = serve(vec![
        ("r2", "2026-05-17T12:30:00Z"),
        ("r1", "2026-05-17T12:20:00Z"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&base, dir.path());
    let prompt = c.spool.join("1AbC.json");

    let first = c.sync().unwrap();
    assert_eq!(first.revisions_written, 1);
    assert_eq!(first.pending, 1);
    assert!(std::fs::read_to_string(&prompt).unwrap().contains("\"r1\""));
    let listing = std::fs::read_to_string(c.spool.join("files.json")).unwrap();
    assert!(listing.contains("Electrons"));

    let second = c.sync().unwrap();
    assert_eq!((second.revisions_written, second.pending), (1, 0));
    assert!(std::fs::read_to_string(&prompt).unwrap().contains("\"r2\""));

    let third = c.sync().unwrap();
    assert_eq!(third.revisions_written, 0);
}

#[test]
fn a_missing_token_is_a_typed_error() {
    let dir = tempfile::tempdir().unwrap();
    let c = Collector {
        endpoints: Endpoints::default(),
        token_path: dir.path().join("absent.json"),
        spool: dir.path().join("spool"),
    };
    assert!(matches!(c.sync(), Err(DriveError::Token(..))));
}
