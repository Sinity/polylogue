use super::*;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

/// A fake Drive: answers token refresh, folder lookup, the listing, the
/// revisions of one prompt and each revision's bytes. Records each path.
fn serve(revisions: Vec<(&'static str, &'static str)>) -> (String, Arc<Mutex<Vec<String>>>) {
    serve_paged(revisions, false)
}

fn serve_paged(
    revisions: Vec<(&'static str, &'static str)>,
    paged: bool,
) -> (String, Arc<Mutex<Vec<String>>>) {
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
                let later = target.contains("pageToken=later");
                let page = if paged {
                    if later {
                        &revisions[1..]
                    } else {
                        &revisions[..1]
                    }
                } else {
                    revisions.as_slice()
                };
                let items: Vec<String> = page
                    .iter()
                    .map(|(id, at)| format!(r#"{{"id":"{id}","modifiedTime":"{at}"}}"#))
                    .collect();
                let next = if paged && !later {
                    r#", "nextPageToken":"later""#
                } else {
                    ""
                };
                format!(r#"{{"revisions":[{}]{next}}}"#, items.join(","))
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

fn collector(base: &str, dir: &Path) -> Drive {
    let token_path = dir.join("token.json");
    std::fs::write(
        &token_path,
        r#"{"client_id":"c","client_secret":"s","refresh_token":"r","token":"old"}"#,
    )
    .unwrap();
    Drive {
        endpoints: Endpoints {
            api: format!("{base}/drive"),
            token: format!("{base}/token"),
        },
        token_path,
        spool: dir.join("spool"),
    }
}

#[test]
fn a_revision_survives_repeated_sync_and_restart_until_exact_archive_acceptance() {
    let (base, seen) = serve_paged(
        vec![
            ("r2", "2026-05-17T12:30:00Z"),
            ("r1", "2026-05-17T12:20:00Z"),
        ],
        true,
    );
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&base, dir.path());
    let prompt = c.spool.join("1AbC.json");

    let first = c.sync().unwrap();
    assert_eq!(first.revisions_written, 1);
    assert_eq!(first.pending, 2);
    assert!(std::fs::read_to_string(&prompt).unwrap().contains("\"r1\""));
    let listing = std::fs::read_to_string(c.spool.join("files.json")).unwrap();
    assert!(listing.contains("Electrons"));

    let original = std::fs::read(&prompt).unwrap();
    let retained = c.pending_revisions().unwrap().remove("1AbC").unwrap();
    assert_eq!(std::fs::read(&retained.body).unwrap(), original);
    assert!(c.load_state().unwrap().is_empty());
    assert_eq!(c.sync().unwrap().revisions_written, 0);
    assert_eq!(std::fs::read(&prompt).unwrap(), original);
    let c = collector(&base, dir.path());
    c.settle(&|_, _, _| Ok(false)).unwrap();
    assert!(c.settle(&|_, _, _| Err("archive rejected".into())).is_err());
    assert_eq!(c.sync().unwrap().revisions_written, 0);
    assert_eq!(std::fs::read(&prompt).unwrap(), original);
    assert_eq!(std::fs::read(&retained.body).unwrap(), original);
    assert!(c.load_state().unwrap().is_empty());
    assert_eq!(
        seen.lock()
            .unwrap()
            .iter()
            .filter(|path| path.starts_with("/drive/files/1AbC/revisions/r1?"))
            .count(),
        1
    );
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .any(|path| path.contains("pageToken=later"))
    );

    c.settle(&|path, hash, length| {
        assert_eq!(path, prompt);
        assert_eq!(hash, polylogue_model::BlobHash::of(&original));
        assert_eq!(length, original.len() as u64);
        Ok(true)
    })
    .unwrap();
    assert_eq!(c.load_state().unwrap().get("1AbC").unwrap(), "r1");
    assert!(!retained.body.exists());
    assert_eq!(std::fs::read(&prompt).unwrap(), original);
    let second = c.sync().unwrap();
    assert_eq!((second.revisions_written, second.pending), (1, 1));
    assert!(std::fs::read_to_string(&prompt).unwrap().contains("\"r2\""));

    c.settle(&|_, _, _| Ok(true)).unwrap();
    let third = c.sync().unwrap();
    assert_eq!(third.revisions_written, 0);
}

#[test]
fn a_missing_token_is_a_typed_error() {
    let dir = tempfile::tempdir().unwrap();
    let c = Drive {
        endpoints: Endpoints::default(),
        token_path: dir.path().join("absent.json"),
        spool: dir.path().join("spool"),
    };
    assert!(matches!(c.sync(), Err(DriveError::Token(..))));
}

#[test]
fn an_unpublished_body_is_reused_and_conflicting_bytes_are_refused() {
    let (base, _) = serve(vec![("r1", "2026-05-17T12:20:00Z")]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&base, dir.path());
    let body = br#"{"runSettings":{},"chunkedPrompt":{"chunks":[]}}"#;
    let retained = c.retain_revision("1AbC", Some("r1"), body).unwrap();
    assert!(!c.spool.join("1AbC.json").exists());
    assert!(c.retain_revision("1AbC", Some("r1"), b"changed").is_err());
    assert_eq!(c.sync().unwrap().revisions_written, 0);
    assert_eq!(std::fs::read(c.spool.join("1AbC.json")).unwrap(), body);
    assert_eq!(std::fs::read(retained.body).unwrap(), body);
}

#[test]
fn an_accepted_cursor_survives_restart_before_pending_cleanup() {
    let dir = tempfile::tempdir().unwrap();
    let c = collector("http://127.0.0.1:1", dir.path());
    let retained = c.retain_revision("1AbC", Some("r1"), b"original").unwrap();
    c.publish(&retained).unwrap();
    c.save_state(&BTreeMap::from([("1AbC".into(), "r1".into())]))
        .unwrap();
    c.settle(&|_, _, _| panic!("durable acceptance was already saved"))
        .unwrap();
    assert!(!retained.body.exists());
    assert_eq!(
        std::fs::read(c.spool.join("1AbC.json")).unwrap(),
        b"original"
    );
    assert_eq!(c.load_state().unwrap().get("1AbC").unwrap(), "r1");
}

#[test]
fn the_predecessor_download_cursor_needs_archive_acceptance() {
    let (base, _) = serve(vec![
        ("r1", "2026-05-17T12:20:00Z"),
        ("r2", "2026-05-17T12:30:00Z"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&base, dir.path());
    std::fs::create_dir_all(&c.spool).unwrap();
    std::fs::write(c.spool.join(STATE), br#"{"1AbC":"r1"}"#).unwrap();
    std::fs::write(c.spool.join("1AbC.json"), b"legacy bytes").unwrap();
    assert_eq!(c.sync().unwrap().revisions_written, 0);
    assert!(c.load_state().unwrap().is_empty());
    assert_eq!(
        std::fs::read(c.spool.join("1AbC.json")).unwrap(),
        b"legacy bytes"
    );
    c.settle(&|_, hash, _| {
        assert_eq!(hash, polylogue_model::BlobHash::of(b"legacy bytes"));
        Ok(true)
    })
    .unwrap();
    assert!(c.load_state().unwrap().is_empty());
    assert_eq!(c.sync().unwrap().revisions_written, 1);
    assert!(
        std::fs::read_to_string(c.spool.join("1AbC.json"))
            .unwrap()
            .contains("r1")
    );
    c.settle(&|_, _, _| Ok(true)).unwrap();
    assert_eq!(c.sync().unwrap().revisions_written, 1);
    assert!(
        std::fs::read_to_string(c.spool.join("1AbC.json"))
            .unwrap()
            .contains("r2")
    );
}

#[test]
fn corrupt_or_newer_revision_state_is_preserved_and_refused() {
    let dir = tempfile::tempdir().unwrap();
    let c = collector("http://127.0.0.1:1", dir.path());
    std::fs::create_dir_all(&c.spool).unwrap();
    for state in [b"broken".as_slice(), br#"{"version":2,"accepted":{}}"#] {
        std::fs::write(c.spool.join(STATE), state).unwrap();
        assert!(c.sync().is_err());
        assert_eq!(std::fs::read(c.spool.join(STATE)).unwrap(), state);
    }
}
