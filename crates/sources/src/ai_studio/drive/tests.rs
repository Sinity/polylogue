use super::*;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

/// A fake Drive: answers token refresh, folder lookup, the listing, the
/// revisions of one prompt (`page` to a page, continued by `pageToken`) and
/// each revision's bytes. Records each path. The test may change the
/// revisions between syncs, as Drive prunes them.
struct FakeDrive {
    base: String,
    seen: Arc<Mutex<Vec<String>>>,
    revisions: Arc<Mutex<Vec<(String, String)>>>,
}

fn serve(revisions: Vec<(&'static str, &'static str)>) -> (String, Arc<Mutex<Vec<String>>>) {
    let drive = serve_pages(owned(revisions), usize::MAX);
    (drive.base, drive.seen)
}

fn owned(revisions: Vec<(&str, &str)>) -> Vec<(String, String)> {
    revisions
        .into_iter()
        .map(|(id, at)| (id.to_owned(), at.to_owned()))
        .collect()
}

fn revision(id: &str) -> Accepted {
    Accepted {
        revision: id.to_owned(),
        modified: None,
    }
}

fn serve_pages(revisions: Vec<(String, String)>, page: usize) -> FakeDrive {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let revisions = Arc::new(Mutex::new(revisions));
    let log = Arc::clone(&seen);
    let listed = Arc::clone(&revisions);
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
                let revisions = listed.lock().unwrap().clone();
                let from: usize = target
                    .split("pageToken=")
                    .nth(1)
                    .map_or(0, |token| token.split('&').next().unwrap().parse().unwrap());
                let to = from.saturating_add(page).min(revisions.len());
                let items: Vec<String> = revisions[from..to]
                    .iter()
                    .map(|(id, at)| format!(r#"{{"id":"{id}","modifiedTime":"{at}"}}"#))
                    .collect();
                let next = if to < revisions.len() {
                    format!(r#", "nextPageToken":"{to}""#)
                } else {
                    String::new()
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
    FakeDrive {
        base,
        seen,
        revisions,
    }
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
fn revision_survives_repeated_sync_and_restart_until_archive_acceptance() {
    let FakeDrive { base, seen, .. } = serve_pages(
        owned(vec![
            ("r2", "2026-05-17T12:30:00Z"),
            ("r1", "2026-05-17T12:20:00Z"),
        ]),
        1,
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
    let kept = c.pending_revisions().unwrap().remove("1AbC").unwrap();
    assert_eq!(std::fs::read(&kept.body).unwrap(), original);
    assert!(c.load_state().unwrap().accepted.is_empty());
    assert_eq!(c.sync().unwrap().revisions_written, 0);
    assert_eq!(std::fs::read(&prompt).unwrap(), original);
    let c = collector(&base, dir.path());
    c.advance(&|_, _, _| Ok(false)).unwrap();
    assert!(
        c.advance(&|_, _, _| Err("archive rejected".into()))
            .is_err()
    );
    assert_eq!(c.sync().unwrap().revisions_written, 0);
    assert_eq!(std::fs::read(&prompt).unwrap(), original);
    assert_eq!(std::fs::read(&kept.body).unwrap(), original);
    assert!(c.load_state().unwrap().accepted.is_empty());
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
            .any(|path| path.contains("pageToken=1"))
    );

    c.advance(&|path, hash, length| {
        assert_eq!(path, prompt);
        assert_eq!(hash, polylogue_model::BlobHash::of(&original));
        assert_eq!(length, original.len() as u64);
        Ok(true)
    })
    .unwrap();
    assert_eq!(c.load_state().unwrap().accepted.get("1AbC").unwrap(), "r1");
    assert!(!kept.body.exists());
    assert_eq!(std::fs::read(&prompt).unwrap(), original);
    let second = c.sync().unwrap();
    assert_eq!((second.revisions_written, second.pending), (1, 1));
    assert!(std::fs::read_to_string(&prompt).unwrap().contains("\"r2\""));

    c.advance(&|_, _, _| Ok(true)).unwrap();
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
    let kept = c.keep_revision("1AbC", Some(revision("r1")), body).unwrap();
    assert!(!c.spool.join("1AbC.json").exists());
    let again = c.keep_revision("1AbC", Some(revision("r1")), body).unwrap();
    assert_eq!(again.body, kept.body);
    assert!(matches!(
        c.keep_revision("1AbC", Some(revision("r1")), b"changed"),
        Err(DriveError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidData
    ));
    assert_eq!(c.sync().unwrap().revisions_written, 0);
    assert_eq!(std::fs::read(c.spool.join("1AbC.json")).unwrap(), body);
    assert_eq!(std::fs::read(kept.body).unwrap(), body);
}

#[test]
fn an_accepted_cursor_survives_restart_before_pending_cleanup() {
    let dir = tempfile::tempdir().unwrap();
    let c = collector("http://127.0.0.1:1", dir.path());
    let kept = c
        .keep_revision("1AbC", Some(revision("r1")), b"original")
        .unwrap();
    c.publish(&kept).unwrap();
    c.save_state(&Cursor {
        accepted: BTreeMap::from([("1AbC".into(), "r1".into())]),
        modified: BTreeMap::new(),
    })
    .unwrap();
    c.advance(&|_, _, _| panic!("acceptance was already saved"))
        .unwrap();
    assert!(!kept.body.exists());
    assert_eq!(
        std::fs::read(c.spool.join("1AbC.json")).unwrap(),
        b"original"
    );
    assert_eq!(c.load_state().unwrap().accepted.get("1AbC").unwrap(), "r1");
}

#[test]
fn python_download_cursor_waits_for_archive_acceptance() {
    let (base, _) = serve(vec![
        ("r1", "2026-05-17T12:20:00Z"),
        ("r2", "2026-05-17T12:30:00Z"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&base, dir.path());
    std::fs::create_dir_all(&c.spool).unwrap();
    std::fs::write(c.spool.join(STATE), br#"{"1AbC":"r1"}"#).unwrap();
    std::fs::write(c.spool.join("1AbC.json"), b"older bytes").unwrap();
    assert_eq!(c.sync().unwrap().revisions_written, 0);
    assert!(c.load_state().unwrap().accepted.is_empty());
    assert_eq!(
        std::fs::read(c.spool.join("1AbC.json")).unwrap(),
        b"older bytes"
    );
    c.advance(&|_, hash, _| {
        assert_eq!(hash, polylogue_model::BlobHash::of(b"older bytes"));
        Ok(true)
    })
    .unwrap();
    assert!(c.load_state().unwrap().accepted.is_empty());
    assert_eq!(c.sync().unwrap().revisions_written, 1);
    assert!(
        std::fs::read_to_string(c.spool.join("1AbC.json"))
            .unwrap()
            .contains("r1")
    );
    c.advance(&|_, _, _| Ok(true)).unwrap();
    assert_eq!(c.sync().unwrap().revisions_written, 1);
    assert!(
        std::fs::read_to_string(c.spool.join("1AbC.json"))
            .unwrap()
            .contains("r2")
    );
}

#[test]
fn corrupt_or_newer_revision_state_is_preserved_and_refused() {
    let (base, _) = serve(vec![]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&base, dir.path());
    std::fs::create_dir_all(&c.spool).unwrap();
    for state in [b"broken".as_slice(), br#"{"version":2,"accepted":{}}"#] {
        std::fs::write(c.spool.join(STATE), state).unwrap();
        assert!(matches!(
            c.sync(),
            Err(DriveError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidData
        ));
        assert_eq!(std::fs::read(c.spool.join(STATE)).unwrap(), state);
    }
}

/// A server that answers the token refresh, the folder lookup and the
/// listing (the request asking for `nextPageToken`) with fixed bodies.
fn serve_fixed(folder: &'static str, listing: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
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
            let reply = if target.starts_with("/token") {
                r#"{"access_token":"t","expires_in":3600}"#
            } else if target.contains("nextPageToken") {
                listing
            } else {
                folder
            };
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                reply.len()
            );
        }
    });
    base
}

#[test]
fn a_listing_reply_without_files_keeps_the_saved_listing() {
    let base = serve_fixed(r#"{"files":[{"id":"folder1"}]}"#, "{}");
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&base, dir.path());
    std::fs::create_dir_all(&c.spool).unwrap();
    let saved = br#"{"files":[{"id":"1AbC"}]}"#;
    std::fs::write(c.spool.join("files.json"), saved).unwrap();
    assert!(matches!(c.sync(), Err(DriveError::Http(..))));
    assert_eq!(std::fs::read(c.spool.join("files.json")).unwrap(), saved);
}

#[test]
fn a_missing_folder_is_reported() {
    let base = serve_fixed(r#"{"files":[]}"#, r#"{"files":[]}"#);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&base, dir.path());
    assert!(matches!(c.sync(), Err(DriveError::NoFolder)));
}

#[test]
fn a_prompt_id_cannot_leave_the_spool_or_name_the_listing() {
    let dir = tempfile::tempdir().unwrap();
    let c = collector("http://127.0.0.1:1", dir.path());
    for id in ["", "../escape", "a/b", "a\\b", ".", "..", "files"] {
        assert!(c.prompt_path(id).is_err(), "{id:?}");
        assert!(
            c.keep_revision(id, Some(revision("r1")), b"x").is_err(),
            "{id:?}"
        );
    }
    assert!(!dir.path().join("escape.json").exists());
    assert!(!c.spool.join("files.json").exists());
}

/// Accepts every pending revision and syncs again, until no revision is
/// left; returns the revision texts in the order they were published.
fn walk(c: &Drive) -> Vec<String> {
    let mut published = Vec::new();
    loop {
        c.advance(&|_, _, _| Ok(true)).unwrap();
        if c.sync().unwrap().revisions_written == 0 {
            return published;
        }
        let prompt = std::fs::read_to_string(c.spool.join("1AbC.json")).unwrap();
        let text: serde_json::Value = serde_json::from_str(&prompt).unwrap();
        published.push(
            text.pointer("/chunkedPrompt/chunks/0/text")
                .unwrap()
                .as_str()
                .unwrap()
                .to_owned(),
        );
    }
}

fn minute(n: usize) -> String {
    format!("2026-05-17T{:02}:{:02}:00Z", n / 60, n % 60)
}

#[test]
fn two_pages_of_revisions_all_captured() {
    let revisions: Vec<(String, String)> =
        (0..250).map(|n| (format!("r{n:03}"), minute(n))).collect();
    let drive = serve_pages(revisions.clone(), 100);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive.base, dir.path());
    let first = c.sync().unwrap();
    assert_eq!((first.revisions_written, first.pending), (1, 250));
    let mut published = vec!["r000".to_owned()];
    published.extend(walk(&c));
    let expected: Vec<String> = revisions.into_iter().map(|(id, _)| id).collect();
    assert_eq!(published, expected);
    assert!(
        drive
            .seen
            .lock()
            .unwrap()
            .iter()
            .any(|path| path.contains("pageToken=200"))
    );
}

#[test]
fn resume_after_pruned_revision_continues_from_next_newer() {
    let drive = serve_pages(
        owned(vec![
            ("r1", "2026-05-17T12:10:00Z"),
            ("r2", "2026-05-17T12:20:00Z"),
            ("r3", "2026-05-17T12:30:00Z"),
            ("r4", "2026-05-17T12:40:00Z"),
        ]),
        usize::MAX,
    );
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive.base, dir.path());
    c.sync().unwrap();
    c.advance(&|_, _, _| Ok(true)).unwrap();
    assert_eq!(c.sync().unwrap().revisions_written, 1);
    c.advance(&|_, _, _| Ok(true)).unwrap();
    assert_eq!(c.load_state().unwrap().revision("1AbC"), Some("r2"));
    // Drive prunes the accepted revision and one older than it.
    drive
        .revisions
        .lock()
        .unwrap()
        .retain(|(id, _)| id != "r1" && id != "r2");
    assert_eq!(walk(&c), ["r3", "r4"]);
}

#[test]
fn legacy_token_does_not_authorize_new_collector_or_advance_pending_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("archive");
    let state = dir.path().join("state/polylogue");
    let spool = state.join("drive/ai-studio");
    std::fs::create_dir_all(&archive).unwrap();
    std::fs::create_dir_all(&spool).unwrap();
    let legacy = state.join("token.json");
    let bytes = br#"{"type":"authorized_user","client_id":"legacy","client_secret":"legacy-secret","refresh_token":"legacy-refresh"}"#;
    std::fs::write(&legacy, bytes).unwrap();
    let pending = Drive {
        endpoints: Endpoints::default(),
        token_path: token_path(&archive),
        spool: spool.clone(),
    }
    .keep_revision("1AbC", Some(revision("r1")), b"retained pending body")
    .unwrap()
    .body;
    let paths = crate::origin::CollectorPaths {
        home: dir.path(),
        archive: &archive,
        spool: &spool,
    };
    assert!(COLLECTOR.setting.is_none());
    let messages = (COLLECTOR.sync)(&paths, &std::sync::atomic::AtomicBool::new(false));
    assert!(
        messages.is_empty(),
        "not signing in is not a collector fault: {messages:?}"
    );
    assert!(needs_authorization(&archive));
    (COLLECTOR.advance.unwrap())(&paths, &mut |_, _, _| {
        panic!("unauthorized collector must not publish pending bytes")
    })
    .unwrap();
    assert_eq!(std::fs::read(legacy).unwrap(), bytes);
    assert_eq!(std::fs::read(pending).unwrap(), b"retained pending body");
    assert!(!token_path(&archive).exists());
}

#[test]
fn invalid_present_credentials_are_reported_instead_of_treated_as_unauthorized() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(token_path(dir.path()), b"not a credential record").unwrap();
    assert!(!needs_authorization(dir.path()));
    let paths = crate::origin::CollectorPaths {
        home: dir.path(),
        archive: dir.path(),
        spool: &dir.path().join("spool"),
    };
    let errors = (COLLECTOR.sync)(&paths, &std::sync::atomic::AtomicBool::new(false));
    assert_eq!(errors.len(), 1);
    assert!(
        errors[0].starts_with(&format!(
            "Drive token {}: ",
            token_path(dir.path()).display()
        )),
        "{errors:?}"
    );
}

#[test]
fn pending_names_decode_only_plain_hex_pairs() {
    assert_eq!(decode("3141"), Some("1A".to_owned()));
    assert_eq!(decode("+1"), None);
    assert_eq!(decode("3"), None);
    assert_eq!(decode("zz"), None);
}
