use super::fake::FakeDrive;
use super::*;

/// A prompt whose one chunk says `text`.
fn body(text: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "runSettings": {},
        "chunkedPrompt": {"chunks": [{"role": "user", "text": text}]}
    }))
    .unwrap()
}

/// A Drive whose folder holds the prompt `1AbC` with `revisions` (ID and
/// modification time, in Drive's list order), each revision's chunk saying
/// its ID.
fn drive(revisions: &[(&str, &str)]) -> FakeDrive {
    let drive = FakeDrive::start();
    drive.create("1AbC", "Electrons", "2026-05-17T12:00:00Z");
    for (id, at) in revisions {
        drive.save("1AbC", id, at, &body(id));
    }
    drive
}

fn revision(id: &str) -> Accepted {
    Accepted {
        revision: id.to_owned(),
        modified: None,
    }
}

fn collector(drive: &FakeDrive, dir: &Path) -> Drive {
    let token_path = dir.join("token.json");
    std::fs::write(
        &token_path,
        r#"{"client_id":"c","client_secret":"s","refresh_token":"r","token":"old"}"#,
    )
    .unwrap();
    Drive {
        endpoints: drive.endpoints(),
        token_path,
        spool: dir.join("spool"),
    }
}

/// A collector whose requests fail: for tests that never reach Drive.
fn offline(dir: &Path) -> Drive {
    Drive {
        endpoints: Endpoints {
            api: "http://127.0.0.1:1".into(),
            token: "http://127.0.0.1:1".into(),
        },
        token_path: dir.join("token.json"),
        spool: dir.join("spool"),
    }
}

/// The text of the prompt's first chunk as published in the spool.
fn published(c: &Drive, id: &str) -> String {
    let prompt: Json =
        serde_json::from_slice(&std::fs::read(c.spool.join(format!("{id}.json"))).unwrap())
            .unwrap();
    prompt
        .pointer("/chunkedPrompt/chunks/0/text")
        .and_then(Json::as_str)
        .unwrap()
        .to_owned()
}

/// The listing's entries by ID.
fn listing(c: &Drive) -> BTreeMap<String, Json> {
    c.load_listing().unwrap()
}

/// Whether a request listed the folder or looked it up.
fn relisted(paths: &[String]) -> bool {
    paths.iter().any(|path| path.starts_with("/drive/files?"))
}

#[test]
fn revision_survives_repeated_sync_and_restart_until_archive_acceptance() {
    let drive = drive(&[
        ("r2", "2026-05-17T12:30:00Z"),
        ("r1", "2026-05-17T12:20:00Z"),
    ]);
    drive.page_size(1);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    let prompt = c.spool.join("1AbC.json");

    let first = c.sync().unwrap();
    assert_eq!(first.revisions_written, 1);
    assert_eq!(first.pending, 2);
    assert_eq!(published(&c, "1AbC"), "r1");
    assert_eq!(listing(&c)["1AbC"]["name"], "Electrons");

    let original = std::fs::read(&prompt).unwrap();
    let kept = c.pending_revisions().unwrap().remove("1AbC").unwrap();
    assert_eq!(std::fs::read(&kept.body).unwrap(), original);
    assert!(c.load_state().unwrap().accepted.is_empty());
    assert_eq!(c.sync().unwrap().revisions_written, 0);
    assert_eq!(std::fs::read(&prompt).unwrap(), original);
    let c = collector(&drive, dir.path());
    c.advance(&|_, _, _| Ok(false)).unwrap();
    let DriveError::Io(error) = c
        .advance(&|_, _, _| Err("archive rejected".into()))
        .unwrap_err()
    else {
        panic!("expected the archive error");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), "archive rejected");
    assert_eq!(c.sync().unwrap().revisions_written, 0);
    assert_eq!(std::fs::read(&prompt).unwrap(), original);
    assert_eq!(std::fs::read(&kept.body).unwrap(), original);
    assert!(c.load_state().unwrap().accepted.is_empty());
    let seen = drive.seen();
    assert_eq!(
        seen.iter()
            .filter(|path| path.starts_with("/drive/files/1AbC/revisions/r1?"))
            .count(),
        1
    );
    assert!(seen.iter().any(|path| path.contains("pageToken=1")));

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
    assert_eq!(published(&c, "1AbC"), "r2");

    c.advance(&|_, _, _| Ok(true)).unwrap();
    let third = c.sync().unwrap();
    assert_eq!(third.revisions_written, 0);
}

#[test]
fn drive_follows_the_changes_feed_from_the_stored_token() {
    let drive = drive(&[("r1", "2026-05-17T12:20:00Z")]);
    drive.page_size(2);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    let first = c.sync().unwrap();
    assert_eq!((first.prompts, first.revisions_written), (1, 1));
    assert!(relisted(&drive.seen()));
    // The token was taken before the listing: the feed starts after the
    // two changes that created the prompt and saved its first revision.
    assert_eq!(c.load_state().unwrap().change_token.as_deref(), Some("2"));
    c.advance(&|_, _, _| Ok(true)).unwrap();

    drive.save("1AbC", "r2", "2026-05-17T12:30:00Z", &body("r2"));
    drive.create("2Def", "Muons", "2026-05-17T13:00:00Z");
    drive.save("2Def", "m1", "2026-05-17T13:00:00Z", &body("m1"));
    drive.create_elsewhere("3Ghi", "Shopping list", "2026-05-17T13:10:00Z");
    let before = drive.seen().len();
    // A restart keeps the token: it is in the collector's state.
    let c = collector(&drive, dir.path());
    let second = c.sync().unwrap();
    let requests = drive.seen()[before..].to_vec();
    assert!(!relisted(&requests), "{requests:?}");
    assert!(
        requests
            .iter()
            .any(|path| path.starts_with("/drive/changes?pageToken=2&")),
        "{requests:?}"
    );
    // Four changes in pages of two.
    assert!(
        requests
            .iter()
            .any(|path| path.starts_with("/drive/changes?pageToken=4&"))
    );
    assert_eq!((second.prompts, second.revisions_written), (2, 2));
    let entries = listing(&c);
    assert_eq!(entries.keys().collect::<Vec<_>>(), ["1AbC", "2Def"]);
    assert_eq!(entries["1AbC"]["headRevisionId"], "r2");
    assert_eq!(entries["2Def"]["name"], "Muons");
    assert!(entries["2Def"].get("parents").is_none());
    assert_eq!(published(&c, "1AbC"), "r2");
    assert_eq!(published(&c, "2Def"), "m1");
    assert_eq!(c.load_state().unwrap().change_token.as_deref(), Some("6"));
}

#[test]
fn a_deleted_prompt_stays_in_the_listing_marked_removed() {
    let drive = drive(&[("r1", "2026-05-17T12:20:00Z")]);
    drive.create("2Def", "Muons", "2026-05-17T13:00:00Z");
    drive.save("2Def", "m1", "2026-05-17T13:00:00Z", &body("m1"));
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    c.sync().unwrap();
    c.advance(&|_, _, _| Ok(true)).unwrap();
    drive.save("1AbC", "r2", "2026-05-17T12:30:00Z", &body("r2"));
    drive.delete("1AbC");
    let before = drive.seen().len();
    let report = c.sync().unwrap();
    assert_eq!((report.prompts, report.removed), (1, 1));
    assert_eq!(report.revisions_written, 0);
    assert!(
        !drive.seen()[before..]
            .iter()
            .any(|path| path.starts_with("/drive/files/1AbC")),
        "a removed prompt's revisions are not asked for"
    );
    let entries = listing(&c);
    assert_eq!(entries["1AbC"]["removed"], true);
    assert_eq!(entries["1AbC"]["name"], "Electrons");
    assert!(entries["2Def"].get("removed").is_none());
    // The revision the archive holds stays in the spool.
    assert_eq!(published(&c, "1AbC"), "r1");

    // A first sync after an older build's listing marks what the folder
    // no longer holds.
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    std::fs::create_dir_all(&c.spool).unwrap();
    std::fs::write(
        c.spool.join(super::super::LISTING),
        br#"{"files":[{"id":"1AbC","name":"Electrons"}]}"#,
    )
    .unwrap();
    c.sync().unwrap();
    let entries = listing(&c);
    assert_eq!(entries["1AbC"]["removed"], true);
    assert!(entries["2Def"].get("removed").is_none());
}

#[test]
fn revisions_order_by_time_whatever_the_fraction_width() {
    // As text, `12:00:00.5Z` sorts before `12:00:00Z`; as instants it is
    // half a second after it.
    let drive = drive(&[
        ("c", "2026-05-17T12:00:01.25Z"),
        ("a", "2026-05-17T12:00:00Z"),
        ("b", "2026-05-17T12:00:00.5Z"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    c.sync().unwrap();
    assert_eq!(published(&c, "1AbC"), "a");
    c.advance(&|_, _, _| Ok(true)).unwrap();
    // Drive prunes the accepted revision: the walk continues at the first
    // revision after it in time.
    drive.prune("1AbC", |id| id != "a");
    assert_eq!(walk(&c), ["b", "c"]);
}

#[test]
fn a_revisions_reply_without_revisions_is_unreadable() {
    let drive = drive(&[("r1", "2026-05-17T12:20:00Z")]);
    drive.answer_when(
        |target| target.starts_with("/drive/files/1AbC/revisions?"),
        r#"{"nextPageToken":null}"#,
    );
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    assert!(matches!(
        c.sync(),
        Err(DriveError::Unreadable(url, reason))
            if url.ends_with("/files/1AbC/revisions") && reason == "no `revisions` array"
    ));
    assert!(!c.spool.join("1AbC.json").exists());
    // The failed sync saved no token: the next one starts over.
    assert_eq!(c.load_state().unwrap().change_token, None);
}

#[test]
fn an_unreadable_revision_time_is_refused() {
    let drive = drive(&[("r1", "17 May 2026")]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    assert!(matches!(c.sync(), Err(DriveError::Unreadable(..))));
    assert!(!c.spool.join("1AbC.json").exists());
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
    let drive = drive(&[("r1", "2026-05-17T12:20:00Z")]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
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
    let c = offline(dir.path());
    let kept = c
        .keep_revision("1AbC", Some(revision("r1")), b"original")
        .unwrap();
    c.publish(&kept).unwrap();
    c.save_state(&Cursor {
        accepted: BTreeMap::from([("1AbC".into(), "r1".into())]),
        ..Cursor::default()
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
    let drive = drive(&[
        ("r1", "2026-05-17T12:20:00Z"),
        ("r2", "2026-05-17T12:30:00Z"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
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
    assert_eq!(published(&c, "1AbC"), "r1");
    c.advance(&|_, _, _| Ok(true)).unwrap();
    assert_eq!(c.sync().unwrap().revisions_written, 1);
    assert_eq!(published(&c, "1AbC"), "r2");
}

#[test]
fn corrupt_or_newer_revision_state_is_preserved_and_refused() {
    let drive = drive(&[]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
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

#[test]
fn a_listing_reply_without_files_keeps_the_saved_listing() {
    let drive = drive(&[("r1", "2026-05-17T12:20:00Z")]);
    drive.answer_when(
        |target| target.starts_with("/drive/files?") && target.contains("nextPageToken"),
        "{}",
    );
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    std::fs::create_dir_all(&c.spool).unwrap();
    let saved = br#"{"files":[{"id":"1AbC"}]}"#;
    std::fs::write(c.spool.join("files.json"), saved).unwrap();
    assert!(matches!(c.sync(), Err(DriveError::Unreadable(..))));
    assert_eq!(std::fs::read(c.spool.join("files.json")).unwrap(), saved);
}

#[test]
fn a_missing_folder_is_reported() {
    let drive = FakeDrive::start();
    drive.without_folder();
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    assert!(matches!(c.sync(), Err(DriveError::NoFolder)));
}

#[test]
fn a_prompt_id_cannot_leave_the_spool_or_name_the_listing() {
    let dir = tempfile::tempdir().unwrap();
    let c = offline(dir.path());
    for (id, valid) in [
        ("", "a"),
        ("../escape", "..-escape"),
        ("a/b", "a-b"),
        ("a\\b", "a-b"),
        (".", ".a"),
        ("..", "..a"),
        ("files", "file"),
    ] {
        let kept = c.keep_revision(id, Some(revision("r1")), b"x");
        let Err(keep_error) = kept else {
            panic!("kept a revision for {id:?}");
        };
        for error in [c.prompt_path(id).unwrap_err(), keep_error] {
            let DriveError::Io(error) = error else {
                panic!("expected an invalid ID for {id:?}, got {error:?}");
            };
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData, "{id:?}");
            assert_eq!(error.to_string(), "invalid Drive prompt ID", "{id:?}");
        }
        assert_eq!(
            c.prompt_path(valid).unwrap(),
            c.spool.join(format!("{valid}.json"))
        );
        let pending = c.keep_revision(valid, Some(revision("r1")), b"x").unwrap();
        assert_eq!(pending.file, valid);
        assert_eq!(std::fs::read(pending.body).unwrap(), b"x");
    }
    assert!(!dir.path().join("escape.json").exists());
    assert!(!c.spool.join("files.json").exists());
}

/// Accepts every pending revision and syncs again, until no revision is
/// left; returns the revision texts in the order they were published.
fn walk(c: &Drive) -> Vec<String> {
    let mut texts = Vec::new();
    loop {
        c.advance(&|_, _, _| Ok(true)).unwrap();
        if c.sync().unwrap().revisions_written == 0 {
            return texts;
        }
        texts.push(published(c, "1AbC"));
    }
}

fn minute(n: usize) -> String {
    format!("2026-05-17T{:02}:{:02}:00Z", n / 60, n % 60)
}

#[test]
fn two_pages_of_revisions_all_captured() {
    let drive = FakeDrive::start();
    drive.create("1AbC", "Electrons", "2026-05-17T00:00:00Z");
    let expected: Vec<String> = (0..250).map(|n| format!("r{n:03}")).collect();
    for (n, id) in expected.iter().enumerate() {
        drive.save("1AbC", id, &minute(n), &body(id));
    }
    drive.page_size(100);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    let first = c.sync().unwrap();
    assert_eq!((first.revisions_written, first.pending), (1, 250));
    let mut texts = vec![published(&c, "1AbC")];
    texts.extend(walk(&c));
    assert_eq!(texts, expected);
    assert!(
        drive
            .seen()
            .iter()
            .any(|path| path.starts_with("/drive/files/1AbC/revisions?")
                && path.contains("pageToken=200"))
    );
}

#[test]
fn resume_after_pruned_revision_continues_from_next_newer() {
    let drive = drive(&[
        ("r1", "2026-05-17T12:10:00Z"),
        ("r2", "2026-05-17T12:20:00Z"),
        ("r3", "2026-05-17T12:30:00Z"),
        ("r4", "2026-05-17T12:40:00Z"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let c = collector(&drive, dir.path());
    c.sync().unwrap();
    c.advance(&|_, _, _| Ok(true)).unwrap();
    assert_eq!(c.sync().unwrap().revisions_written, 1);
    c.advance(&|_, _, _| Ok(true)).unwrap();
    assert_eq!(c.load_state().unwrap().revision("1AbC"), Some("r2"));
    // Drive prunes the accepted revision and one older than it.
    drive.prune("1AbC", |id| id != "r1" && id != "r2");
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
