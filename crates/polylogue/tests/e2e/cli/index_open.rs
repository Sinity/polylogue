use super::*;

#[test]
fn the_daemon_reopens_the_archive_after_sigterm_and_sigkill() {
    use rustix::process::{Pid, Signal, kill_process};
    use serde_json::json;
    use std::os::unix::process::ExitStatusExt;

    for signal in [Signal::TERM, Signal::KILL] {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let project = home.join(".claude/projects/-restart");
        std::fs::create_dir_all(&project).unwrap();
        let first_file = project.join("first.jsonl");
        std::fs::write(&first_file, line("u1", "first", "Plan the garden.")).unwrap();
        let env = crate::Fixture::new(
            &dir.path().join("archive"),
            &home,
            &dir.path().join("config"),
        );

        for (messages, expected_rows) in [
            (1, json!([{"id": "claude-code:first"}])),
            (
                2,
                json!([{"id": "claude-code:first"}, {"id": "claude-code:second"}]),
            ),
        ] {
            let mut daemon = start_daemon(&env);
            wait_status(&env, |status| status["store"]["messages"] == messages);

            let rows = query_rows(&cli(
                &env,
                &["--json", "sessions | sort by id", "--fields", "id"],
            ));
            assert_eq!(rows["rows"], expected_rows, "{signal:?}: {rows}");

            let out = cli(&env, &["--json", "session:claude-code:first", "read"]);
            assert!(out.status.success(), "{signal:?}: {out:?}");
            let page = read_page(json(&out));
            assert!(page.document.footer.next.is_none(), "{signal:?}: {page:?}");
            assert_eq!(
                stored_text::texts(&page),
                ["Plan the garden."],
                "{signal:?}"
            );

            if messages == 1 {
                let pid = Pid::from_raw(daemon.0.id().try_into().unwrap()).unwrap();
                kill_process(pid, signal).unwrap();
                let exit = daemon.0.wait().unwrap();
                if signal == Signal::TERM {
                    assert!(exit.success(), "{exit}");
                } else {
                    assert_eq!(exit.signal(), Some(libc::SIGKILL), "{exit}");
                }
                // The reopened daemon must read the first session from the archive.
                std::fs::remove_file(&first_file).unwrap();
                std::fs::write(
                    project.join("second.jsonl"),
                    line("u2", "second", "Plant the rosemary."),
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn the_daemon_rebuilds_an_index_from_another_schema_and_serves() {
    for (change, reason) in [
        (
            format!(
                "PRAGMA user_version={}",
                polylogue_store::schema::INDEX_VERSION + 1
            ),
            "format_version",
        ),
        (
            "CREATE TABLE synthetic_schema(value)".into(),
            "schema_differs",
        ),
        ("PRAGMA application_id=0".into(), "older_format"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let home = dir.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        let (archive, lock) = polylogue_store::Archive::open_or_create(&root).unwrap();
        // An installed index: a build installs its file when it ends.
        polylogue_store::Writer::start(&archive)
            .unwrap()
            .build(Vec::new())
            .unwrap();
        let conn = rusqlite::Connection::open(archive.index_db()).unwrap();
        conn.execute("INSERT INTO meta VALUES ('kept-marker','present')", [])
            .unwrap();
        conn.execute_batch(&change).unwrap();
        drop(conn);
        drop(lock);
        let env = crate::Fixture::new(&root, &home, &dir.path().join("config"));
        let _daemon = start_daemon(&env);
        let status = wait_status(&env, |s| {
            s["index"]["rebuilding"]["state"] == "no" && s["store"]["index_rebuild"].is_object()
        });
        let rebuild = &status["store"]["index_rebuild"];
        assert_eq!(rebuild["reason"]["kind"], reason, "{change}: {rebuild}");
        assert_ne!(rebuild["previous_schema"], rebuild["schema"], "{rebuild}");
        let conn = rusqlite::Connection::open(archive.index_db()).unwrap();
        let marker: Option<String> = conn
            .query_row("SELECT value FROM meta WHERE key='kept-marker'", [], |r| {
                r.get(0)
            })
            .ok();
        assert_eq!(marker, None, "{change}");
    }
}

#[test]
fn the_daemon_refuses_an_archive_in_another_format_and_leaves_it_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let (archive, lock) =
        polylogue_store::Archive::open_or_create(&dir.path().join("archive")).unwrap();
    drop(lock);
    let conn = rusqlite::Connection::open(archive.archive_db()).unwrap();
    conn.execute_batch("DROP TABLE revision").unwrap();
    conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))
        .unwrap();
    drop(conn);
    let before = std::fs::read(archive.archive_db()).unwrap();
    let mut command = crate::command(dir.path());
    command
        .arg("daemon")
        .env("XDG_CONFIG_HOME", dir.path().join("config"))
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut daemon = Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap());
    let mut stderr = String::new();
    daemon
        .0
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert_eq!(daemon.0.wait().unwrap().code(), Some(78), "{stderr}");
    assert!(
        stderr.contains(&format!(
            "archive.db at version {} is not in this Polylogue build's format",
            polylogue_store::schema::ARCHIVE_VERSION
        )),
        "{stderr}"
    );
    assert!(!stderr.contains("serving"), "{stderr}");
    assert!(!archive.socket().exists());
    assert_eq!(std::fs::read(archive.archive_db()).unwrap(), before);
}
