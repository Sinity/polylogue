use super::*;

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
    ] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let home = dir.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        let (archive, lock) = polylogue_store::Archive::open_or_create(&root).unwrap();
        let conn = rusqlite::Connection::open(archive.index_db()).unwrap();
        conn.execute("INSERT INTO meta VALUES ('kept-marker','present')", [])
            .unwrap();
        conn.execute_batch(&change).unwrap();
        drop(conn);
        drop(lock);
        let env = crate::Fixture::new(&root, &home, &dir.path().join("config"));
        let _daemon = start_daemon(&env);
        let status = wait_status(&env, |s| s["store"]["index_rebuild"]["building"] == false);
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
    assert!(stderr.contains("no migration"), "{stderr}");
    assert!(!archive.socket().exists());
    assert_eq!(std::fs::read(archive.archive_db()).unwrap(), before);
}
