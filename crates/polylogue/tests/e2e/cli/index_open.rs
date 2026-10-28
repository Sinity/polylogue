use super::*;

#[test]
fn the_daemon_refuses_an_incompatible_index_without_deleting_its_rows() {
    for (change, message) in [
        (
            format!(
                "PRAGMA user_version={}",
                polylogue_store::schema::INDEX_VERSION + 1
            ),
            "index format",
        ),
        ("CREATE TABLE synthetic_schema(value)".into(), "no migration"),
        (
            "INSERT INTO meta(key,value) VALUES ('unsynced','1'),('build_boot_id','different-boot') ON CONFLICT(key) DO UPDATE SET value=excluded.value".into(),
            "another boot",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        let (archive, lock) =
            polylogue_store::Archive::open_or_create(&dir.path().join("archive")).unwrap();
        let conn = rusqlite::Connection::open(archive.index_db()).unwrap();
        conn.execute("INSERT INTO meta VALUES ('kept-marker','present')", [])
            .unwrap();
        conn.execute_batch(&change).unwrap();
        drop(conn);
        drop(lock);
        let mut command = crate::command(dir.path());
        command
            .arg("daemon")
            .env("POLYLOGUE_ARCHIVE", &archive.root)
            .env("POLYLOGUE_HOME", &home)
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
        assert_eq!(
            daemon.0.wait().unwrap().code(),
            Some(78),
            "{change}: {stderr}"
        );
        assert!(stderr.contains(message), "{change}: {stderr}");
        assert!(!archive.socket().exists());
        let conn = rusqlite::Connection::open(archive.index_db()).unwrap();
        assert_eq!(
            conn.query_row("SELECT value FROM meta WHERE key='kept-marker'", [], |r| {
                r.get::<_, String>(0)
            })
            .unwrap(),
            "present"
        );
    }
}
