use super::*;

fn history(dir: &Path) -> PathBuf {
    let path = dir.join("history.db");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE TABLE runs (id INTEGER PRIMARY KEY, project TEXT NOT NULL, checkout TEXT NOT NULL,
            command TEXT NOT NULL, args TEXT NOT NULL, agentctl_job TEXT, started_ms INTEGER NOT NULL,
            status TEXT NOT NULL, failure TEXT)",
    )
    .unwrap();
    path
}

fn run(path: &Path, checkout: &str, command: &str, args: &str, status: &str, at: i64) {
    Connection::open(path)
        .unwrap()
        .execute(
            "INSERT INTO runs(project, checkout, command, args, agentctl_job, started_ms, status, failure)
             VALUES ('polylogue', ?1, ?2, ?3, 'job-' || ?5, ?5, ?4, CASE WHEN ?4 = 'failed' THEN 'tests failed' END)",
            params![checkout, command, args, status, at],
        )
        .unwrap();
}

#[test]
fn a_failure_stands_until_the_same_run_or_a_whole_check_passes_in_its_checkout() {
    let dir = tempfile::tempdir().unwrap();
    let path = history(dir.path());
    let w = "/w/one";
    run(&path, w, "test", r#"["-p","store"]"#, "failed", 10);
    run(&path, w, "test", r#"["-p","store"]"#, "failed", 20);
    run(&path, w, "test", r#"["-p","query"]"#, "failed", 30);
    run(&path, w, "test", r#"["-p","query"]"#, "passed", 40);
    run(&path, w, "check", "[]", "failed", 50);
    // Another checkout's pass changes nothing here.
    run(&path, "/w/two", "check", "[]", "passed", 60);
    run(&path, "/w/oneother", "check", "[]", "failed", 70);
    let failed = failed_since(&path, &[w.into()], 0).unwrap();
    let checks: Vec<(&str, u64)> = failed.iter().map(|f| (f.check.as_str(), f.times)).collect();
    assert_eq!(
        checks,
        [("polylogue check", 1), ("polylogue test -p store", 2)],
        "{failed:?}"
    );
    assert_eq!(failed[1].job.as_deref(), Some("job-20"));
    // A whole check passing later clears every failure in its checkout.
    run(&path, w, "check", "[]", "passed", 80);
    assert!(failed_since(&path, &[w.into()], 0).unwrap().is_empty());
    assert!(
        failed_since(&dir.path().join("absent.db"), &[w.into()], 0)
            .unwrap()
            .is_empty()
    );
}
