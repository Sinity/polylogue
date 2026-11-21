use super::*;
use std::process::{Command, Stdio};

#[test]
fn writer_stop_child() {
    if std::env::var_os("POLYLOGUE_SQL_STOP_CHILD").is_none() {
        return;
    }
    crate::timing::set_workers_enabled(true);
    let mut archive = test_archive();
    archive
        .store_reading(
            batch(
                "sql-stop",
                Route::Live,
                0,
                vec![row(
                    0,
                    text_message(
                        "sql-stop-message",
                        "sql-stop-session",
                        Role::User,
                        "SyntheticPayloadNeverInSqlEvents",
                        None,
                    ),
                )],
                &["sql-stop-session"],
            ),
            Some(&SOURCE_RECORD),
        )
        .unwrap();
    assert!(archive.writer.flush().unwrap().errors.is_empty());
    archive.writer.stop();
}
#[test]
fn normal_writer_stop_emits_final_sql_work_before_process_exit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("child.log");
    let output = std::fs::OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(&path)
        .unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "writer::tests::sql_attribution::writer_stop_child",
            "--nocapture",
        ])
        .env("POLYLOGUE_SQL_STOP_CHILD", "1")
        .stdout(Stdio::from(output.try_clone().unwrap()))
        .stderr(Stdio::from(output))
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if std::time::Instant::now() > deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!(
                "writer stop child did not end; {}",
                std::fs::read_to_string(&path).unwrap()
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let log = std::fs::read_to_string(path).unwrap();
    assert!(status.success(), "{log}");
    let records: Vec<serde_json::Value> = log
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let final_report = records
        .iter()
        .filter_map(|r| r.get("writer_sql_profile"))
        .find(|r| r["final"] == true)
        .unwrap_or_else(|| panic!("no final SQL work: {log}"));
    assert_eq!(final_report["phases"]["row_apply"]["calls"], 1);
    assert_eq!(final_report["phases"]["row_commit"]["calls"], 1);
    assert_eq!(final_report["statement_time_is_exclusive"], false);
    assert!(!final_report["statements"].as_array().unwrap().is_empty());
    for sql in records.iter().filter_map(|r| r.get("writer_sql_statement")) {
        assert_eq!(sql["expanded"], false);
        assert!(
            !sql["sql"]
                .as_str()
                .unwrap()
                .contains("SyntheticPayloadNeverInSqlEvents")
        );
    }
}
