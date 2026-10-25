//! End to end: the real binary as a daemon over a temporary archive and
//! home directory, driven through the CLI.

use std::io::{BufRead, BufReader};
#[cfg(target_os = "linux")]
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn start_daemon(env: &[(&str, &Path)]) -> Daemon {
    let mut command = Command::new(crate::BIN);
    command.arg("daemon").stderr(Stdio::piped());
    for (key, value) in env {
        command.env(key, value);
    }
    let mut daemon = Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap());
    let mut first = String::new();
    BufReader::new(daemon.0.stderr.as_mut().unwrap())
        .read_line(&mut first)
        .unwrap();
    assert!(first.contains("serving"), "{first}");
    daemon
}

/// Polls `status` until `ready` holds and the loop is idle, so whatever pass
/// made `ready` true has also derived from its rows. A state that never comes
/// is ended by the test runner's timeout.
fn wait_status(
    env: &[(&str, &Path)],
    ready: impl Fn(&serde_json::Value) -> bool,
) -> serde_json::Value {
    loop {
        let out = cli(env, &["--json", "status"]);
        assert!(out.status.success(), "{out:?}");
        let status = json(&out);
        if status["busy"] == false && ready(&status) {
            return status;
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
fn killing_the_test_parent_stops_its_real_daemon() {
    use std::os::unix::process::ExitStatusExt;

    let dir = tempfile::tempdir().unwrap();
    let mut owner = Daemon(
        polylogue_testkit::spawn_daemon(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "cli::daemon_parent_fixture",
                    "--ignored",
                    "--nocapture",
                ])
                .env("POLYLOGUE_DAEMON_PARENT_FIXTURE", dir.path())
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped()),
        )
        .unwrap(),
    );
    let mut stdout = BufReader::new(owner.0.stdout.take().unwrap());
    let mut line = String::new();
    loop {
        assert_ne!(stdout.read_line(&mut line).unwrap(), 0, "{line}");
        if line.contains("daemon_pid:") {
            break;
        }
        line.clear();
    }
    let daemon_pid: u64 = line
        .split_once("daemon_pid:")
        .unwrap()
        .1
        .trim()
        .parse()
        .unwrap();
    let mut stderr = BufReader::new(owner.0.stderr.take().unwrap());
    loop {
        line.clear();
        assert_ne!(
            stderr.read_line(&mut line).unwrap(),
            0,
            "no daemon readiness"
        );
        if line.contains("serving") {
            break;
        }
    }
    let running = Command::new(crate::BIN)
        .args(["--json", "status"])
        .env("POLYLOGUE_ARCHIVE", dir.path().join("archive"))
        .env("POLYLOGUE_HOME", dir.path().join("home"))
        .env("XDG_CONFIG_HOME", dir.path().join("config"))
        .output()
        .unwrap();
    assert!(running.status.success(), "{running:?}");
    assert_eq!(json(&running)["pid"].as_u64(), Some(daemon_pid));
    assert!(matches!(
        polylogue_store::Archive::open_or_create(&dir.path().join("archive")),
        Err(polylogue_store::StoreError::Locked { .. })
    ));
    let exit = polylogue_testkit::ProcessExit::watch(daemon_pid.try_into().unwrap()).unwrap();
    owner.0.kill().unwrap();
    assert_eq!(owner.0.wait().unwrap().signal(), Some(9));
    // Stderr can close before other descriptors during process teardown.
    // Wait for process exit before checking that the archive lock is released.
    exit.wait().unwrap();
    let mut remaining = String::new();
    stderr.read_to_string(&mut remaining).unwrap();
    let (_archive, _lock) = polylogue_store::Archive::open_or_create(&dir.path().join("archive"))
        .expect("the killed owner's daemon must release its archive lock");
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "subprocess fixture for killing_the_test_parent_stops_its_real_daemon"]
fn daemon_parent_fixture() {
    let Some(dir) = std::env::var_os("POLYLOGUE_DAEMON_PARENT_FIXTURE") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let daemon = Daemon(
        polylogue_testkit::spawn_daemon(
            Command::new(crate::BIN)
                .arg("daemon")
                .env("POLYLOGUE_ARCHIVE", dir.join("archive"))
                .env("POLYLOGUE_HOME", dir.join("home"))
                .env("XDG_CONFIG_HOME", dir.join("config"))
                .stderr(Stdio::inherit()),
        )
        .unwrap(),
    );
    println!("daemon_pid:{}", daemon.0.id());
    std::io::stdout().flush().unwrap();
    let mut end = String::new();
    std::io::stdin().read_line(&mut end).unwrap();
    drop(daemon);
}

#[cfg(target_os = "linux")]
fn io_counts(pid: u64) -> (u64, u64) {
    let text = std::fs::read_to_string(format!("/proc/{pid}/io")).unwrap();
    let value = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(name))
            .unwrap()
            .trim()
            .parse::<u64>()
            .unwrap()
    };
    (value("rchar:"), value("read_bytes:"))
}

#[test]
fn idle_passes_do_not_reread_complete_or_waiting_input_contents() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let project = home.join(".claude/projects/-idle");
    std::fs::create_dir_all(&project).unwrap();
    let complete = line("u1", "idle", &"synthetic content ".repeat(4096));
    std::fs::write(
        project.join("complete.jsonl"),
        line("c1", "complete", &"another complete input ".repeat(4096)),
    )
    .unwrap();
    let partial = line("u2", "idle", "the record is still being written");
    std::fs::write(
        project.join("idle.jsonl"),
        format!("{complete}{}", &partial[..20]),
    )
    .unwrap();
    let env = [
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("POLYLOGUE_HOME", home.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let first = start_daemon(&env);
    wait_status(&env, |s| s["last_pass"]["inputs_seen"].as_u64() == Some(2));
    drop(first);
    let daemon = start_daemon(&env);
    let mut status = wait_status(&env, |s| s["last_pass"]["inputs_seen"].as_u64() == Some(2));
    assert_eq!(status["store"]["messages"], 2);
    assert_eq!(status["store"]["parse_states"]["waiting"], 1);
    assert_eq!(status["last_pass"]["bytes_read"], 0, "{status}");
    for generation in 0..3 {
        // An irrelevant path under the watched tree wakes a real pass without
        // changing a source. The pass reply, not a sleep, delimits observation.
        let previous = status["passes"].as_u64().unwrap();
        #[cfg(target_os = "linux")]
        let before = io_counts(u64::from(daemon.0.id()));
        std::fs::write(project.join(".idle-wakeup"), generation.to_string()).unwrap();
        status = wait_status(&env, |s| s["passes"].as_u64() > Some(previous));
        assert_eq!(status["last_pass"]["bytes_read"], 0, "{status}");
        assert_eq!(status["last_pass"]["bytes_stored"], 0, "{status}");
        assert_eq!(status["last_pass"]["inputs_changed"], 0, "{status}");
        #[cfg(target_os = "linux")]
        {
            let after = io_counts(u64::from(daemon.0.id()));
            eprintln!(
                "idle pass {generation}: /proc/io rchar delta {}, physical read_bytes delta {}; input-content bytes_read 0",
                after.0 - before.0,
                after.1 - before.1
            );
        }
    }
    {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(project.join("idle.jsonl"))
            .unwrap();
        std::io::Write::write_all(&mut file, &partial.as_bytes()[20..]).unwrap();
    }
    status = wait_status(&env, |s| {
        s["store"]["messages"].as_u64() == Some(3)
            && s["last_pass"]["bytes_stored"]
                .as_u64()
                .is_some_and(|n| n > 0)
    });
    assert_eq!(status["store"]["parse_states"]["done"], 2);
}

fn cli(env: &[(&str, &Path)], args: &[&str]) -> Output {
    let mut c = Command::new(crate::BIN);
    for (k, v) in env {
        c.env(k, v);
    }
    c.args(args).output().unwrap()
}

fn json(out: &Output) -> serde_json::Value {
    serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stdout)))
}

fn line(uuid: &str, session: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({
            "type": "user", "uuid": uuid, "sessionId": session,
            "timestamp": "2026-03-01T10:00:00Z",
            "message": {"role": "user", "content": text}
        })
    )
}

#[test]
fn status_list_read_and_import_through_the_daemon_then_daemon_required() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let project = home.join(".claude/projects/-p");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("live.jsonl"),
        line("u1", "live", "from a live file"),
    )
    .unwrap();
    let export = dir.path().join("export.jsonl");
    std::fs::write(&export, line("x1", "exported", "from an import")).unwrap();
    let env = [
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("POLYLOGUE_HOME", home.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];

    let daemon = start_daemon(&env);

    // The live file is picked up by the loop's first pass.
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(1));

    let out = cli(&env, &["--json", "import", export.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(json(&out)["imported"].as_array().unwrap().len(), 1);

    let out = cli(&env, &["--json", "list"]);
    let ids: Vec<String> = json(&out)["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_owned())
        .collect();
    assert!(ids.contains(&"claude-code:live".to_owned()), "{ids:?}");
    assert!(ids.contains(&"claude-code:exported".to_owned()), "{ids:?}");

    let out = cli(&env, &["--json", "read", "claude-code:exported"]);
    let read = json(&out);
    assert_eq!(read["messages"][0]["blocks"][0]["text"], "from an import");

    let out = cli(&env, &["read", "claude-code:live"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains("from a live file"));

    drop(daemon);
    let out = cli(&env, &["--json", "status"]);
    assert_eq!(out.status.code(), Some(6));
    assert_eq!(json(&out)["error"]["kind"], "daemon_required");
}

#[test]
fn ai_studio_import_and_drive_title_read_identically_after_a_rebuild_from_stored_bytes() {
    let w = polylogue_testkit::ClaudeHome::new();
    let config = w.dir.path().join("config");
    let dir = w.config(8).spool_dir("ai-studio");
    std::fs::create_dir_all(&dir).unwrap();
    let listing = dir.join("files.json");
    std::fs::write(&listing,serde_json::to_vec(&serde_json::json!({"files":[{"id":"drive-id","name":"Electrons","createdTime":"2026-05-17T12:00:00Z"}]})).unwrap()).unwrap();
    let prompt = w.dir.path().join("drive-id.json");
    std::fs::write(&prompt,serde_json::to_vec(&serde_json::json!({
        "chunkedPrompt":{"chunks":[{"role":"user","text":"What is an electron?"},{"role":"model","text":"A lepton.","tokenCount":3}]},
        "runSettings":{"model":"models/example"},"systemInstruction":{"text":"Be brief."}
    })).unwrap()).unwrap();
    let env = [
        ("POLYLOGUE_ARCHIVE", w.root.as_path()),
        ("POLYLOGUE_HOME", w.home.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["sessions"].as_u64() == Some(1));
    let imported = cli(&env, &["--json", "import", prompt.to_str().unwrap()]);
    assert!(
        imported.status.success(),
        "{}",
        String::from_utf8_lossy(&imported.stderr)
    );
    let read = cli(&env, &["--json", "read", "ai-studio:drive-id"]);
    assert!(
        read.status.success(),
        "{}",
        String::from_utf8_lossy(&read.stderr)
    );
    let before = json(&read);
    assert_eq!(before["session"]["title"]["text"], "Electrons");
    assert_eq!(before["messages"][0]["blocks"][0]["text"], "Be brief.");
    assert_eq!(
        before["messages"][1]["blocks"][0]["text"],
        "What is an electron?"
    );
    assert_eq!(before["messages"][2]["blocks"][0]["text"], "A lepton.");
    assert_eq!(before["messages"][2]["model"], "example");
    assert_eq!(before["messages"][2]["usage"]["output"], 3);
    drop(daemon);
    std::fs::remove_file(prompt).unwrap();
    std::fs::remove_file(listing).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let daemon = start_daemon(&env);
    // The title comes from the Drive listing, a separate input that the
    // same pass parses after the messages; an idle loop has done both.
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(3));
    let after = cli(&env, &["--json", "read", "ai-studio:drive-id"]);
    assert!(
        after.status.success(),
        "{}",
        String::from_utf8_lossy(&after.stderr)
    );
    assert_eq!(json(&after), before);
    drop(daemon);
}

#[test]
fn the_hook_spools_where_the_daemon_reads_never_outside_polylogue_home() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let elsewhere = dir.path().join("xdg");
    let mut child = Command::new(crate::BIN)
        .args(["hook", "Stop"])
        .env("POLYLOGUE_HOME", &home)
        .env("XDG_STATE_HOME", &elsewhere)
        .env("POLYLOGUE_ARCHIVE", dir.path().join("archive"))
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), br#"{"session_id": "s1"}"#).unwrap();
    assert!(child.wait().unwrap().success());
    let spool = home.join(".local/state/polylogue/hooks");
    let spool_files: Vec<_> = std::fs::read_dir(&spool)
        .unwrap_or_else(|e| panic!("{}: {e}", spool.display()))
        .collect();
    assert_eq!(spool_files.len(), 1);
    assert!(!elsewhere.exists(), "the tools' own state is not touched");
}

mod structured_json;

mod read_pages;

mod read_masking;

mod search_snippets;

mod hook_errors;

mod grok_export;

mod codex_reports;

mod parents;

mod hermes_state;

mod claude_account_text;
mod hermes_events;

#[cfg(target_os = "linux")]
mod attachments;

#[cfg(target_os = "linux")]
mod read_bench;

#[cfg(target_os = "linux")]
#[test]
fn the_daemon_refuses_an_archive_with_another_schema_and_exits_with_its_code() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("archive");
    let home = dir.path().join("home");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let (opened, lock) = polylogue_store::Archive::open_or_create(&archive).unwrap();
    drop(lock);
    // As an archive written before the `excised` table was declared.
    rusqlite::Connection::open(opened.archive_db())
        .unwrap()
        .execute_batch("DROP TABLE excised")
        .unwrap();
    let mut command = Command::new(crate::BIN);
    command
        .arg("daemon")
        .env("POLYLOGUE_ARCHIVE", &archive)
        .env("POLYLOGUE_HOME", &home)
        .env("XDG_CONFIG_HOME", &config)
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
    let status = daemon.0.wait().unwrap();
    assert_eq!(status.code(), Some(78), "{stderr}");
    assert!(
        stderr.contains("archive.db has another schema")
            && stderr.contains("export")
            && stderr.contains("import"),
        "{stderr}"
    );
    assert!(!stderr.contains("serving"), "{stderr}");
    assert!(!archive.join("daemon.sock").exists());
}
