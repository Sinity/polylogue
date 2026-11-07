//! End to end: the real binary as a daemon over a temporary archive and
//! home directory, driven through the CLI.

use std::io::{BufRead, BufReader};
#[cfg(target_os = "linux")]
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

pub(super) struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub(super) fn start_daemon(env: &crate::Fixture) -> Daemon {
    let mut command = command(env);
    command.arg("daemon").stderr(Stdio::piped());
    let mut daemon = Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap());
    let mut first = String::new();
    BufReader::new(daemon.0.stderr.as_mut().unwrap())
        .read_line(&mut first)
        .unwrap();
    assert!(first.contains("serving"), "{first}");
    daemon
}

/// Polls `status` until `ready` holds and the loop is idle, so whatever pass
/// made `ready` true has also derived from its rows. Polls pause briefly so
/// the CLI processes do not compete with the daemon for the host; a state
/// that has not come within the deadline fails with the last status seen.
pub(super) fn wait_status(
    env: &crate::Fixture,
    ready: impl Fn(&serde_json::Value) -> bool,
) -> serde_json::Value {
    const DEADLINE: Duration = Duration::from_secs(60);
    const PAUSE: Duration = Duration::from_millis(25);
    let started = Instant::now();
    loop {
        // The poll lifts its deadline, so a loaded host delays the answer
        // instead of turning it into a timeout.
        let out = cli(env, &["--deadline", "none", "--json", "status"]);
        assert!(out.status.success(), "{out:?}");
        let status = json(&out);
        // The archive counts come once the daemon's first count ends.
        if status["busy"] == false && status["store"].is_object() && ready(&status) {
            return status;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "the awaited status did not come within {DEADLINE:?}; last status: {status}"
        );
        std::thread::sleep(PAUSE);
    }
}

#[cfg(target_os = "linux")]
#[test]
fn killing_the_test_parent_stops_its_real_daemon() {
    use std::os::unix::process::ExitStatusExt;

    let dir = tempfile::tempdir().unwrap();
    let mut owner = Daemon(
        polylogue_testkit::spawn_daemon(
            crate::test_process(dir.path())
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
    let running = crate::command(dir.path())
        .args(["--json", "status"])
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
            crate::command(&dir)
                .arg("daemon")
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
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
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

fn command(env: &crate::Fixture) -> Command {
    env.command()
}

pub(super) fn cli(env: &crate::Fixture, args: &[&str]) -> Output {
    let mut c = command(env);
    c.args(args).output().unwrap()
}

fn json(out: &Output) -> serde_json::Value {
    let answer: command_surface::Answer<serde_json::Value> = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stdout)));
    answer
        .value()
        .cloned()
        .unwrap_or_else(|| serde_json::to_value(answer).unwrap())
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
    // Unrelated JSONL is recognized only in its preserved tool layout
    // (docs/capture.md, Import).
    let export = dir.path().join("exports/.claude/projects/-p/export.jsonl");
    std::fs::create_dir_all(export.parent().unwrap()).unwrap();
    std::fs::write(&export, line("x1", "exported", "from an import")).unwrap();
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());

    let daemon = start_daemon(&env);

    // The live file is picked up by the loop's first pass.
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(1));

    let out = cli(&env, &["--json", "import", export.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        json(&out)["imported"].as_array().unwrap().len(),
        1,
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );

    let out = cli(&env, &["--json", "list"]);
    let ids: Vec<String> = json(&out)["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_owned())
        .collect();
    assert!(ids.contains(&"claude-code:live".to_owned()), "{ids:?}");
    assert!(ids.contains(&"claude-code:exported".to_owned()), "{ids:?}");

    let out = cli(&env, &["--json", "read", "session:claude-code:exported"]);
    let read: polylogue_model::read::ReadPage = serde_json::from_value(json(&out)).unwrap();
    let mut imported_items = read_current::Items::default();
    imported_items.push(&read);
    assert!(
        read.document.footer.next.is_none(),
        "single short imported message must fit its root"
    );
    assert_eq!(imported_items.text(), "from an import");

    let out = cli(&env, &["read", "session:claude-code:live"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains("from a live file"));

    drop(daemon);
    let out = cli(&env, &["--json", "status"]);
    assert_eq!(out.status.code(), Some(6));
    assert_eq!(json(&out)["error"]["code"], "daemon_required");
}

#[test]
fn ai_studio_import_and_drive_title_read_identically_after_a_rebuild_from_stored_bytes() {
    use polylogue_model::read::{ItemKind, ReadHeader, ReadItem, ReadPage, ValueState};

    fn reading(env: &crate::Fixture) -> (ReadHeader, Vec<ReadItem>) {
        // The system instruction is a harness message, which a default read
        // collapses to an indicator (docs/reading.md, harness messages).
        let mut out = cli(
            env,
            &[
                "--json",
                "read",
                "session:ai-studio:drive-id",
                "--show",
                "harness:full",
            ],
        );
        let mut items = read_current::Items::default();
        let mut header = None;
        let mut continuations = std::collections::BTreeSet::new();
        loop {
            assert!(out.status.success(), "{out:?}");
            let page: ReadPage = serde_json::from_value(json(&out)).unwrap();
            if let Some(expected) = &header {
                assert_eq!(&page.document.header, expected);
            } else {
                header = Some(page.document.header.clone());
            }
            items.push(&page);
            let Some(next) = page.document.footer.next else {
                break;
            };
            assert!(
                continuations.insert(next.cli.clone()),
                "repeated continuation"
            );
            let stages = polylogue_model::command::effective_stages(&next.cli);
            assert_eq!(stages.len(), 1);
            assert_eq!(stages[0].program, "polylogue");
            let mut args = vec!["--json"];
            args.extend(stages[0].args.iter().map(String::as_str));
            out = cli(env, &args);
        }
        (header.unwrap(), items.into_values())
    }

    fn messages(env: &crate::Fixture, destination: &Path) -> Vec<polylogue_model::Message> {
        std::fs::create_dir(destination).unwrap();
        let out = cli(
            env,
            &[
                "--json",
                "read",
                "session:ai-studio:drive-id",
                "--format",
                "jsonl",
                "--to",
                destination.to_str().unwrap(),
            ],
        );
        assert!(out.status.success(), "{out:?}");
        let bytes = std::fs::read_to_string(destination.join("records.jsonl")).unwrap();
        let mut records = bytes
            .lines()
            .map(|line| serde_json::from_str::<polylogue_model::bundle::Record>(line).unwrap());
        let Some(polylogue_model::bundle::Record::Manifest(manifest)) = records.next() else {
            panic!("canonical bundle must start with its manifest");
        };
        manifest.validate().unwrap();
        records
            .filter_map(|record| match record {
                polylogue_model::bundle::Record::Message(row) => Some(row.message),
                _ => None,
            })
            .collect()
    }

    let w = polylogue_testkit::ClaudeHome::new();
    let config = w.dir.path().join("config");
    let env = crate::Fixture::new(w.root.as_path(), w.home.as_path(), config.as_path());
    // Read the actual child environment, whose XDG state root differs from
    // ClaudeHome's standalone engine fixture and its HOME fallback.
    let child = command(&env);
    let state = child
        .get_envs()
        .find_map(|(key, value)| {
            (key == "XDG_STATE_HOME").then(|| std::path::PathBuf::from(value.unwrap()))
        })
        .unwrap();
    let dir = state.join("polylogue/drive/ai-studio");
    std::fs::create_dir_all(&dir).unwrap();
    let listing = dir.join("files.json");
    let listing_bytes = serde_json::to_vec(&serde_json::json!({"files":[{"id":"drive-id","name":"Electrons","createdTime":"2026-05-17T12:00:00Z"}]})).unwrap();
    std::fs::write(&listing, &listing_bytes).unwrap();
    let prompt = w.dir.path().join("drive-id.json");
    let prompt_bytes = serde_json::to_vec(&serde_json::json!({
        "chunkedPrompt":{"chunks":[{"role":"user","text":"What is an electron?"},{"role":"model","text":"A lepton.","tokenCount":3}]},
        "runSettings":{"model":"models/example"},"systemInstruction":{"text":"Be brief."}
    })).unwrap();
    std::fs::write(&prompt, &prompt_bytes).unwrap();
    let daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["sessions"].as_u64() == Some(1));
    let imported = cli(&env, &["--json", "import", prompt.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(3));
    let before = reading(&env);
    assert_eq!(before.0.title, ValueState::known("Electrons".to_owned()));
    let texts: Vec<_> = before
        .1
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["Be brief.", "What is an electron?", "A lepton."]);
    let headers: Vec<_> = before
        .1
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::MessageHeader { header } => Some((&item.source.reference, header)),
            _ => None,
        })
        .collect();
    let expected_ids = [
        "ai-studio:8:drive-id#system:0:",
        "ai-studio:8:drive-id#position:1:0",
        "ai-studio:8:drive-id#position:1:1",
    ];
    assert_eq!(
        headers
            .iter()
            .map(|(reference, _)| reference.as_str())
            .collect::<Vec<_>>(),
        expected_ids.map(|id| format!("message:{id}"))
    );
    assert_eq!(headers[2].1.model, ValueState::known("example".to_owned()));
    // Usage belongs to canonical Message, rather than the presentation header.
    let before_messages = messages(&env, &w.dir.path().join("before-bundle"));
    assert_eq!(
        before_messages
            .iter()
            .map(|m| m.id.to_string())
            .collect::<Vec<_>>(),
        expected_ids
    );
    assert_eq!(before_messages[2].model.as_deref(), Some("example"));
    assert_eq!(before_messages[2].usage.output, Some(3));
    assert_eq!(std::fs::read(&prompt).unwrap(), prompt_bytes);
    assert_eq!(std::fs::read(&listing).unwrap(), listing_bytes);
    drop(daemon);
    std::fs::remove_file(&prompt).unwrap();
    std::fs::remove_file(&listing).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let daemon = start_daemon(&env);
    // The title's separate Drive input and prompt must both reparse from
    // retained source bytes, after their provider files have disappeared.
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(3));
    let after = reading(&env);
    assert_eq!(after, before);
    assert_eq!(
        messages(&env, &w.dir.path().join("after-bundle")),
        before_messages
    );
    assert!(!prompt.exists() && !listing.exists());
    drop(daemon);
}

#[test]
fn the_hook_spools_in_the_captured_xdg_state_directory() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let elsewhere = dir.path().join("xdg");
    let mut child = crate::command(dir.path())
        .args(["hook", "Stop"])
        .env("XDG_STATE_HOME", &elsewhere)
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), br#"{"session_id": "s1"}"#).unwrap();
    assert!(child.wait().unwrap().success());
    let spool = elsewhere.join("polylogue/hooks");
    let spool_files: Vec<_> = std::fs::read_dir(&spool)
        .unwrap_or_else(|e| panic!("{}: {e}", spool.display()))
        .collect();
    assert_eq!(spool_files.len(), 1);
    assert!(
        !home.join(".local/state/polylogue/hooks").exists(),
        "captured XDG state must not spill into the HOME default"
    );
    let record = std::fs::read_to_string(spool_files[0].as_ref().unwrap().path()).unwrap();
    let record: serde_json::Value = serde_json::from_str(record.trim()).unwrap();
    assert_eq!(record["session_id"], "s1");
    assert_eq!(record["event"], "Stop");
}

mod structured_json;

mod read_current;
mod read_pages;

mod stored_text;

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
fn the_daemon_refuses_an_archive_it_has_no_migration_from_and_exits_with_its_code() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("archive");
    let home = dir.path().join("home");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let (opened, lock) = polylogue_store::Archive::open_or_create(&archive).unwrap();
    drop(lock);
    // As an archive written by a build that declared version 1 with other
    // tables.
    rusqlite::Connection::open(opened.archive_db())
        .unwrap()
        .execute_batch("DROP TABLE revision")
        .unwrap();
    let mut command = crate::command(dir.path());
    command
        .arg("daemon")
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
        stderr.contains("has tables this Polylogue build has no migration from"),
        "{stderr}"
    );
    assert!(!stderr.contains("serving"), "{stderr}");
    assert!(!archive.join("daemon.sock").exists());
}

mod python;

#[test]
fn killed_daemon_runtime_files_are_removed_with_its_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("archive");
    let home = dir.path().join("home");
    let config = dir.path().join("config");
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let daemon = start_daemon(&env);
    wait_status(&env, |_| true);
    let runtime = dir.path().join("runtime/polylogue");
    // Readiness of the socket precedes the HTTP listener's access files.
    loop {
        if runtime.exists()
            && std::fs::read_dir(&runtime).unwrap().any(|entry| {
                let path = entry.unwrap().path();
                path.join("port").exists() && path.join("token").exists()
            })
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(daemon);
    assert!(runtime.exists());
    dir.close().unwrap();
    assert!(!runtime.exists());
}

mod query_surfaces;

mod agent_guidance;

#[cfg(target_os = "linux")]
mod terminal;

mod index_open;
