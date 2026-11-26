//! End to end: the real binary as a daemon over a temporary archive and
//! home directory, driven through the CLI.

#[cfg(target_os = "linux")]
use std::io::Write;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

/// Execute the typed continuation as supplied by the daemon. CLI formatting
/// and execution are tested in the binary through its actual declarations.
fn follow(
    env: &crate::Fixture,
    call: &polylogue_model::read::ReadyCall,
) -> command_surface::Answer<serde_json::Value> {
    use std::io::Write as _;
    use std::os::unix::net::UnixStream;
    assert_eq!(call.request.operation, "rows");
    let mut socket = UnixStream::connect(env.archive().join("daemon.sock")).unwrap();
    socket.set_read_timeout(Some(READY_DEADLINE)).unwrap();
    serde_json::to_writer(&mut socket, &call.request).unwrap();
    socket.write_all(b"\n").unwrap();
    socket_answer(socket)
}

/// The final answer of one reply on the socket: its server-sent-event
/// frames, of which the `answer` frame is last.
pub(super) fn socket_answer(mut socket: impl Read) -> command_surface::Answer<serde_json::Value> {
    let mut decoder = command_surface::SseDecoder::new();
    let mut bytes = [0; 8192];
    loop {
        let count = socket.read(&mut bytes).unwrap();
        assert!(count > 0, "the reply ended before its answer");
        for frame in decoder.push(&bytes[..count]).unwrap() {
            if frame.event == "answer" {
                return serde_json::from_str(&frame.data).unwrap();
            }
        }
    }
}

pub(super) fn listing(value: serde_json::Value) -> serde_json::Value {
    assert_eq!(value["kind"], "listing", "{value}");
    value["reply"].clone()
}

fn read_page(value: serde_json::Value) -> polylogue_model::read::ReadPage {
    assert_eq!(value["kind"], "steps", "{value}");
    let steps = value["reply"]["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 1, "{value}");
    let outcome = &steps[0]["outcome"];
    assert_eq!(outcome["status"], "done", "{value}");
    assert_eq!(outcome["reply"]["kind"], "read", "{value}");
    serde_json::from_value(outcome["reply"]["reply"].clone()).unwrap()
}

/// The attachment range a line's one read step answered with.
fn attachment_page(value: &serde_json::Value) -> serde_json::Value {
    assert_eq!(value["kind"], "steps", "{value}");
    let outcome = &value["reply"]["steps"][0]["outcome"];
    assert_eq!(outcome["status"], "done", "{value}");
    assert_eq!(outcome["reply"]["kind"], "attachment", "{value}");
    outcome["reply"]["reply"].clone()
}

fn read_failure(answer: command_surface::Answer<serde_json::Value>) -> command_surface::Failure {
    if let command_surface::Outcome::Error(error) = &answer.outcome {
        return error.clone();
    }
    let value = answer.value().unwrap();
    let steps = value["reply"]["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 1, "{answer:?}");
    assert_eq!(steps[0]["outcome"]["status"], "failed", "{answer:?}");
    serde_json::from_value(steps[0]["outcome"]["error"].clone()).unwrap()
}

fn followed_page(
    env: &crate::Fixture,
    call: &polylogue_model::read::ReadyCall,
) -> polylogue_model::read::ReadPage {
    let answer = follow(env, call);
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    read_page(answer.value().unwrap().clone())
}

const READY_DEADLINE: Duration = Duration::from_secs(60);

pub(super) struct Daemon(pub(super) Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Starts the fixture's daemon and waits until it has caught up with its
/// home and archive: its first index installed and its first pass done.
pub(super) fn start_daemon(env: &crate::Fixture) -> Daemon {
    start_daemon_with(env, command(env))
}

/// Starts `command`, already holding the fixture's arguments, as the daemon,
/// waits for it to serve and then for it to catch up ([`wait_caught_up`]).
fn start_daemon_with(env: &crate::Fixture, command: Command) -> Daemon {
    let daemon = serving(command);
    wait_caught_up(env);
    daemon
}

/// Starts `command` as the daemon and waits for its readiness line.
fn serving(mut command: Command) -> Daemon {
    command.arg("daemon").stderr(Stdio::piped());
    let mut daemon = Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap());
    use std::os::fd::AsRawFd;
    let mut stderr = daemon.0.stderr.take().unwrap();
    let mut startup = Vec::new();
    let until = Instant::now() + READY_DEADLINE;
    loop {
        let left = until.saturating_duration_since(Instant::now());
        assert!(
            !left.is_zero(),
            "daemon readiness timed out; status={:?}; stderr:\n{}",
            daemon.0.try_wait().unwrap(),
            String::from_utf8_lossy(&startup)
        );
        let mut event = libc::pollfd {
            fd: stderr.as_raw_fd(),
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        // SAFETY: event is writable, and this owned pipe stays open through poll.
        #[allow(unsafe_code)]
        let ready = unsafe {
            libc::poll(
                &mut event,
                1,
                left.as_millis().clamp(1, i32::MAX as u128) as i32,
            )
        };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            panic!(
                "daemon readiness pipe: {error}; stderr:\n{}",
                String::from_utf8_lossy(&startup)
            );
        }
        if ready == 0 {
            continue;
        }
        let mut bytes = [0; 4096];
        assert!(
            event.revents & (libc::POLLIN | libc::POLLHUP) != 0,
            "daemon readiness pipe events={}; status={:?}; stderr:\n{}",
            event.revents,
            daemon.0.try_wait().unwrap(),
            String::from_utf8_lossy(&startup)
        );
        let count = match stderr.read(&mut bytes) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => panic!(
                "daemon readiness read: {error}; stderr:\n{}",
                String::from_utf8_lossy(&startup)
            ),
        };
        if count == 0 {
            panic!(
                "daemon closed readiness output; status={:?}; stderr:\n{}",
                daemon.0.try_wait().unwrap(),
                String::from_utf8_lossy(&startup)
            );
        }
        startup.extend_from_slice(&bytes[..count]);
        if startup
            .split_inclusive(|byte| *byte == b'\n')
            .any(|line| line.ends_with(b"\n") && line.starts_with(b"<6>polylogue: serving "))
        {
            assert!(
                daemon.0.try_wait().unwrap().is_none(),
                "daemon exited after readiness; stderr:\n{}",
                String::from_utf8_lossy(&startup)
            );
            daemon.0.stderr = Some(stderr);
            break;
        }
    }
    daemon
}

/// Waits for the daemon to catch up with what it was given (`status
/// --wait`): no index being built, no pass running or due, derived figures
/// and search text current, and the status read after the latest commit.
/// Every test that reads what it gave the daemon (a start, an import, a
/// written file) waits here first. While `ready` does not hold for a
/// caught-up status, the change the test made has not reached the daemon
/// yet, and the next wait is for a status that caught up after a later
/// change (`--after`): each answer follows something the daemon did, and
/// nothing is polled.
pub(super) fn wait_status(
    env: &crate::Fixture,
    ready: impl Fn(&serde_json::Value) -> bool,
) -> serde_json::Value {
    let wait = format!("{}s", READY_DEADLINE.as_secs());
    let mut after: Option<String> = None;
    loop {
        // The wait sets its own bound, so a loaded host delays the answer
        // instead of turning it into a timeout.
        let mut args = vec!["--deadline", "none", "--json", "status", "--wait", &wait];
        if let Some(after) = &after {
            args.extend(["--after", after]);
        }
        let out = cli(env, &args);
        let answer: command_surface::Answer<serde_json::Value> =
            serde_json::from_slice(&out.stdout)
                .unwrap_or_else(|error| panic!("invalid status: {error}: {out:?}"));
        assert_eq!(
            out.status.code(),
            Some(i32::from(answer.exit_code())),
            "{out:?}"
        );
        let status = answer
            .value()
            .unwrap_or_else(|| panic!("status failed: {out:?}"))
            .clone();
        assert_eq!(
            status["caught_up"], true,
            "the daemon did not catch up within {READY_DEADLINE:?}: {status}"
        );
        if status["store"].is_object() && ready(&status) {
            return status;
        }
        let changes = status["changes"].as_u64().unwrap();
        assert!(
            after.is_none_or(|after| changes > after.parse().unwrap()),
            "the awaited status did not come within {READY_DEADLINE:?}: {status}"
        );
        after = Some(changes.to_string());
    }
}

/// Waits until the daemon has caught up with what it was given; see
/// [`wait_status`].
pub(super) fn wait_caught_up(env: &crate::Fixture) -> serde_json::Value {
    wait_status(env, |_| true)
}

/// The current rows of `query` once the daemon has caught up, which must
/// satisfy `expected`.
pub(super) fn caught_up_rows(
    env: &crate::Fixture,
    query: &str,
    expected: impl Fn(&serde_json::Value) -> bool,
) -> serde_json::Value {
    wait_caught_up(env);
    let out = cli(env, &["--json", query]);
    let rows = query_rows(&out);
    assert!(expected(&rows), "{rows}");
    rows
}

pub(super) fn query_rows(out: &Output) -> serde_json::Value {
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&out.stdout).unwrap();
    match answer {
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(value),
            ..
        } => {
            assert!(out.status.success(), "{out:?}");
            listing(value)
        }
        command_surface::Answer {
            outcome: command_surface::Outcome::Empty(value),
            ..
        } => {
            assert_eq!(out.status.code(), Some(1), "{out:?}");
            let value = listing(value);
            assert_eq!(value["count"], 0, "{value}");
            assert_eq!(value["rows"], serde_json::json!([]), "{value}");
            value
        }
        answer => panic!("expected rows or typed empty query: {answer:?}; {out:?}"),
    }
}

#[cfg(target_os = "linux")]
#[test]
fn killing_the_test_parent_stops_its_real_daemon() {
    use std::os::unix::process::ExitStatusExt;

    let dir = tempfile::tempdir().unwrap();
    let mut first = Daemon(
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
    let mut stdout = BufReader::new(first.0.stdout.take().unwrap());
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
    let mut stderr = BufReader::new(first.0.stderr.take().unwrap());
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
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&running.stdout).unwrap();
    assert_eq!(running.status.code(), Some(i32::from(answer.exit_code())));
    assert!(
        matches!(
            &answer.outcome,
            command_surface::Outcome::Ok(_) | command_surface::Outcome::Degraded { .. }
        ),
        "{answer:?}"
    );
    assert_eq!(answer.value().unwrap()["pid"].as_u64(), Some(daemon_pid));
    assert!(matches!(
        polylogue_store::Archive::open_or_create(&dir.path().join("archive")),
        Err(polylogue_store::StoreError::Locked { .. })
    ));
    let exit = polylogue_testkit::ProcessExit::watch(daemon_pid.try_into().unwrap()).unwrap();
    first.0.kill().unwrap();
    assert_eq!(first.0.wait().unwrap().signal(), Some(9));
    // Stderr can close before other descriptors during process teardown.
    // Wait for process exit before checking that the archive lock is released.
    exit.wait().unwrap();
    let mut remaining = String::new();
    stderr.read_to_string(&mut remaining).unwrap();
    let (_archive, _lock) = polylogue_store::Archive::open_or_create(&dir.path().join("archive"))
        .expect("the killed first daemon must release its archive lock");
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

    let out = cli(&env, &["--json", "sessions"]);
    let ids: Vec<String> = listing(json(&out))["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_owned())
        .collect();
    let mut ids = ids;
    ids.sort();
    assert_eq!(ids, ["claude-code:exported", "claude-code:live"]);

    let out = cli(&env, &["--json", "session:claude-code:exported", "read"]);
    let read: polylogue_model::read::ReadPage = read_page(json(&out));
    let mut imported_items = read_current::Items::default();
    imported_items.push(&read);
    assert!(
        read.document.footer.next.is_none(),
        "single short imported message must fit its root"
    );
    assert_eq!(imported_items.text(), "from an import");

    let out = cli(&env, &["session:claude-code:live", "read"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains("from a live file"));

    drop(daemon);
    let out = cli(&env, &["--json", "status"]);
    assert_eq!(out.status.code(), Some(6));
    assert_eq!(json(&out)["error"]["code"], "daemon_required");
    // The web reader is the running daemon's: without one there is no
    // reader to open.
    let out = cli(&env, &["--json", "web"]);
    assert_eq!(out.status.code(), Some(6));
    assert_eq!(json(&out)["error"]["code"], "daemon_required");
}

#[test]
fn ai_studio_drive_prompt_and_title_read_identically_after_a_rebuild_from_stored_bytes() {
    use polylogue_model::read::{ItemKind, ReadHeader, ReadItem, ValueState};

    fn reading(env: &crate::Fixture) -> (ReadHeader, Vec<ReadItem>) {
        // The system instruction is a harness message, which a default read
        // collapses to an indicator (docs/reading.md, harness messages).
        let out = cli(
            env,
            &[
                "--json",
                "session:ai-studio:drive-id",
                "read",
                "--show",
                "harness:full",
            ],
        );
        assert!(out.status.success(), "{out:?}");
        let mut page = read_page(json(&out));
        let mut items = read_current::Items::default();
        let mut header = None;
        let mut continuations = std::collections::BTreeSet::new();
        loop {
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
                continuations.insert(serde_json::to_string(&next.request).unwrap()),
                "repeated continuation"
            );
            page = followed_page(env, &next);
        }
        (header.unwrap(), items.into_values())
    }

    fn messages(env: &crate::Fixture, destination: &Path) -> Vec<polylogue_model::Message> {
        std::fs::create_dir(destination).unwrap();
        let out = cli(
            env,
            &[
                "--json",
                "session:ai-studio:drive-id",
                "read",
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
            panic!("bundle must start with its manifest");
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
    // Read the child's environment, whose XDG state root differs from
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
    // The collector names a prompt's file by its Drive file ID, which joins
    // it to the listing's title; an imported copy is named by its content.
    let prompt = dir.join("drive-id.json");
    let prompt_bytes = serde_json::to_vec(&serde_json::json!({
        "chunkedPrompt":{"chunks":[{"role":"user","text":"What is an electron?"},{"role":"model","text":"A lepton.","tokenCount":3}]},
        "runSettings":{"model":"models/example"},"systemInstruction":{"text":"Be brief."}
    })).unwrap();
    std::fs::write(&prompt, &prompt_bytes).unwrap();
    let daemon = start_daemon(&env);
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
    // Chunks without IDs are named by content: the first 16 hex digits of
    // BLAKE3("polylogue-message-content", 0, ["user",{"role":"user",...}]).
    let expected_ids = [
        "ai-studio:8:drive-id#system:0:",
        "ai-studio:drive-id~user-db51419c0768f7f8-0",
        "ai-studio:drive-id~assistant-f18fd0cdf2acb700-0",
    ];
    assert_eq!(
        headers
            .iter()
            .map(|(reference, _)| reference.as_str())
            .collect::<Vec<_>>(),
        expected_ids.map(|id| format!("message:{id}"))
    );
    assert_eq!(headers[2].1.model, ValueState::known("example".to_owned()));
    // Usage belongs to the stored message, rather than the presentation header.
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
    // stored source bytes, after their provider files have disappeared.
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
        .args(["internal", "hook", "Stop"])
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
mod seen;

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
    let started = std::time::Instant::now();
    loop {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the daemon never wrote its HTTP access files"
        );
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

mod surface_table;

mod agent_guidance;

#[cfg(target_os = "linux")]
mod terminal;

mod index_open;

mod entry;

#[cfg(target_os = "linux")]
#[test]
fn terminal_resume_without_recorded_directory_is_refused_with_exit_3() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let project = home.join(".claude/projects/-p");
    std::fs::create_dir_all(&project).unwrap();
    // The record names no working directory.
    std::fs::write(project.join("s1.jsonl"), line("u1", "s1", "no directory")).unwrap();
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let _daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(1));

    use rustix::pty::{OpenptFlags, grantpt, ioctl_tiocgptpeer, openpt, unlockpt};
    let flags = OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC;
    let master = openpt(flags).unwrap();
    grantpt(&master).unwrap();
    unlockpt(&master).unwrap();
    let terminal = std::fs::File::from(ioctl_tiocgptpeer(&master, flags).unwrap());
    let out = env
        .command()
        .args(["session:claude-code:s1", "resume"])
        .stdin(terminal.try_clone().unwrap())
        .stdout(terminal)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    drop(master);
    // docs/interfaces.md#exit-codes: no way to reopen the session is `refused`.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("directory_unknown"), "{stderr}");
    let fallback = stderr
        .lines()
        .find_map(|line| {
            line.split_once("continue with context instead: ")
                .map(|(_, command)| command)
        })
        .expect("a usable context-assisted command");
    let continued = printed_cli(&env, fallback);
    assert!(continued.status.success(), "{continued:?}");
    assert!(
        String::from_utf8_lossy(&continued.stdout).contains("no directory"),
        "{continued:?}"
    );
    assert_eq!(fallback, "polylogue session:claude-code:s1 brief");
}

#[test]
fn failed_step_exits_with_its_code_and_prints_undo() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let project = home.join(".claude/projects/-line-prefix");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("s1.jsonl"),
        line("u1", "s1", "Synthetic failed-prefix control"),
    )
    .unwrap();
    let env = crate::Fixture::new(&archive, &home, &dir.path().join("config"));
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["messages"] == 1);
    let words = [
        "--request-id",
        "line-prefix",
        "session:claude-code:s1",
        "tag",
        "prefix-done",
        "read",
        "--view",
        "raw",
        "--show",
        "tools:full",
        "tag",
        "not-run",
    ];
    let mut undo = None;
    for _ in 0..2 {
        let output = cli(&env, &words);
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        let text = String::from_utf8(output.stderr).unwrap();
        assert!(text.contains("view raw does not accept tools"), "{text}");
        assert!(text.contains("tag not run"), "{text}");
        let command = text
            .lines()
            .find_map(|line| line.strip_prefix("undo: "))
            .expect("the committed prefix supplies undo");
        if let Some(previous) = &undo {
            assert_eq!(command, previous);
        }
        undo = Some(command.to_owned());
        let listed = cli(
            &env,
            &["--json", "annotations where kind:tag", "--fields", "key"],
        );
        assert!(listed.status.success(), "{listed:?}");
        let listing = json(&listed);
        assert_eq!(listing["kind"], "listing");
        assert_eq!(
            listing["reply"]["rows"],
            serde_json::json!([{"key":"prefix-done"}])
        );
    }
    let withdrawn = printed_cli(&env, undo.as_deref().unwrap());
    assert!(withdrawn.status.success(), "{withdrawn:?}");
    let remaining = listing(json(&cli(&env, &["--json", "annotations where kind:tag"])));
    assert_eq!(remaining["count"], 0);
    let mut json_words = vec!["--json", "--request-id", "line-prefix-json"];
    json_words.extend_from_slice(&words[2..]);
    let tag = json_words
        .iter()
        .position(|word| *word == "prefix-done")
        .unwrap();
    json_words[tag] = "json-prefix-done";
    let output = cli(&env, &json_words);
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&output.stdout).unwrap();
    let reports = answer.value().unwrap()["reply"]["steps"]
        .as_array()
        .unwrap();
    assert_eq!(reports.len(), 3);
    assert_eq!(reports[0]["outcome"]["status"], "done");
    assert_eq!(reports[1]["outcome"]["status"], "failed");
    assert_eq!(reports[2]["outcome"]["status"], "not_run");
    assert_eq!(reports[0]["outcome"]["undo"].as_array().unwrap().len(), 1);
    let failure: command_surface::Failure =
        serde_json::from_value(reports[1]["outcome"]["error"].clone()).unwrap();
    assert_eq!(
        output.status.code(),
        Some(i32::from(failure.kind.exit_code()))
    );
}

fn two_session_cli() -> (tempfile::TempDir, crate::Fixture, Daemon) {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-line");
    std::fs::create_dir_all(&project).unwrap();
    for id in ["s1", "s2"] {
        std::fs::write(
            project.join(format!("{id}.jsonl")),
            line(id, id, "Synthetic line candidate"),
        )
        .unwrap();
    }
    let env = crate::Fixture::new(
        &dir.path().join("archive"),
        &home,
        &dir.path().join("config"),
    );
    let daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["messages"] == 2);
    (dir, env, daemon)
}

#[test]
fn verb_alone_refused_off_terminal_with_lines() {
    let (_dir, env, _daemon) = two_session_cli();
    let output = cli(&env, &["--json", "read"]);
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&output.stdout).unwrap();
    let failure = answer
        .failure()
        .expect("a verb without a choice is refused off terminal");
    assert_eq!(output.status.code(), Some(i32::from(answer.exit_code())));
    assert_eq!(failure.code.as_deref(), Some("ambiguous"));
    let hint = failure.hint.as_deref().unwrap();
    assert!(
        hint.lines().any(|line| line == "polylogue latest read"),
        "{hint}"
    );
    assert!(
        hint.lines().any(|line| line == "polylogue SESSION read"),
        "{hint}"
    );
    let output = cli(&env, &["--json", "sessions | sort by id", "pick", "read"]);
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&output.stdout).unwrap();
    let failure = answer.failure().expect("a picker needs a terminal");
    assert_eq!(failure.code.as_deref(), Some("ambiguous"));
    assert_eq!(
        failure.candidates(),
        [
            "polylogue session:claude-code:s1 read",
            "polylogue session:claude-code:s2 read"
        ]
    );
}

#[test]
fn explain_option_runs_nothing() {
    let (_dir, env, _daemon) = two_session_cli();
    let output = cli(
        &env,
        &[
            "--json",
            "--explain",
            "sessions",
            "tag",
            "not-written",
            "--all",
        ],
    );
    assert!(output.status.success(), "{output:?}");
    let explained = json(&output);
    assert_eq!(explained["kind"], "explained");
    let steps = explained["reply"]["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 1, "{explained}");
    assert_eq!(steps[0]["verb"], "tag", "{explained}");
    assert_eq!(steps[0]["access"], "write", "{explained}");
    assert!(
        steps[0]["would"]
            .as_str()
            .unwrap()
            .starts_with("would tag the rows of "),
        "{explained}"
    );
    assert_eq!(
        listing(json(&cli(&env, &["--json", "annotations where kind:tag"])))["count"],
        0
    );
}

#[test]
fn ordered_read_destinations_receive_their_own_steps() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let project = home.join(".claude/projects/-line-delivery");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("s1.jsonl"),
        line("u1", "s1", "Synthetic ordered delivery body"),
    )
    .unwrap();
    let env = crate::Fixture::new(&archive, &home, &dir.path().join("config"));
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["messages"] == 1);
    let before = dir.path().join("before.md");
    let after = dir.path().join("after.md");
    let output = cli(
        &env,
        &[
            "--deadline",
            "none",
            "--request-id",
            "ordered-destinations",
            "session:claude-code:s1",
            "read",
            "--show",
            "annotations",
            "--to",
            before.to_str().unwrap(),
            "tag",
            "delivery-after-first-read",
            "read",
            "--show",
            "annotations",
            "--to",
            after.to_str().unwrap(),
            "read",
            "--show",
            "annotations",
        ],
    );
    assert!(output.status.success(), "{output:?}");
    let before = std::fs::read_to_string(before).unwrap();
    let after = std::fs::read_to_string(after).unwrap();
    assert!(
        before.contains("Synthetic ordered delivery body"),
        "{before}"
    );
    assert!(!before.contains("delivery-after-first-read"), "{before}");
    assert!(after.contains("delivery-after-first-read"), "{after}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Synthetic ordered delivery body"),
        "{stdout}"
    );
    assert!(stdout.contains("delivery-after-first-read"), "{stdout}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    let undo = stderr
        .lines()
        .filter_map(|line| line.strip_prefix("undo: "))
        .collect::<Vec<_>>();
    assert_eq!(undo.len(), 1, "{stderr}");
    let withdrawn = printed_cli(&env, undo[0]);
    assert!(withdrawn.status.success(), "{withdrawn:?}");
    assert_eq!(
        listing(json(&cli(&env, &["--json", "annotations where kind:tag"])))["count"],
        0
    );
}
/// Drive the real binary through a PTY, waiting for a seeded candidate before
/// choosing. A failed annotation still kills and reaps the owned client.
fn choose(env: &crate::Fixture, args: &[&str], keys: &[u8]) -> Output {
    choose_output(env, args, keys, false)
}
fn choose_output(env: &crate::Fixture, args: &[&str], keys: &[u8], capture: bool) -> Output {
    use rustix::pty::{OpenptFlags, grantpt, ioctl_tiocgptpeer, openpt, unlockpt};
    use std::fs::File;
    use std::sync::mpsc;
    struct Client(Option<Child>);
    impl Drop for Client {
        fn drop(&mut self) {
            if let Some(child) = self.0.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let flags = OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC;
    let master = openpt(flags).unwrap();
    grantpt(&master).unwrap();
    unlockpt(&master).unwrap();
    let slave = File::from(ioctl_tiocgptpeer(&master, flags).unwrap());
    assert!(
        Command::new("stty")
            .args(["rows", "24", "cols", "100"])
            .stdin(slave.try_clone().unwrap())
            .status()
            .unwrap()
            .success()
    );
    let mut master = File::from(master);
    let mut command = if capture {
        let mut command = Command::new("setsid");
        env.environment(&mut command);
        command.args(["--ctty", "--wait", crate::BIN]);
        command
    } else {
        env.command()
    };
    let child = command
        .args(args)
        .stdin(slave.try_clone().unwrap())
        .stdout(if capture {
            Stdio::piped()
        } else {
            Stdio::from(slave.try_clone().unwrap())
        })
        .stderr(if capture {
            Stdio::from(slave.try_clone().unwrap())
        } else {
            Stdio::piped()
        })
        .spawn()
        .unwrap();
    let mut client = Client(Some(child));
    drop(command);
    drop(slave);
    let (send, receive) = mpsc::channel();
    let mut reader = master.try_clone().unwrap();
    let output_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut buffer = [0; 8192];
        while let Ok(count) = reader.read(&mut buffer) {
            if count == 0 {
                break;
            }
            bytes.extend_from_slice(&buffer[..count]);
            let _ = send.send(buffer[..count].to_vec());
        }
        bytes
    });
    let deadline = Instant::now() + READY_DEADLINE;
    let mut shown = Vec::new();
    while !String::from_utf8_lossy(&shown).contains("Alpha picker candidate") {
        let left = deadline
            .checked_duration_since(Instant::now())
            .expect("picker did not show its accepted candidates");
        match receive.recv_timeout(left) {
            Ok(bytes) => shown.extend(bytes),
            Err(error) => {
                let mut child = client.0.take().unwrap();
                let _ = child.kill();
                let output = child.wait_with_output().unwrap();
                panic!(
                    "picker ended before its candidates: {error}; shown={}; child={output:?}",
                    String::from_utf8_lossy(&shown)
                );
            }
        }
    }
    master.write_all(keys).unwrap();
    while client.0.as_mut().unwrap().try_wait().unwrap().is_none() {
        assert!(
            Instant::now() < deadline,
            "picker did not finish after selection"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let mut output = client.0.take().unwrap().wait_with_output().unwrap();
    drop(master);
    let terminal = output_reader.join().unwrap();
    if capture {
        output.stderr = terminal;
    } else {
        output.stdout = terminal;
    }
    output
}

fn plain_terminal(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut characters = text.chars();
    let mut result = String::new();
    while let Some(character) = characters.next() {
        if character == '\x1b' && characters.next() == Some('[') {
            for character in characters.by_ref() {
                if ('@'..='~').contains(&character) {
                    break;
                }
            }
        } else if character != '\r' {
            result.push(character);
        }
    }
    result
}

pub(super) fn printed_cli(env: &crate::Fixture, command: &str) -> Output {
    let mut shell = Command::new("sh");
    env.environment(&mut shell);
    let script = format!("polylogue() {{ \"$POLYLOGUE_TEST_BINARY\" \"$@\"; }}\n{command}");
    shell
        .env("POLYLOGUE_TEST_BINARY", crate::BIN)
        .args(["-c", &script])
        .output()
        .unwrap()
}

#[test]
fn picker_prefix_suffix_retry_cancellation_and_complete_undo_set() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-picker");
    std::fs::create_dir_all(&project).unwrap();
    for (id, title) in [
        ("s1", "Alpha picker candidate"),
        ("s2", "Beta picker candidate"),
    ] {
        std::fs::write(project.join(format!("{id}.jsonl")), line(id, id, title)).unwrap();
    }
    let env = crate::Fixture::new(
        &dir.path().join("archive"),
        &home,
        &dir.path().join("config"),
    );
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["messages"] == 2);
    for (id, title) in [
        ("s1", "Alpha picker candidate"),
        ("s2", "Beta picker candidate"),
    ] {
        let out = cli(
            &env,
            &[
                &format!("session:claude-code:{id}"),
                "note",
                title,
                "--as",
                "title",
            ],
        );
        assert!(out.status.success(), "{out:?}");
    }
    let chosen = choose_output(&env, &["sessions | sort by id", "pick"], b"\r", true);
    assert!(chosen.status.success(), "{chosen:?}");
    assert_eq!(chosen.stdout, b"session:claude-code:s1\n");
    assert!(String::from_utf8_lossy(&chosen.stderr).contains("Alpha picker candidate"));
    let direct = choose(&env, &["sessions | sort by id", "pick"], b"\r");
    assert!(direct.status.success(), "{direct:?}");
    assert_eq!(
        plain_terminal(&direct.stdout)
            .lines()
            .filter(|line| *line == "session:claude-code:s1")
            .count(),
        1
    );
    let explained = cli(
        &env,
        &[
            "--explain",
            "sessions",
            "tag",
            "not-written",
            "--all",
            "pick",
            "tag",
            "also-not-written",
        ],
    );
    assert!(explained.status.success(), "{explained:?}");
    let empty = json(&cli(&env, &["--json", "annotations where kind:tag"]));
    assert_eq!(empty["reply"]["count"], 0, "{empty}");
    let simple = choose(
        &env,
        &[
            "--request-id",
            "simple-picker",
            "sessions | sort by id",
            "pick",
            "note",
            "Selected through the picker",
        ],
        b"\r",
    );
    assert!(simple.status.success(), "{simple:?}");
    let picked = json(&cli(
        &env,
        &["--json", "annotations where id:simple-picker-pick-suffix-1"],
    ));
    assert_eq!(picked["reply"]["count"], 1, "{picked}");
    for (id, count) in [("s1", 1), ("s2", 0)] {
        let selected = json(&cli(
            &env,
            &[
                "--json",
                &format!("annotations where kind:note about:session:claude-code:{id}"),
            ],
        ));
        assert_eq!(selected["reply"]["count"], count, "{selected}");
    }
    let words = [
        "--request-id",
        "picker-prefix-suffix",
        "sessions where NOT tag:prefix | sort by id",
        "tag",
        "prefix",
        "--all",
        "pick",
        "tag",
        "suffix",
    ];
    let mut undo = Vec::new();
    for _ in 0..2 {
        let output = choose(&env, &words, b"\r");
        assert!(output.status.success(), "{output:?}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        let current: Vec<_> = stderr
            .lines()
            .filter_map(|line| line.strip_prefix("undo: "))
            .map(str::to_owned)
            .collect();
        assert_eq!(current.len(), 2, "{stderr}");
        if undo.is_empty() {
            undo = current;
        } else {
            assert_eq!(current, undo);
        }
        let found = json(&cli(&env, &["--json", "annotations where kind:tag"]));
        assert_eq!(found["reply"]["count"], 3, "{found}");
        let chosen = json(&cli(
            &env,
            &["--json", "sessions where tag:suffix", "--fields", "id"],
        ));
        assert_eq!(
            chosen["reply"]["rows"],
            serde_json::json!([{"id":"claude-code:s1"}]),
            "{chosen}"
        );
    }
    for command in undo.iter().rev() {
        let output = printed_cli(&env, command);
        assert!(output.status.success(), "{output:?}");
    }
    let found = json(&cli(&env, &["--json", "annotations where kind:tag"]));
    assert_eq!(found["reply"]["count"], 0, "{found}");
    let cancelled = choose(
        &env,
        &[
            "--request-id",
            "picker-cancel",
            "sessions where NOT tag:prefix | sort by id",
            "tag",
            "prefix",
            "--all",
            "pick",
            "tag",
            "suffix",
        ],
        b"\x1b",
    );
    assert!(cancelled.status.success(), "{cancelled:?}");
    let stderr = String::from_utf8(cancelled.stderr).unwrap();
    assert_eq!(
        stderr
            .lines()
            .filter(|line| line.starts_with("undo: "))
            .count(),
        1,
        "{stderr}"
    );
    let found = json(&cli(&env, &["--json", "annotations where kind:tag"]));
    assert_eq!(found["reply"]["count"], 2, "{found}");
    let suffix = json(&cli(&env, &["--json", "sessions where tag:suffix"]));
    assert_eq!(suffix["reply"]["count"], 0, "{suffix}");
    let prior = cli(
        &env,
        &[
            "--request-id",
            "tag-existing",
            "session:claude-code:s1",
            "tag",
            "only-once",
        ],
    );
    assert!(prior.status.success(), "{prior:?}");
    let mixed_args = [
        "--json",
        "--request-id",
        "tag-mixed",
        "sessions | sort by id",
        "tag",
        "only-once",
        "--all",
    ];
    let mixed = cli(&env, &mixed_args);
    assert!(mixed.status.success(), "{mixed:?}");
    let first = json(&mixed);
    let undo = first["reply"]["steps"][0]["outcome"]["undo"]
        .as_array()
        .unwrap();
    assert_eq!(undo.len(), 1);
    let retry = cli(&env, &mixed_args);
    assert!(retry.status.success(), "{retry:?}");
    assert_eq!(
        json(&retry)["reply"]["steps"][0]["outcome"]["undo"],
        serde_json::json!(undo)
    );
    let restored = printed_cli(&env, undo[0].as_str().unwrap());
    assert!(restored.status.success(), "{restored:?}");
    let remaining = json(&cli(
        &env,
        &[
            "--json",
            "sessions where tag:only-once | sort by id",
            "--fields",
            "id",
        ],
    ));
    assert_eq!(
        remaining["reply"]["rows"],
        serde_json::json!([{"id":"claude-code:s1"}])
    );
    let already = cli(
        &env,
        &[
            "--json",
            "--request-id",
            "tag-already",
            "session:claude-code:s1",
            "tag",
            "only-once",
        ],
    );
    assert!(already.status.success(), "{already:?}");
    assert!(
        json(&already)["reply"]["steps"][0]["outcome"]
            .get("undo")
            .is_none()
    );
}

#[test]
fn printed_cli_continuation_and_previous_keep_the_accepted_options() {
    use polylogue_model::read::ReadDocument;
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-pages");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("s1.jsonl"),
        line("u1", "s1", &"Synthetic λ🙂 page text. ".repeat(100)),
    )
    .unwrap();
    let env = crate::Fixture::new(
        &dir.path().join("archive"),
        &home,
        &dir.path().join("config"),
    );
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["messages"] == 1);
    let output = cli(
        &env,
        &[
            "--budget",
            "333",
            "--deadline",
            "none",
            "--request-id",
            "cli-page-grants",
            "--all",
            "session:claude-code:s1",
            "read",
            "--format",
            "json",
            "--budget",
            "1",
        ],
    );
    assert!(output.status.success(), "{output:?}");
    let first: ReadDocument = serde_json::from_slice(&output.stdout).unwrap();
    let next = first.footer.next.as_ref().expect("one token must continue");
    let check_options = |call: &polylogue_model::read::ReadyCall| {
        assert_eq!(
            call.request.budget_tokens,
            command_surface::TokenBudget::Tokens(333)
        );
        assert_eq!(call.request.deadline, command_surface::Deadline::None);
        assert!(!call.request.fresh);
        assert_eq!(call.request.format, command_surface::Format::Text);
        assert_eq!(call.request.arguments["all"], true);
        assert_eq!(call.request.arguments["steps"][0]["budget"], 1);
        assert_eq!(call.request.arguments["steps"][0]["format"], "json");
    };
    check_options(next);
    let stderr = String::from_utf8(output.stderr).unwrap();
    let commands = stderr
        .lines()
        .filter_map(|line| line.strip_prefix("continue: "))
        .collect::<Vec<_>>();
    assert_eq!(commands.len(), 1, "{stderr}");
    assert!(commands[0].contains(&next.request.request_id));
    let continued = printed_cli(&env, commands[0]);
    assert!(continued.status.success(), "{continued:?}");
    let second: ReadDocument = serde_json::from_slice(&continued.stdout).unwrap();
    let expected = followed_page(&env, next);
    assert_eq!(second.items, expected.document.items);
    assert_eq!(second.header, expected.document.header);
    let previous = second
        .footer
        .previous
        .as_ref()
        .expect("second page has a predecessor");
    check_options(previous);
    let cursor = previous.request.arguments["steps"][0]["cursor"]
        .as_str()
        .unwrap();
    let returned = cli(
        &env,
        &[
            "--budget",
            "333",
            "--deadline",
            "none",
            "--request-id",
            &previous.request.request_id,
            "--all",
            "session:claude-code:s1",
            "read",
            "--format",
            "json",
            "--budget",
            "1",
            "--cursor",
            cursor,
        ],
    );
    assert!(returned.status.success(), "{returned:?}");
    let returned: ReadDocument = serde_json::from_slice(&returned.stdout).unwrap();
    assert_eq!(returned.items, first.items);
    assert_eq!(returned.header, first.header);
}

#[test]
fn saved_query_retry_and_printed_undo_keep_immutable_results() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-saved");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("s1.jsonl"),
        line("u1", "s1", "a saved selection"),
    )
    .unwrap();
    let env = crate::Fixture::new(
        &dir.path().join("archive"),
        &home,
        &dir.path().join("config"),
    );
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["messages"] == 1);
    let args = [
        "--json",
        "--request-id",
        "save-cli",
        "sessions",
        "save",
        "my-history",
        "--watch",
        "--hold",
        "20m",
    ];
    let first = cli(&env, &args);
    assert!(first.status.success(), "{first:?}");
    let first = json(&first);
    let step = &first["reply"]["steps"][0];
    assert_eq!(step["outcome"]["status"], "done", "{first}");
    let report = &step["outcome"]["reply"]["reply"];
    assert_eq!(report["request"], "save-cli-1");
    assert_eq!(report["saved"]["id"], "save-cli-1.1");
    assert_eq!(report["saved"]["revision"], 1);
    assert_eq!(report["watch"]["id"], "save-cli-1.2");
    assert_eq!(report["kept"]["result"], "result:save-cli-1");
    assert_eq!(report["kept"]["rows"], 1);
    let undo = step["outcome"]["undo"].as_array().unwrap();
    assert_eq!(undo.len(), 1, "{first}");
    let retry = cli(&env, &args);
    assert!(retry.status.success(), "{retry:?}");
    let retry = json(&retry);
    let retried = &retry["reply"]["steps"][0]["outcome"];
    for field in ["saved", "kept", "watch"] {
        assert_eq!(retried["reply"]["reply"][field], report[field]);
    }
    assert_eq!(retried["reply"]["reply"]["repeat"], true);
    assert_eq!(retried["undo"], step["outcome"]["undo"]);
    let plain = cli(
        &env,
        &[
            "--request-id",
            "save-plain",
            "my-history",
            "save",
            "my-history",
        ],
    );
    assert!(plain.status.success(), "{plain:?}");
    let shown = format!(
        "{}{}",
        String::from_utf8_lossy(&plain.stdout),
        String::from_utf8_lossy(&plain.stderr)
    );
    assert!(shown.contains("same question; no new revision"), "{shown}");
    assert!(shown.contains("result:save-plain-1"), "{shown}");
    assert!(!shown.contains("undo:"), "{shown}");
    for _ in 0..2 {
        let withdrawn = printed_cli(&env, undo[0].as_str().unwrap());
        assert!(withdrawn.status.success(), "{withdrawn:?}");
    }
    for kind in ["saved_query", "signal"] {
        let out = cli(&env, &["--json", &format!("annotations where kind:{kind}")]);
        assert_eq!(
            out.status.code(),
            Some(1),
            "typed empty rows use exit 1: {out:?}"
        );
        assert_eq!(json(&out)["reply"]["count"], 0);
    }
    for reference in [
        "result:save-cli-1",
        "result:save-plain-1",
        "result:my-history",
    ] {
        let out = cli(&env, &["--json", reference]);
        assert!(out.status.success(), "{out:?}");
        assert_eq!(json(&out)["reply"]["count"], 1, "{out:?}");
    }
}

#[test]
fn resume_without_rows_uses_unfinished_here_and_explicit_targets_still_work() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let sources = home.join(".claude/projects/-resume");
    let here = dir.path().join("work");
    let elsewhere = dir.path().join("elsewhere");
    for path in [&sources, &here, &elsewhere] {
        std::fs::create_dir_all(path).unwrap();
    }
    for (id, cwd, closed) in [
        ("open", &here, false),
        ("closed", &here, true),
        ("elsewhere", &elsewhere, false),
    ] {
        let mut user: serde_json::Value =
            serde_json::from_str(&line(id, id, "resume candidate")).unwrap();
        user["cwd"] = serde_json::json!(cwd);
        let mut input = format!("{user}\n");
        if closed {
            input.push_str(&format!(
                "{}\n",
                serde_json::json!({
                    "type":"assistant", "uuid":"finished-reply", "sessionId":id,
                    "timestamp":"2026-03-01T10:00:01Z", "cwd":cwd,
                    "message":{"role":"assistant","content":[{"type":"text","text":"finished"}]}
                })
            ));
        }
        std::fs::write(sources.join(format!("{id}.jsonl")), input).unwrap();
    }
    let env = crate::Fixture::new(
        &dir.path().join("archive"),
        &home,
        &dir.path().join("config"),
    );
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["messages"] == 4);
    let run = |args: &[&str]| {
        command(&env)
            .current_dir(&here)
            .args(args)
            .output()
            .unwrap()
    };
    let automatic = run(&["--json", "--dry-run", "resume"]);
    assert!(automatic.status.success(), "{automatic:?}");
    let automatic = json(&automatic);
    let would = automatic["reply"]["steps"][0]["outcome"]["would"]
        .as_str()
        .unwrap();
    assert!(would.contains("claude-code:open"), "{automatic}");
    assert!(!would.contains("claude-code:closed"), "{automatic}");
    assert!(!would.contains("claude-code:elsewhere"), "{automatic}");
    let explicit = run(&[
        "--json",
        "--dry-run",
        "session:claude-code:closed",
        "resume",
    ]);
    assert!(explicit.status.success(), "{explicit:?}");
    let explicit = json(&explicit);
    let would = explicit["reply"]["steps"][0]["outcome"]["would"]
        .as_str()
        .unwrap();
    assert!(would.contains("claude-code:closed"), "{explicit}");
    assert!(would.contains("--resume"), "{explicit}");
}

#[test]
fn grammar_and_feed_failures_show_the_correction_and_exit_two() {
    let (_dir, env, _daemon) = two_session_cli();
    let output = cli(&env, &["read", "claude-code:ID"]);
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        concat!(
            "polylogue: not_a_verb: \"claude-code:ID\" is not a verb; read takes no words\n",
            "polylogue: try: polylogue claude-code:ID read\n"
        )
    );
    let output = cli(&env, &["dev", "feed", "--after", "bogus"]);
    assert_eq!(output.status.code(), Some(2));
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&output.stdout).unwrap();
    let failure = answer.failure().unwrap();
    assert_eq!(failure.code.as_deref(), Some("bad_value"));
    assert_eq!(failure.kind, command_surface::FailureKind::BadValue);
    assert_eq!(
        failure.message,
        "after takes a position GENERATION:INDEX:ARCHIVE"
    );
}

#[test]
fn streamed_read_deadline_prints_its_hint_and_next_command() {
    let (_dir, env, _daemon) = two_session_cli();
    let output = cli(
        &env,
        &[
            "session:claude-code:s1",
            "read",
            "--follow",
            "--deadline",
            "200",
        ],
    );
    assert_eq!(output.status.code(), Some(7));
    let text = String::from_utf8(output.stderr).unwrap();
    assert_eq!(
        text.lines()
            .find(|line| line.starts_with("polylogue: hint:")),
        Some("polylogue: hint: rerun with `--deadline none`, or narrow to one session")
    );
    let next = text
        .lines()
        .find_map(|line| line.strip_prefix("polylogue: complete without a deadline: "))
        .unwrap();
    let words: Vec<_> = next.split_whitespace().collect();
    assert_eq!(&words[..2], &["polylogue", "--request-id"]);
    assert_eq!(
        &words[3..],
        &[
            "--deadline",
            "none",
            "--",
            "session:claude-code:s1",
            "read",
            "--follow"
        ]
    );
}
