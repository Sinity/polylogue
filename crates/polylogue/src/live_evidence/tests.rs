use super::*;
use polylogue_store::LiveRoot;

fn now() -> jiff::Timestamp {
    jiff::Timestamp::from_second(1_790_000_000).unwrap()
}

/// A fake `/proc` with processes `(pid, ppid, start, comm)`; each process's
/// children are listed in its one task's `children` file.
fn fake_proc(dir: &Path, processes: &[(u32, u32, u64, &str)]) {
    let _ = std::fs::remove_dir_all(dir);
    for (pid, ppid, start, comm) in processes {
        let base = dir.join(pid.to_string());
        std::fs::create_dir_all(base.join("task").join(pid.to_string())).unwrap();
        let mut fields = vec!["S".to_owned(), ppid.to_string()];
        fields.extend((5..22).map(|_| "0".to_owned()));
        fields.push(start.to_string());
        std::fs::write(
            base.join("stat"),
            format!("{pid} ({comm}) {}", fields.join(" ")),
        )
        .unwrap();
        let children: Vec<String> = processes
            .iter()
            .filter(|p| p.1 == *pid)
            .map(|p| p.0.to_string())
            .collect();
        std::fs::write(
            base.join("task").join(pid.to_string()).join("children"),
            children.join(" "),
        )
        .unwrap();
        std::fs::write(base.join("cmdline"), format!("{comm}\0--flag\0")).unwrap();
    }
}

fn root(session: &str, pid: u32, start: u64, boot: &str) -> LiveRoot {
    LiveRoot {
        session: format!("claude-code:{session}"),
        root: ProcessIdentity {
            boot: boot.into(),
            pid,
            start,
        },
        open_call: true,
    }
}

fn events(lines: &[Value]) -> Vec<(String, u64)> {
    lines
        .iter()
        .map(|l| {
            (
                l["event"].as_str().unwrap().to_owned(),
                l["payload"]["pid"].as_u64().unwrap(),
            )
        })
        .collect()
}

#[test]
fn stat_lines_with_spaces_and_parentheses_in_the_name_parse() {
    let line = "41 (tmux: server (1)) S 7 41 41 0 -1 4194560 1 0 0 0 0 0 0 0 20 0 1 0 98765 0 0";
    assert_eq!(
        parse_stat(line),
        Some(Stat {
            ppid: 7,
            start: 98765,
            comm: "tmux: server (1)".into(),
            zombie: false,
        })
    );
}

#[test]
fn the_watcher_writes_only_changes_and_reads_nothing_without_roots() {
    let dir = tempfile::tempdir().unwrap();
    let proc_root = dir.path().join("proc");
    // No roots: nothing is read, even from a `/proc` that does not exist.
    let mut w = Watcher::new(dir.path().join("absent"), "b1".into(), Some(0));
    assert!(w.update_roots(&[], now()).is_empty());
    assert!(w.sample(now()).is_empty());
    fake_proc(
        &proc_root,
        &[
            (10, 1, 500, "claude"),
            (11, 10, 600, "bash"),
            (12, 11, 700, "cargo"),
        ],
    );
    let mut w = Watcher::new(proc_root.clone(), "b1".into(), Some(0));
    let roots = [root("s", 10, 500, "b1")];
    assert_eq!(
        events(&w.update_roots(&roots, now())),
        [("Started".into(), 10)]
    );
    assert_eq!(
        events(&w.sample(now())),
        [("Started".into(), 11), ("Started".into(), 12)]
    );
    // The child carries argv and cwd as read; environment is never read.
    // A sample with no change writes nothing.
    w.watched.get_mut("claude-code:s").unwrap().next_sample = Instant::now();
    assert!(w.sample(now()).is_empty());
    // A child that exits is written once.
    fake_proc(&proc_root, &[(10, 1, 500, "claude"), (11, 10, 600, "bash")]);
    w.watched.get_mut("claude-code:s").unwrap().next_sample = Instant::now();
    assert_eq!(events(&w.sample(now())), [("Ended".into(), 12)]);
    // The same roots again add nothing.
    assert!(w.update_roots(&roots, now()).is_empty());
}

#[test]
fn a_root_from_another_boot_or_start_time_is_written_gone() {
    let dir = tempfile::tempdir().unwrap();
    let proc_root = dir.path().join("proc");
    fake_proc(&proc_root, &[(10, 1, 999, "other")]);
    let mut w = Watcher::new(proc_root, "b2".into(), Some(1_700_000_000_000));
    let lines = w.update_roots(
        &[root("old", 10, 500, "b1"), root("reused", 10, 500, "b2")],
        now(),
    );
    assert_eq!(
        events(&lines),
        [("RootGone".into(), 10), ("RootGone".into(), 10)]
    );
    // An earlier boot's root ended at that boot's end, not now.
    assert_eq!(lines[0]["at_ns"], "1700000000000000000");
    assert_eq!(lines[0]["route"], "process");
    assert_eq!(lines[0]["session_id"], "old");
    assert!(w.watched.is_empty());
    // Until the evidence stops naming them, gone roots are not written again.
    assert!(
        w.update_roots(
            &[root("old", 10, 500, "b1"), root("reused", 10, 500, "b2")],
            now()
        )
        .is_empty()
    );
}

#[test]
fn an_exited_root_waiting_to_be_reaped_is_gone() {
    let dir = tempfile::tempdir().unwrap();
    let proc_root = dir.path().join("proc");
    fake_proc(&proc_root, &[(10, 1, 500, "claude")]);
    let stat = proc_root.join("10").join("stat");
    let text = std::fs::read_to_string(&stat)
        .unwrap()
        .replace(") S ", ") Z ");
    std::fs::write(&stat, text).unwrap();
    let mut w = Watcher::new(proc_root, "b1".into(), None);
    let roots = [root("s", 10, 500, "b1")];
    assert_eq!(
        events(&w.update_roots(&roots, now())),
        [("RootGone".into(), 10)]
    );
    assert!(w.update_roots(&roots, now()).is_empty());
}

/// A failed spool open must not consume an exit observation or change its time.
#[test]
fn root_gone_retries_after_the_process_spool_becomes_writable() {
    for already_gone in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let proc_root = dir.path().join("proc");
        let spool = dir.path().join("spool");
        fake_proc(&proc_root, &[(10, 1, 500, "claude")]);
        let mut watcher = Watcher::new(proc_root.clone(), "b1".into(), None);
        let roots = [root("s", 10, 500, "b1")];
        if !already_gone {
            let started = watcher.update_roots(&roots, now());
            watcher.append_process(&spool, &started).unwrap();
        }
        fake_proc(&proc_root, &[]);
        let observed = if already_gone {
            watcher.update_roots(&roots, now())
        } else {
            watcher.exited(&BTreeSet::from(["claude-code:s".into()]), now())
        };
        assert_eq!(events(&observed), [("RootGone".into(), 10)]);
        let process = spool.join("process");
        std::fs::create_dir_all(&spool).unwrap();
        if !already_gone {
            std::fs::rename(&process, spool.join("saved-process")).unwrap();
        }
        std::fs::write(&process, b"blocked").unwrap();
        assert!(watcher.append_process(&spool, &observed).is_err());
        assert!(!watcher.pending.is_empty());
        std::fs::remove_file(&process).unwrap();
        let later = jiff::Timestamp::from_second(now().as_second() + 10).unwrap();
        let repeated = watcher.update_roots(&roots, later);
        assert!(repeated.is_empty());
        watcher.append_process(&spool, &repeated).unwrap();
        assert!(watcher.pending.is_empty());
        assert!(watcher.exited(&BTreeSet::new(), later).is_empty());
        watcher.append_process(&spool, &[]).unwrap();
        let bytes = std::fs::read(process.join("b1.ndjson")).unwrap();
        let lines: Vec<Value> = serde_json::Deserializer::from_slice(&bytes)
            .into_iter()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(lines, observed);
    }
}

#[test]
fn a_partial_process_spool_write_retries_only_its_unwritten_suffix() {
    struct FullAfterPrefix {
        file: std::fs::File,
        remaining: usize,
        interrupted: bool,
    }
    impl Write for FullAfterPrefix {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if std::mem::take(&mut self.interrupted) {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            if self.remaining == 0 {
                return Err(std::io::Error::from_raw_os_error(28));
            }
            let n = self.file.write(&bytes[..bytes.len().min(self.remaining)])?;
            self.remaining -= n;
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.file.flush()
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let spool = dir.path().join("spool");
    std::fs::write(&spool, b"blocked").unwrap();
    let mut watcher = Watcher::new(dir.path().join("absent"), "b1".into(), None);
    let observed = watcher.update_roots(&[root("s", 10, 500, "b1")], now());
    assert!(watcher.append_process(&spool, &observed).is_err());
    let expected = watcher.pending.clone();
    std::fs::remove_file(&spool).unwrap();
    let mut writer = FullAfterPrefix {
        file: spool_file(&spool, "process", "b1").unwrap(),
        remaining: 13,
        interrupted: true,
    };
    assert!(write_pending(&mut watcher.pending, &mut writer).is_err());
    assert_eq!(watcher.pending, expected[13..]);
    drop(writer);
    let suffix = watcher.pending.clone();
    let mut full: &mut [u8] = &mut [];
    let error = write_pending(&mut watcher.pending, &mut full).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::WriteZero);
    assert_eq!(watcher.pending, suffix);
    watcher.append_process(&spool, &[]).unwrap();
    watcher.append_process(&spool, &[]).unwrap();
    let bytes = std::fs::read(spool.join("process/b1.ndjson")).unwrap();
    assert_eq!(bytes, expected);
    let lines: Vec<Value> = serde_json::Deserializer::from_slice(&bytes)
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(lines, observed);
}

#[test]
fn a_root_exit_is_seen_through_its_pidfd_at_once() {
    let mut child = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap();
    let pid = child.id();
    let me = identity(pid).expect("a live child has an identity");
    let mut w = Watcher::new(PathBuf::from("/proc"), me.boot.clone(), None);
    let roots = [LiveRoot {
        session: "claude-code:child".into(),
        root: me,
        open_call: false,
    }];
    assert_eq!(
        events(&w.update_roots(&roots, now())),
        [("Started".into(), u64::from(pid))]
    );
    assert!(w.wait(Duration::from_millis(20)).is_empty());
    child.kill().unwrap();
    let started = Instant::now();
    let ready = w.wait(Duration::from_secs(5));
    assert!(started.elapsed() < Duration::from_millis(100));
    child.wait().unwrap();
    let gone = w.exited(&ready, now());
    assert_eq!(events(&gone), [("RootGone".into(), u64::from(pid))]);
}

#[test]
fn the_registry_reader_writes_each_change_once_with_the_file_whole() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("4242.json");
    let write = |status: &str| {
        std::fs::write(
            &file,
            serde_json::to_vec(
                &json!({"pid": 4242, "sessionId": "s9", "status": status, "kind": "interactive"}),
            )
            .unwrap(),
        )
        .unwrap();
        // A distinct modification time for each write.
        let f = std::fs::File::options().write(true).open(&file).unwrap();
        let later = SystemTime::now()
            + Duration::from_secs(match status {
                "busy" => 1,
                _ => 2,
            });
        f.set_modified(later).unwrap();
    };
    let mut r = Registry::new(dir.path().to_path_buf());
    let identity = |pid| {
        Some(ProcessIdentity {
            boot: "b1".into(),
            pid,
            start: 77,
        })
    };
    write("busy");
    let lines = r.read(now(), identity);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["route"], "registry");
    assert_eq!(lines[0]["provider"], "claude-code");
    assert_eq!(lines[0]["session_id"], "s9");
    assert_eq!(lines[0]["payload"]["status"], "busy");
    assert_eq!(lines[0]["payload"]["start"], 77);
    assert_eq!(lines[0]["payload"]["registry"]["kind"], "interactive");
    assert!(r.read(now(), identity).is_empty(), "no change, no line");
    write("blocked");
    assert_eq!(r.read(now(), identity)[0]["payload"]["status"], "blocked");
}

/// A stand-in for `codex app-server proxy`: answers `thread/loaded/list`
/// from the file `threads` and `thread/read` from the file `status` beside
/// it, and every other request with an empty result.
fn fake_codex(dir: &Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let program = dir.join("codex");
    std::fs::write(
        &program,
        r#"#!/bin/sh
dir=$(dirname "$0")
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  [ -z "$id" ] && continue
  case "$line" in
    *'"thread/loaded/list"'*) printf '{"id":%s,"result":{"data":%s,"nextCursor":null}}\n' "$id" "$(cat "$dir/threads")" ;;
    *'"thread/read"'*) printf '{"id":%s,"result":{"thread":{"status":%s}}}\n' "$id" "$(cat "$dir/status")" ;;
    *) printf '{"id":%s,"result":{}}\n' "$id" ;;
  esac
done
"#,
    )
    .unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
    program
}

/// The Codex status reader writes one registry line per thread status
/// change, the status object whole, `notLoaded` when a thread leaves the
/// loaded list, and nothing while the control socket is absent.
#[test]
fn the_codex_reader_writes_each_thread_status_change_once() {
    let dir = tempfile::tempdir().unwrap();
    let program = fake_codex(dir.path());
    let socket = dir.path().join("app-server-control.sock");
    let thread = "0199a1b2-3c4d-7e8f-9012-345678901234";
    let set = |threads: &str, status: &str| {
        std::fs::write(dir.path().join("threads"), threads).unwrap();
        std::fs::write(dir.path().join("status"), status).unwrap();
    };
    let words = |lines: &[Value]| -> Vec<(String, String)> {
        lines
            .iter()
            .map(|l| {
                assert_eq!(l["route"], "registry");
                assert_eq!(l["provider"], "codex");
                (
                    l["session_id"].as_str().unwrap().to_owned(),
                    l["payload"]["status"].as_str().unwrap().to_owned(),
                )
            })
            .collect()
    };
    let mut reader = CodexStatus::new(program, socket.clone());
    set(&format!(r#"["{thread}"]"#), r#"{"type":"idle"}"#);
    // No control socket: no connection and no lines.
    assert!(reader.read(now()).is_empty());
    std::fs::write(&socket, "").unwrap();
    assert_eq!(
        words(&reader.read(now())),
        vec![(thread.to_owned(), "idle".to_owned())]
    );
    assert!(reader.read(now()).is_empty(), "no change, no line");
    set(
        &format!(r#"["{thread}"]"#),
        r#"{"type":"active","activeFlags":["waitingOnApproval"]}"#,
    );
    let lines = reader.read(now());
    assert_eq!(
        words(&lines),
        vec![(thread.to_owned(), "waitingOnApproval".to_owned())]
    );
    assert_eq!(
        lines[0]["payload"]["registry"]["status"]["activeFlags"][0],
        "waitingOnApproval"
    );
    set("[]", r#"{"type":"idle"}"#);
    assert_eq!(
        words(&reader.read(now())),
        vec![(thread.to_owned(), "notLoaded".to_owned())]
    );
    std::fs::remove_file(&socket).unwrap();
    assert!(reader.read(now()).is_empty());
}

/// The live clock announces a session once its instant from the previous
/// reading has passed. The readings come from the real view, which only
/// ever gives instants after the time it is read at.
#[test]
fn the_live_clock_announces_each_transition_once() {
    use polylogue_model::live::{LIVE_WINDOW_MS, LiveFacts, fold, view};
    let mut facts = LiveFacts::default();
    // Hook evidence with no process: working for the live window, then stale.
    fold(&mut facts, "hook_user_prompt_submit", 0, &json!({}));
    let reading = |now: i64| -> Vec<(String, i64)> {
        view(Some(&facts), None, now)
            .until
            .map(|u| ("claude-code:a".to_owned(), u))
            .into_iter()
            .collect()
    };
    // One step of the worker: take what passed, then remember a reading.
    let mut clock = LiveClock::default();
    let mut step = |now: i64| {
        let due = clock.take_due(now);
        clock.remember(&reading(now));
        (due, clock.wait(now))
    };
    let (due, wait) = step(1_000);
    assert!(due.is_empty());
    assert_eq!(
        wait,
        Some(Duration::from_millis((LIVE_WINDOW_MS + 1 - 1_000) as u64))
    );
    // Before the instant: nothing.
    assert!(step(LIVE_WINDOW_MS).0.is_empty());
    // After it: the session is announced once.
    let after = LIVE_WINDOW_MS + 5;
    assert_eq!(step(after).0, ["claude-code:a"]);
    assert!(step(after + 10).0.is_empty());
}

fn git_in(repo: &Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args([
            "-c",
            "user.name=Example",
            "-c",
            "user.email=example@example.invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A commit in a session's worktree that contains files of its footprint
/// writes one line naming exactly those files; a file the commit does not
/// contain stays, and a step with no commit writes nothing.
#[test]
fn a_commit_in_the_worktree_names_the_footprint_files_it_contains() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    git_in(&repo, &["init", "--initial-branch=main"]);
    git_in(&repo, &["commit", "--allow-empty", "-m", "root"]);
    let a = repo.join("src/a.rs");
    let b = repo.join("src/b.rs");
    std::fs::write(&a, "a").unwrap();
    std::fs::write(&b, "b").unwrap();
    let footprint = polylogue_store::LiveFootprint {
        session: "claude-code:edits".into(),
        cwd: repo.join("src").display().to_string(),
        // One absolute path, one relative to the working directory.
        files: vec![a.display().to_string(), "b.rs".into()],
    };
    let mut watcher = CommitWatcher::default();
    // First sight records HEAD; nothing has been committed since.
    assert!(
        watcher
            .read(std::slice::from_ref(&footprint), now())
            .is_empty()
    );
    assert!(
        watcher
            .read(std::slice::from_ref(&footprint), now())
            .is_empty()
    );
    git_in(&repo, &["add", "src/a.rs"]);
    git_in(&repo, &["commit", "-m", "a"]);
    let lines = watcher.read(std::slice::from_ref(&footprint), now());
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["route"], "git");
    assert_eq!(lines[0]["event"], "Commit");
    assert_eq!(
        lines[0]["payload"]["files"],
        json!([a.display().to_string()])
    );
    assert!(
        watcher
            .read(std::slice::from_ref(&footprint), now())
            .is_empty()
    );
    git_in(&repo, &["add", "src/b.rs"]);
    git_in(&repo, &["commit", "-m", "b"]);
    let lines = watcher.read(std::slice::from_ref(&footprint), now());
    assert_eq!(lines[0]["payload"]["files"], json!(["b.rs"]));
    // The fold takes the named files out of the footprint.
    let mut facts = polylogue_model::live::LiveFacts::default();
    facts.footprint.insert("b.rs".into());
    facts.footprint.insert("c.rs".into());
    polylogue_model::live::fold(&mut facts, "git_commit", 1, &lines[0]["payload"]);
    assert_eq!(facts.footprint.into_iter().collect::<Vec<_>>(), ["c.rs"]);
}

#[test]
fn commit_lists_parse_into_commits_and_files() {
    let text = "\u{0}aaa 1790000000\n\nsrc/a.rs\nsrc/b.rs\n\u{0}bbb 1790000060\n\n";
    assert_eq!(
        parse_commits(text),
        vec![
            (
                "aaa".to_owned(),
                1_790_000_000,
                vec!["src/a.rs".to_owned(), "src/b.rs".to_owned()]
            ),
            ("bbb".to_owned(), 1_790_000_060, vec![])
        ]
    );
}
