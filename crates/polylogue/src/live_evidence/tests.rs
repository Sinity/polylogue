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

/// law: pl.live.sampler-proportional
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

/// law: pl.live.pid-reuse
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

/// law: pl.live.root-gone-prompt
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
