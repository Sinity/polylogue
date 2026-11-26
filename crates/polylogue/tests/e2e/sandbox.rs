//! Daemons in the testkit's sandbox under a parent whose environment points
//! at a marker directory full of the tools' files: the daemon reads none of
//! them and records nothing from them (docs/testing.md, Scenarios), and
//! `cargo xtask run` serves its scratch archive and leaves no process.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The port a daemon listens on when nothing names another.
const DEFAULT_PORT: u16 = 8765;

fn session_line(uuid: &str, session: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({
            "type": "user", "uuid": uuid, "sessionId": session,
            "timestamp": "2026-03-01T10:00:00Z",
            "message": {"role": "user", "content": "synthetic"}
        })
    )
}

/// A home holding every kind of file the daemon would find there: Claude
/// Code and Codex sessions, a Hermes home, Polylogue's spool and a
/// deployment that names an archive inside it.
fn marker_home(marker: &Path) {
    let files = [
        (
            ".claude/projects/-p/outside.jsonl",
            session_line("o1", "outside"),
        ),
        (
            ".codex/sessions/2026/03/01/rollout-2026-03-01T10-00-00-outside.jsonl",
            polylogue_testkit::codex::observations("outside"),
        ),
        (".hermes/state.db", "synthetic".into()),
        (".local/state/polylogue/spool/outside", "synthetic".into()),
        (
            ".config/polylogue/polylogue.toml",
            format!(
                "archive = {}\n",
                serde_json::Value::from(marker.join("archive").to_str().unwrap())
            ),
        ),
    ];
    for (path, text) in files {
        let path = marker.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    for directory in [".local/share", ".cache", ".runtime"] {
        std::fs::create_dir_all(marker.join(directory)).unwrap();
    }
}

/// Every path under `root` with its bytes (none for a directory).
fn tree(root: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    let mut entries = BTreeMap::new();
    let mut pending = vec![root.to_owned()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            let relative = path.strip_prefix(root).unwrap().to_owned();
            if path.is_dir() {
                entries.insert(relative, None);
                pending.push(path);
            } else {
                entries.insert(relative, Some(std::fs::read(&path).unwrap()));
            }
        }
    }
    entries
}

/// The parent's variables that would lead a daemon into `marker`.
fn marker_environment(marker: &Path) -> Vec<(&'static str, PathBuf)> {
    vec![
        ("HOME", marker.to_owned()),
        ("CLAUDE_CONFIG_DIR", marker.join(".claude")),
        ("CODEX_HOME", marker.join(".codex")),
        ("HERMES_HOME", marker.join(".hermes")),
        ("XDG_CONFIG_HOME", marker.join(".config")),
        ("XDG_DATA_HOME", marker.join(".local/share")),
        ("XDG_STATE_HOME", marker.join(".local/state")),
        ("XDG_CACHE_HOME", marker.join(".cache")),
        ("XDG_RUNTIME_DIR", marker.join(".runtime")),
        (
            "POLYLOGUE_LISTEN",
            format!("127.0.0.1:{DEFAULT_PORT}").into(),
        ),
    ]
}

/// Runs [`daemon_under_a_marker_parent`] in a child of this test whose
/// environment holds nothing but the marker's variables, and checks that
/// the marker directory is as it was.
fn under_a_marker_parent() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("marker");
    marker_home(&marker);
    let before = tree(&marker);
    let tmp = dir.path().join("tmp");
    std::fs::create_dir(&tmp).unwrap();
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "sandbox::daemon_under_a_marker_parent",
            "--ignored",
            "--nocapture",
        ])
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("TMPDIR", &tmp)
        .envs(marker_environment(&marker))
        .env("POLYLOGUE_SANDBOX_TEST_ROOT", dir.path().join("sandbox"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // The child ran its test rather than finding none to run.
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed"),
        "{output:?}"
    );
    assert_eq!(tree(&marker), before);
}

#[test]
fn a_sandboxed_daemon_ignores_the_parent_environment() {
    under_a_marker_parent();
}

#[test]
fn a_sandboxed_daemon_listens_on_its_own_port_while_the_default_one_is_taken() {
    // Held for the whole test; a daemon of the host may hold it instead.
    let _default = match std::net::TcpListener::bind(("127.0.0.1", DEFAULT_PORT)) {
        Ok(listener) => Some(listener),
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => None,
        Err(error) => panic!("binding the default port: {error}"),
    };
    under_a_marker_parent();
}

#[test]
#[ignore = "child of the sandbox tests, run under a marker environment"]
fn daemon_under_a_marker_parent() {
    let Some(root) = std::env::var_os("POLYLOGUE_SANDBOX_TEST_ROOT") else {
        return;
    };
    let root = PathBuf::from(root);
    let home = root.join("home");
    let project = home.join(".claude/projects/-p");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(project.join("inside.jsonl"), session_line("i1", "inside")).unwrap();
    let env = crate::Fixture::new(&root.join("archive"), &home, &root.join("config"));
    let port = env.sandbox().listen().port();
    assert_ne!(port, DEFAULT_PORT);
    let mut command = env.command();
    command.arg("daemon").stderr(Stdio::piped());
    let mut daemon = crate::cli::Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap());
    // The daemon reports its web reader right after it starts serving.
    let mut stderr = BufReader::new(daemon.0.stderr.take().unwrap());
    let mut web = String::new();
    loop {
        web.clear();
        assert_ne!(stderr.read_line(&mut web).unwrap(), 0, "the daemon stopped");
        if web.contains("polylogue: web reader") {
            break;
        }
    }
    assert_eq!(
        web.trim_end(),
        format!("<6>polylogue: web reader on http://127.0.0.1:{port}/")
    );
    std::thread::spawn(move || std::io::copy(&mut stderr, &mut std::io::sink()));
    let mut http = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    http.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .unwrap();
    let mut reply = String::new();
    http.read_to_string(&mut reply).unwrap();
    assert!(reply.starts_with("HTTP/1.1 "), "{reply}");

    let status = crate::cli::wait_status(&env, |s| s["store"]["messages"].as_u64() > Some(0));
    assert_eq!(status["archive"], root.join("archive").to_str().unwrap());
    assert_eq!(status["store"]["inputs"], 1, "{status}");
    assert_eq!(status["store"]["sessions"], 1, "{status}");
    assert_eq!(status["store"]["messages"], 1, "{status}");
    let rows = crate::cli::query_rows(&crate::cli::cli(&env, &["--json", "sessions"]));
    let ids: Vec<_> = rows["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["claude-code:inside"]);
}

/// `cargo xtask run` under a parent whose tools' variables point at the
/// marker: the CLI script it prints answers `status` from the scratch
/// archive, and stopping the command leaves no process and frees the
/// archive.
#[test]
fn cargo_xtask_run_serves_its_scratch_archive_and_stops_with_its_daemon() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("marker");
    marker_home(&marker);
    let before = tree(&marker);
    let root = dir.path().join("run");
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut xtask = Command::new(env!("CARGO"));
    // Cargo keeps the build the test runner already made. The parent keeps
    // what Cargo needs (its home and the toolchain) and gains the tools'
    // variables and the default port, but not this package's context.
    for (name, _) in std::env::vars_os() {
        let name = name.to_string_lossy();
        if name == "CARGO_MANIFEST_DIR"
            || name == "CARGO_MANIFEST_PATH"
            || name.starts_with("CARGO_PKG_")
        {
            xtask.env_remove(&*name);
        }
    }
    xtask
        .current_dir(&checkout)
        .args(["run", "--locked", "--quiet", "-p", "xtask", "--", "run"])
        .arg("--binary")
        .arg(crate::BIN)
        .arg("--root")
        .arg(&root)
        .envs(marker_environment(&marker).into_iter().filter(|(name, _)| {
            matches!(
                *name,
                "CLAUDE_CONFIG_DIR" | "CODEX_HOME" | "HERMES_HOME" | "POLYLOGUE_LISTEN"
            )
        }))
        .stdin(Stdio::null())
        .stdout(Stdio::piped());
    let mut xtask = crate::cli::Daemon(xtask.spawn().unwrap());
    let mut stdout = BufReader::new(xtask.0.stdout.take().unwrap());
    let mut printed = BTreeMap::new();
    while !printed.contains_key("cli") {
        let mut line = String::new();
        assert_ne!(
            stdout.read_line(&mut line).unwrap(),
            0,
            "xtask run stopped: {printed:?}"
        );
        if let Some((key, value)) = line.trim_end().split_once(": ") {
            printed.insert(key.to_owned(), value.to_owned());
        }
    }
    let daemon: i32 = printed["daemon"].parse().unwrap();
    assert_eq!(printed["archive"], root.join("archive").to_str().unwrap());
    let status = Command::new(&printed["cli"])
        .args(["--deadline", "none", "--json", "status", "--wait", "60s"])
        .output()
        .unwrap();
    assert!(status.status.success(), "{status:?}");
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&status.stdout).unwrap();
    let status = answer.value().unwrap();
    assert_eq!(status["pid"], daemon);
    assert_eq!(status["archive"], printed["archive"]);
    assert_eq!(status["caught_up"], true, "{status}");
    assert_eq!(status["store"]["inputs"], 0, "{status}");

    // SAFETY: kill takes scalar arguments.
    #[allow(unsafe_code)]
    let signalled = unsafe { libc::kill(xtask.0.id() as libc::pid_t, libc::SIGTERM) };
    assert_eq!(signalled, 0);
    xtask.0.wait().unwrap();
    // The command reaps its daemon before it exits: a daemon or a process
    // of its group still there outlived the command.
    for target in [daemon, -daemon] {
        // SAFETY: kill takes scalar arguments; signal 0 only checks.
        #[allow(unsafe_code)]
        let found = unsafe { libc::kill(target, 0) };
        assert_eq!(
            (found, std::io::Error::last_os_error().raw_os_error()),
            (-1, Some(libc::ESRCH)),
            "process {target} outlived xtask run"
        );
    }
    let (_archive, _lock) = polylogue_store::Archive::open_or_create(&root.join("archive"))
        .expect("the stopped daemon released its archive");
    assert_eq!(tree(&marker), before);
}
