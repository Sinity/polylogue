//! End to end: the real binary as a daemon over a temporary archive and
//! home directory, driven through the CLI.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_polylogue");

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn cli(env: &[(&str, &Path)], args: &[&str]) -> Output {
    let mut c = Command::new(BIN);
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
    ];

    let mut child = Command::new(BIN)
        .arg("daemon")
        .env("POLYLOGUE_ARCHIVE", &archive)
        .env("POLYLOGUE_HOME", &home)
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stderr = BufReader::new(child.stderr.take().unwrap());
    let mut first = String::new();
    stderr.read_line(&mut first).unwrap();
    assert!(first.contains("serving"), "{first}");
    let daemon = Daemon(child);

    // The live file is picked up by the loop's first pass.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let out = cli(&env, &["--json", "status"]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        if json(&out)["store"]["messages"].as_u64() == Some(1) {
            break;
        }
        assert!(Instant::now() < deadline, "the live file was never parsed");
        std::thread::sleep(Duration::from_millis(50));
    }

    let out = cli(&env, &["--json", "import", export.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(json(&out)["imported"].as_array().unwrap().len(), 1);

    let out = cli(&env, &["--json", "list"]);
    let ids: Vec<String> = json(&out)
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
