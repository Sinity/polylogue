//! A real daemon over a temporary archive and home for in-process tests:
//! its loop reads only the temporary home, and its socket is served on a
//! thread of its own.

use crate::daemon::server::{self, Daemon, handle};
use crate::daemon::{Request, Response};
use polylogue_engine::Config;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct World {
    pub _dir: tempfile::TempDir,
    pub home: PathBuf,
    pub daemon: Arc<Daemon>,
    _lock: polylogue_store::ArchiveLock,
}

pub fn claude_line(uuid: &str, session: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({
            "type": "user", "uuid": uuid, "sessionId": session,
            "timestamp": "2026-03-01T10:00:00Z",
            "message": {"role": "user", "content": text}
        })
    )
}

/// A daemon whose loop has parsed one Claude Code session, `claude-code:s1`,
/// holding `text`, with its socket served.
pub fn world(text: &str) -> World {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-p");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(project.join("s1.jsonl"), claude_line("u1", "s1", text)).unwrap();
    let mut config = Config::from_env();
    config.home = home.clone();
    config.tool_dirs.clear();
    config.workers = 2;
    let (daemon, lock) = server::start(&dir.path().join("archive"), config).unwrap();
    let listener = server::bind(&daemon).unwrap();
    let d = Arc::clone(&daemon);
    std::thread::spawn(move || server::listen(&listener, &d));
    let world = World {
        _dir: dir,
        home,
        daemon,
        _lock: lock,
    };
    world.wait_for(|data| data["store"]["messages"].as_u64() == Some(1));
    world
}

impl World {
    /// Waits until `status` satisfies `ok`.
    pub fn wait_for(&self, ok: impl Fn(&serde_json::Value) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Response::Ok { data } = handle(Request::Status, &self.daemon)
                && ok(&data)
            {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "the daemon never reached the state"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Sets an archive setting through the daemon.
    pub fn set(&self, key: &str, value: &str) {
        let r = handle(
            Request::Setting {
                request_id: None,
                key: key.into(),
                value: Some(value.into()),
                machine: false,
            },
            &self.daemon,
        );
        assert!(matches!(r, Response::Ok { .. }), "{r:?}");
    }
}
