//! A real daemon over a temporary archive and home for in-process tests:
//! its loop reads only the temporary home, and its socket is served on a
//! thread of its own.

use crate::daemon::{Request, Response};
use crate::daemon::{
    server,
    service::{self, Daemon, handle},
};
use polylogue_engine::Config;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct TestDaemon {
    pub home: PathBuf,
    pub daemon: Arc<Daemon>,
    server: server::Server,
    runtime: service::Runtime,
    // Removed only after both owners have joined their workers.
    pub _dir: tempfile::TempDir,
}

/// A daemon whose loop has parsed one Claude Code session, `claude-code:s1`,
/// holding `text`, with its first derivation pass finished and socket served.
pub fn start(text: &str) -> TestDaemon {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-p");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("s1.jsonl"),
        polylogue_testkit::claude_user("u1", "s1", text),
    )
    .unwrap();
    let mut config = Config::from_env();
    config.home = home.clone();
    config.tool_dirs.clear();
    config.workers = 2;
    let runtime = service::start(
        &dir.path().join("archive"),
        config,
        service::Backing::Standalone,
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let server = server::Server::start(Arc::clone(&daemon)).unwrap();
    let world = TestDaemon {
        _dir: dir,
        home,
        daemon,
        server,
        runtime,
    };
    world.wait_for(|data| {
        data["store"]["messages"].as_u64() == Some(1)
            && data["store"]["search_pending"].as_u64() == Some(0)
            && data["last_pass"]["seconds"]
                .as_f64()
                .is_some_and(|seconds| seconds > 0.0)
    });
    world
}

impl TestDaemon {
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

impl Drop for TestDaemon {
    fn drop(&mut self) {
        self.runtime.stop_admission();
        if let Err(error) = self.server.shutdown() {
            eprintln!("test daemon socket: {error}");
        }
        if let Err(error) = self.runtime.shutdown() {
            eprintln!("test daemon runtime: {error}");
        }
    }
}

pub mod assets;
