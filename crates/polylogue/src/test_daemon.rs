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
use std::time::Duration;

pub struct TestDaemon {
    pub home: PathBuf,
    /// The loop's settings the daemon was started with.
    pub config: Config,
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
        config.clone(),
        service::ArchiveMode::Standalone,
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let server = server::Server::start(Arc::clone(&daemon)).unwrap();
    let daemon = TestDaemon {
        _dir: dir,
        home,
        config,
        daemon,
        server,
        runtime,
    };
    daemon.wait_for(|data| {
        data["store"]["messages"].as_u64() == Some(1)
            && data["store"]["search_pending"].as_u64() == Some(0)
    });
    daemon
}

impl TestDaemon {
    /// Waits until `status` satisfies `ok` and then until the loop is idle,
    /// so the pass that made `ok` hold has also derived from its rows. A
    /// state that never comes is ended by the test runner's timeout.
    pub fn wait_for(&self, ok: impl Fn(&serde_json::Value) -> bool) {
        let status = || match handle(Request::Status, &self.daemon) {
            Response::Ok { data, .. } => data,
            other => panic!("status failed: {other:?}"),
        };
        while !ok(&status()) {
            std::thread::sleep(Duration::from_millis(10));
        }
        // Checked after `ok`: an idle loop seen now has finished any pass
        // that wrote what `ok` saw.
        while status()["busy"] != false {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Restarts the daemon over the same synthetic archive and home.
    pub fn restart(&mut self) {
        let root = self.daemon.archive.root.clone();
        self.runtime.stop_accepting();
        self.server.shutdown().unwrap();
        self.runtime.shutdown().unwrap();
        self.runtime =
            service::start(&root, self.config.clone(), service::ArchiveMode::Standalone).unwrap();
        self.daemon = Arc::clone(&self.runtime.daemon);
        self.server = server::Server::start(Arc::clone(&self.daemon)).unwrap();
    }

    pub(crate) fn watch_reads(
        &self,
    ) -> std::sync::mpsc::Receiver<polylogue_store::read::limit::ReadLimit> {
        let (sender, receiver) = std::sync::mpsc::channel();
        *self.daemon.read_limits.lock().unwrap() = Some(sender);
        receiver
    }

    pub(crate) fn wait_for_sql(limit: &polylogue_store::read::limit::ReadLimit) {
        let start = std::time::Instant::now();
        while limit.sqlite_steps() < 10_000 {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "request never reached SQL"
            );
            std::thread::yield_now();
        }
    }

    pub(crate) fn assert_slot_freed(&self) -> Duration {
        let start = std::time::Instant::now();
        let reads = &self.daemon.reads;
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let limit = crate::daemon::execution::limit(
                        Some(crate::daemon::execution::deadline(Duration::from_secs(2))),
                        Default::default(),
                    );
                    limit.scope(|| reads.reader().map(drop)).unwrap();
                })
                .join()
                .unwrap();
        });
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_secs(2));
        elapsed
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
        self.runtime.stop_accepting();
        if let Err(error) = self.server.shutdown() {
            eprintln!("test daemon socket: {error}");
        }
        if let Err(error) = self.runtime.shutdown() {
            eprintln!("test daemon runtime: {error}");
        }
    }
}

pub mod assets;
mod embeddings;
