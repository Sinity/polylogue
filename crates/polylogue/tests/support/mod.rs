//! A real `polylogue` daemon over a temporary archive, driven through the CLI.
//! Each test crate adds its own import and read helpers in an `impl World`.
#![allow(dead_code)]
use serde_json::Value;
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

pub const BINARY: &str = env!("CARGO_BIN_EXE_polylogue");

/// A child process killed and reaped on drop.
pub struct OwnedChild(pub Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A temporary directory on tmpfs. The daemon fsyncs every archive write, and
/// on the btrfs work disk those syncs dominate test time. An inherited TMPDIR
/// that is already on tmpfs (set by `xtask check`) is kept.
pub fn tempdir() -> tempfile::TempDir {
    let inherited = std::env::temp_dir();
    if inherited.starts_with("/dev/shm") || !Path::new("/dev/shm").is_dir() {
        tempfile::tempdir().unwrap()
    } else {
        tempfile::tempdir_in("/dev/shm").unwrap()
    }
}

pub struct World {
    pub child: Option<OwnedChild>,
    pub work: tempfile::TempDir,
    _socket_owner: tempfile::TempDir,
    pub archive: PathBuf,
    pub runtime: PathBuf,
    pub scratch: PathBuf,
    pub log: PathBuf,
}
impl World {
    /// Starts a daemon on a fresh archive and waits until `status` succeeds.
    pub fn new() -> Self {
        let work = tempdir();
        let archive = work.path().join("archive");
        let scratch = work.path().join("tmp");
        fs::create_dir(&scratch).unwrap();
        // Only the socket lives in /tmp (short path limit); everything else
        // stays in the test's work directory.
        let socket_owner = tempfile::tempdir_in("/tmp").unwrap();
        let runtime = socket_owner.path().join("run");
        fs::create_dir(&runtime).unwrap();
        let log = work.path().join("daemon.log");
        let mut world = Self {
            child: None,
            work,
            _socket_owner: socket_owner,
            archive,
            runtime,
            scratch,
            log,
        };
        world.start();
        world
    }
    pub fn command(&self) -> Command {
        let mut command = Command::new(BINARY);
        command
            .arg("--archive")
            .arg(&self.archive)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("TMPDIR", &self.scratch);
        command
    }
    /// Runs one CLI command with a 40 s deadline.
    pub fn cli(&self, args: &[&str]) -> Output {
        // Output goes to files so large output cannot fill a pipe and block
        // the child while we enforce its deadline.
        let mut stdout = tempfile::tempfile_in(&self.scratch).unwrap();
        let mut stderr = tempfile::tempfile_in(&self.scratch).unwrap();
        let mut child = OwnedChild(
            self.command()
                .args(args)
                .stdout(Stdio::from(stdout.try_clone().unwrap()))
                .stderr(Stdio::from(stderr.try_clone().unwrap()))
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(40);
        let status = loop {
            assert!(
                stdout.metadata().unwrap().len() <= 5_000_000,
                "CLI stdout larger than 5 MB"
            );
            assert!(
                stderr.metadata().unwrap().len() <= 1_000_000,
                "CLI stderr larger than 1 MB"
            );
            if let Some(status) = child.0.try_wait().unwrap() {
                break status;
            }
            assert!(
                Instant::now() < deadline,
                "CLI deadline {args:?}; daemon: {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(2));
        };
        stdout.seek(SeekFrom::Start(0)).unwrap();
        stderr.seek(SeekFrom::Start(0)).unwrap();
        let mut out = vec![];
        let mut err = vec![];
        stdout.read_to_end(&mut out).unwrap();
        stderr.read_to_end(&mut err).unwrap();
        Output {
            status,
            stdout: out,
            stderr: err,
        }
    }
    pub fn log(&self) -> String {
        fs::read_to_string(&self.log).unwrap_or_default()
    }
    pub fn start(&mut self) {
        assert!(self.child.is_none());
        self.child = Some(OwnedChild(
            self.command()
                .arg("daemon")
                .stdout(Stdio::null())
                .stderr(Stdio::from(File::create(&self.log).unwrap()))
                .spawn()
                .unwrap(),
        ));
        // Generous: the suite runs while other builds load the machine.
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let out = self.cli(&["status"]);
            if out.status.success() {
                let value: Value = serde_json::from_slice(&out.stdout).unwrap();
                assert_eq!(value["archive"], self.archive.to_str().unwrap());
                return;
            }
            assert!(
                self.child.as_mut().unwrap().0.try_wait().unwrap().is_none(),
                "daemon exited: {}; status {out:?}",
                self.log()
            );
            assert!(
                Instant::now() < deadline,
                "daemon startup: {}; status {out:?}",
                self.log()
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
    pub fn kill(&mut self) {
        drop(self.child.take());
    }
    pub fn restart(&mut self) {
        self.kill();
        self.start();
    }
    /// Writes `bytes` to a file in the work directory.
    pub fn input(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.work.path().join(name);
        fs::write(&path, bytes).unwrap();
        path
    }
}
impl Drop for World {
    fn drop(&mut self) {
        self.kill();
    }
}
