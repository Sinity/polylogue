//! The environment of a spawned daemon or CLI that must not reach the
//! caller's files: a cleared environment whose every location lies under the
//! sandbox's root, and a web port of its own.

use std::ffi::OsString;
use std::io;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::Command;

/// A sandbox under `root`: `home`, the configuration directory
/// (`XDG_CONFIG_HOME`), and the data, state, cache, runtime and temporary
/// directories at `root/data`, `root/state`, `root/cache`, `root/runtime` and
/// `root/tmp`.
///
/// A process given the sandbox ([`Sandbox::apply`]) inherits nothing but
/// `PATH`: `HOME`, every XDG directory, every root variable the sources
/// declare (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `HERMES_HOME`, ...) and
/// `TMPDIR` point under the root, and `POLYLOGUE_LISTEN` names a port that
/// was free when the sandbox was made. A daemon in it discovers no file and
/// reads no deployment outside the root, and does not compete for the
/// default web port.
#[derive(Debug, Clone)]
pub struct Sandbox {
    root: PathBuf,
    home: PathBuf,
    config: PathBuf,
    listen: SocketAddr,
}

impl Sandbox {
    /// A sandbox with its home at `root/home` and its configuration at
    /// `root/config`.
    pub fn new(root: &Path) -> io::Result<Self> {
        Self::with_home(root, &root.join("home"), &root.join("config"))
    }

    /// A sandbox whose home and configuration directory lie elsewhere, for a
    /// test that lays out its own home.
    pub fn with_home(root: &Path, home: &Path, config: &Path) -> io::Result<Self> {
        // A port the kernel gave out as free. Should another process take it
        // before the daemon binds it, the daemon serves without its web
        // reader and says so; nothing else depends on the port.
        let listen = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?.local_addr()?;
        Ok(Self {
            root: std::path::absolute(root)?,
            home: std::path::absolute(home)?,
            config: std::path::absolute(config)?,
            listen,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn home(&self) -> &Path {
        &self.home
    }

    /// Where the daemon's web reader listens.
    pub fn listen(&self) -> SocketAddr {
        self.listen
    }

    /// The deployment file the daemon and the CLI read by default.
    pub fn deployment(&self) -> PathBuf {
        self.config.join("polylogue/polylogue.toml")
    }

    /// Creates the sandbox's directories; the runtime directory is private,
    /// as the XDG base directory specification requires.
    pub fn create_directories(&self) -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        for directory in [
            &self.home,
            &self.config,
            &self.root.join("data"),
            &self.root.join("state"),
            &self.root.join("cache"),
            &self.runtime(),
            &self.root.join("tmp"),
        ] {
            std::fs::create_dir_all(directory)?;
        }
        std::fs::set_permissions(self.runtime(), std::fs::Permissions::from_mode(0o700))
    }

    /// Writes a deployment that names `archive` and a UTC logical day, unless
    /// one is already there (a test may write its own).
    pub fn write_deployment(&self, archive: &Path) -> io::Result<()> {
        let path = self.deployment();
        if path.exists() {
            return Ok(());
        }
        let archive = archive.to_str().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "the archive path is not UTF-8")
        })?;
        let text = format!(
            "archive = {}\n\n[day]\nzone = \"UTC\"\nstart = \"04:00:00\"\n",
            toml_string(archive)
        );
        std::fs::create_dir_all(path.parent().expect("the deployment has a directory"))?;
        std::fs::write(path, text)
    }

    /// Every variable a process in the sandbox gets, and nothing else.
    pub fn environment(&self) -> Vec<(&'static str, OsString)> {
        let mut environment: Vec<(&'static str, OsString)> = vec![
            ("PATH", std::env::var_os("PATH").unwrap_or_default()),
            ("LANG", "C.UTF-8".into()),
        ];
        // `HOME` and the sources' root variables first: the XDG directories
        // below replace the ones the sources also declare.
        environment.extend(
            polylogue_sources::origin::synthetic_home_environment(&self.home)
                .into_iter()
                .map(|(name, path)| (name, path.into_os_string())),
        );
        for (name, path) in [
            ("XDG_CONFIG_HOME", self.config.clone()),
            ("XDG_DATA_HOME", self.root.join("data")),
            ("XDG_STATE_HOME", self.root.join("state")),
            ("XDG_CACHE_HOME", self.root.join("cache")),
            ("XDG_RUNTIME_DIR", self.runtime()),
            ("TMPDIR", self.root.join("tmp")),
        ] {
            environment.retain(|(existing, _)| *existing != name);
            environment.push((name, path.into_os_string()));
        }
        environment.push(("POLYLOGUE_LISTEN", self.listen.to_string().into()));
        environment
    }

    /// Gives `command` the sandbox's environment ([`Sandbox::environment`])
    /// in place of its own. Callers add narrower variables after.
    pub fn apply<'a>(&self, command: &'a mut Command) -> &'a mut Command {
        command.env_clear().envs(self.environment())
    }

    fn runtime(&self) -> PathBuf {
        self.root.join("runtime")
    }
}

/// `text` as a TOML basic string: quotes, backslashes and control
/// characters escaped, everything else as it is.
fn toml_string(text: &str) -> String {
    let mut quoted = String::from('"');
    for character in text.chars() {
        match character {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            control if control.is_control() => {
                quoted.push_str(&format!("\\u{:04X}", u32::from(control)));
            }
            other => quoted.push(other),
        }
    }
    quoted.push('"');
    quoted
}

#[cfg(test)]
mod tests;
