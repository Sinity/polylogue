//! The binary's one deployment/environment boundary. Grammar-only entry points
//! use ProcessInputs; loading deployment files happens after those early exits.
//! Every client loads a `Client` (paths, the archive, output settings);
//! only the daemon and the commands that run the engine in the calling
//! process extend it into a `Startup` (identity, the token file,
//! `sinex.toml`, `signals.toml`).
pub mod signals;
use polylogue_model::WallClock;
use query_language::time::DaySetting;
use serde::Deserialize;
pub use signals::SignalDeployment;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::{OsStr, OsString},
    io::IsTerminal,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("configuration {0}")]
    Invalid(String),
    #[error("{}: {source}", path.display())]
    File {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Sinex owns the logical day in primary mode")]
    SinexDay,
    #[error("the current directory is unavailable")]
    WorkingDirectory,
}
impl Error {
    /// The failure a start-up that met this error leaves with: an
    /// unreadable file or working directory is `io`, a value the
    /// configuration cannot take is `bad_value`.
    pub fn failure(&self) -> command_surface::Failure {
        let code = match self {
            Self::File { .. } | Self::WorkingDirectory => polylogue_model::Code::Io,
            Self::Invalid(_) | Self::SinexDay => polylogue_model::Code::BadValue,
        };
        crate::protocol::failure(code, self.to_string())
    }
}
#[derive(Clone)]
pub struct EnvSnapshot {
    values: BTreeMap<String, OsString>,
}
impl std::fmt::Debug for EnvSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvSnapshot")
            .field("keys", &self.values.keys().collect::<Vec<_>>())
            .finish()
    }
}
impl EnvSnapshot {
    pub fn capture() -> Self {
        let mut names: BTreeSet<&str> = [
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
            "XDG_CACHE_HOME",
            "XDG_RUNTIME_DIR",
            "POLYLOGUE_LISTEN",
            "POLYLOGUE_MCP_SOCKET",
            "POLYLOGUE_AUTH_TOKEN_FILE",
            "PAGER",
            "ZDOTDIR",
            "NO_COLOR",
            "TERM",
            "NOTIFY_SOCKET",
            "WATCHDOG_USEC",
            "WATCHDOG_PID",
            "COMPLETE",
            "_CLAP_COMPLETE_INDEX",
            "_POLYLOGUE_COMPLETE_PREFIX",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDE_PID",
            "CLAUDECODE",
            "CODEX_THREAD_ID",
            "CODEX_TOOL_CALL_ID",
            "HERMES_SESSION_ID",
            "GEMINI_CLI",
            "HERMES_HOME",
            "KITTY_PID",
            "KITTY_WINDOW_ID",
            "DBUS_SESSION_BUS_ADDRESS",
            "VOYAGE_API_KEY",
        ]
        .into_iter()
        .collect();
        names.extend(
            polylogue_sources::origin::variable_roots()
                .into_iter()
                .map(|(variable, _)| variable),
        );
        Self::from_values(
            names
                .into_iter()
                .filter_map(|name| std::env::var_os(name).map(|value| (name.to_string(), value))),
        )
    }
    pub fn from_values(values: impl IntoIterator<Item = (String, OsString)>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }
    pub fn get_os(&self, key: &str) -> Option<&OsStr> {
        self.values.get(key).map(OsString::as_os_str)
    }
    pub fn text(&self, key: &str) -> Result<Option<&str>, Error> {
        self.get_os(key)
            .map(|value| {
                value
                    .to_str()
                    .ok_or_else(|| Error::Invalid(format!("{key} is not UTF-8")))
            })
            .transpose()
    }
    #[cfg(test)]
    pub fn caller_environment(&self) -> BTreeMap<String, String> {
        [
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDE_PID",
            "CLAUDECODE",
            "CODEX_THREAD_ID",
            "CODEX_TOOL_CALL_ID",
            "HERMES_SESSION_ID",
            "GEMINI_CLI",
            "HOME",
            "HERMES_HOME",
            "KITTY_PID",
            "KITTY_WINDOW_ID",
        ]
        .into_iter()
        .filter_map(|key| {
            self.get_os(key)
                .and_then(OsStr::to_str)
                .map(|value| (key.into(), value.into()))
        })
        .collect()
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputConfig {
    pub stdout_terminal: bool,
    pub stderr_terminal: bool,
    no_color: bool,
}
impl OutputConfig {
    pub fn new(
        stdout_terminal: bool,
        stderr_terminal: bool,
        no_color: bool,
        term: Option<&str>,
    ) -> Self {
        Self {
            stdout_terminal,
            stderr_terminal,
            no_color: no_color || term == Some("dumb"),
        }
    }
    pub fn stdout_color(self) -> bool {
        self.stdout_terminal && !self.no_color
    }
    pub fn stderr_color(self) -> bool {
        self.stderr_terminal && !self.no_color
    }
}
#[derive(Debug, Clone)]
pub struct Identity {
    pub machine: String,
    pub host: String,
}
#[derive(Debug, Clone)]
pub struct ProcessInputs {
    pub environment: EnvSnapshot,
    pub output: OutputConfig,
    pub clock: WallClock,
    pub identity: Option<Identity>,
    pub workers: Option<usize>,
    pub memory_bytes: Option<u64>,
    pub cwd: Option<PathBuf>,
}
impl ProcessInputs {
    pub fn capture() -> Self {
        let environment = EnvSnapshot::capture();
        let (workers, memory_bytes) = process_allowance();
        let output = OutputConfig::new(
            std::io::stdout().is_terminal(),
            std::io::stderr().is_terminal(),
            environment.get_os("NO_COLOR").is_some(),
            environment.get_os("TERM").and_then(OsStr::to_str),
        );
        Self {
            environment,
            output,
            clock: WallClock::new(jiff::Timestamp::now),
            identity: None,
            workers,
            memory_bytes,
            cwd: std::env::current_dir().ok(),
        }
    }
}
/// What this process may use: its cores (physical cores within its CPU
/// affinity, at most its control groups' CPU quota) and its memory (what
/// the system has available now, at most its control groups' limit).
fn process_allowance() -> (Option<usize>, Option<u64>) {
    let mut workers =
        physical_cores().or_else(|| std::thread::available_parallelism().ok().map(usize::from));
    let mut bytes = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|memory| {
            memory.lines().find_map(|line| {
                let mut fields = line.split_whitespace();
                if fields.next()? != "MemAvailable:" {
                    return None;
                }
                fields.next()?.parse::<u64>().ok()?.checked_mul(1024)
            })
        });
    if let Ok(groups) = std::fs::read_to_string("/proc/self/cgroup")
        && let Some(group) = groups.lines().find_map(|line| line.strip_prefix("0::"))
    {
        let root = Path::new("/sys/fs/cgroup");
        let path = root.join(group.trim_start_matches('/'));
        for parent in path
            .ancestors()
            .take_while(|parent| parent.starts_with(root))
        {
            if let Ok(limit) = std::fs::read_to_string(parent.join("memory.max"))
                && let Ok(limit) = limit.trim().parse::<u64>()
            {
                bytes = Some(bytes.map_or(limit, |bytes| bytes.min(limit)));
            }
            if let Ok(limit) = std::fs::read_to_string(parent.join("cpu.max")) {
                let mut fields = limit.split_whitespace();
                if let (Some(quota), Some(period)) = (fields.next(), fields.next())
                    && let (Ok(quota), Ok(period)) = (quota.parse::<u64>(), period.parse::<u64>())
                    && period > 0
                {
                    let cpus = (quota / period).max(1) as usize;
                    workers = Some(workers.map_or(cpus, |workers| workers.min(cpus)));
                }
            }
        }
    }
    (workers, bytes)
}

/// The physical cores among the CPUs this process may run on: hardware
/// threads of one core share its execution units, so a second worker on a
/// core adds little.
fn physical_cores() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let allowed = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?;
    let mut cores = std::collections::BTreeSet::new();
    for range in allowed.trim().split(',') {
        let (first, last) = match range.split_once('-') {
            Some((first, last)) => (first.parse::<usize>().ok()?, last.parse::<usize>().ok()?),
            None => {
                let cpu = range.parse::<usize>().ok()?;
                (cpu, cpu)
            }
        };
        for cpu in first..=last {
            let siblings = std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/core_cpus_list"
            ))
            .ok()?;
            cores.insert(siblings.trim().to_owned());
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

#[derive(Debug, Clone)]
pub struct Paths {
    pub home: PathBuf,
    pub config: PathBuf,
    pub data: PathBuf,
    pub cache: PathBuf,
    pub runtime: PathBuf,
}
impl Paths {
    pub fn resolve(env: &EnvSnapshot) -> Result<Self, Error> {
        let home = env
            .get_os("HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .ok_or_else(|| Error::Invalid("HOME must be an absolute path".into()))?;
        let xdg = |name, fallback: &str| -> Result<PathBuf, Error> {
            // An empty value counts as unset (XDG base directory spec).
            let value = env
                .get_os(name)
                .filter(|value| !value.is_empty())
                .map_or_else(|| home.join(fallback), PathBuf::from);
            if !value.is_absolute() {
                return Err(Error::Invalid(format!("{name} must be an absolute path")));
            }
            Ok(value)
        };
        let config = xdg("XDG_CONFIG_HOME", ".config")?;
        let data = xdg("XDG_DATA_HOME", ".local/share")?;
        let state = xdg("XDG_STATE_HOME", ".local/state")?;
        let cache = xdg("XDG_CACHE_HOME", ".cache")?;
        let runtime = env
            .get_os("XDG_RUNTIME_DIR")
            .map_or_else(|| state.join("runtime"), PathBuf::from);
        if !runtime.is_absolute() {
            return Err(Error::Invalid("XDG_RUNTIME_DIR must be absolute".into()));
        }
        Ok(Self {
            home,
            config,
            data,
            cache,
            runtime,
        })
    }
    /// Resolve one declared archive selection from supplied deployment values.
    pub fn archive(
        &self,
        overrides: &Overrides,
        cwd: &Path,
        file_archive: Option<&Path>,
        file_base: &Path,
    ) -> PathBuf {
        overrides
            .archive
            .as_ref()
            .map(|path| absolute(path.clone(), cwd))
            .or_else(|| file_archive.map(|path| absolute(path.into(), file_base)))
            .unwrap_or_else(|| self.data.join("polylogue"))
    }
    pub fn runtime_dir(&self, archive: &Path) -> PathBuf {
        self.runtime
            .join("polylogue")
            .join(crate::runtime_dir::archive_key(archive))
    }
    /// Rebuildable Git roots, scoped to the working tree whose HEAD was read.
    pub fn git_roots(&self, repository: &Path) -> PathBuf {
        let key = blake3::hash(repository.as_os_str().as_encoded_bytes());
        self.cache
            .join("polylogue")
            .join("git-roots")
            .join(format!("{}.json", &key.to_hex()[..32]))
    }
}
#[derive(Clone)]
pub struct HttpConfig {
    pub listen: SocketAddr,
    pub token: Option<BearerToken>,
}
#[derive(Clone)]
pub struct BearerToken(String);
impl BearerToken {
    pub fn expose(&self) -> &str {
        &self.0
    }
}
impl std::fmt::Debug for BearerToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BearerToken([redacted])")
    }
}
#[derive(Clone)]
pub struct Config {
    pub paths: Paths,
    pub archive: PathBuf,
    pub http: HttpConfig,
    pub identity: Identity,
    pub requests: polylogue_model::user::RequestMetadata,
    pub signals: SignalDeployment,
    pub notification_bus_address: Option<String>,
    pub embedding_credentials: crate::embeddings::Credentials,
    standalone_day: Option<DaySetting>,
}
impl Config {
    pub fn standalone_day(&self) -> Result<&DaySetting, Error> {
        self.standalone_day.as_ref().ok_or(Error::SinexDay)
    }
    /// A separate synthetic archive receives its own explicit day and no
    /// deployment outlets or paid credentials from the operator's archive.
    pub fn standalone_for_archive(
        &self,
        archive: PathBuf,
        day: DaySetting,
    ) -> Result<Arc<Self>, Error> {
        if !archive.is_absolute() {
            return Err(Error::Invalid(
                "the standalone archive path must be absolute".into(),
            ));
        }
        let mut config = self.clone();
        config.archive = archive;
        config.standalone_day = Some(day);
        config.http = HttpConfig {
            listen: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            token: None,
        };
        config.signals = SignalDeployment::default();
        config.notification_bus_address = None;
        config.embedding_credentials = crate::embeddings::Credentials::new(None);
        Ok(Arc::new(config))
    }
}
/// What every client invocation loads: where the archive and the
/// per-user directories are, and how output is shown.
#[derive(Debug, Clone)]
pub struct Client {
    pub paths: Paths,
    pub archive: PathBuf,
    pub(crate) mcp_socket: Option<PathBuf>,
    pub clock: WallClock,
    pub output: OutputConfig,
    /// The pager a terminal read is shown through: `$PAGER`, else `less -FRX`.
    pub pager: String,
    /// Zsh reads its startup files here when ZDOTDIR is set.
    pub zsh_directory: PathBuf,
    /// The deployment's logical day, or why it cannot be resolved; a
    /// standalone daemon refuses to start without it.
    day: Result<DaySetting, String>,
    deployment: DeploymentFile,
}
impl Client {
    /// The zone a client shows times in: the logical day's, or UTC when
    /// the deployment names none that resolves.
    pub fn zone(&self) -> jiff::tz::TimeZone {
        self.day
            .as_ref()
            .map_or(jiff::tz::TimeZone::UTC, |day| day.zone().clone())
    }
}
/// The parts of the loaded deployment file only `Startup::extend` reads.
#[derive(Debug, Clone)]
struct DeploymentFile {
    overrides: Overrides,
    cwd: PathBuf,
    /// The directory of `polylogue.toml`, against which its relative paths
    /// and its neighbouring files resolve.
    base: PathBuf,
    http: HttpFile,
}
/// A client's configuration extended with what only the daemon and the
/// commands that run the engine here need.
pub struct Startup {
    pub client: Client,
    pub config: Arc<Config>,
    pub engine: polylogue_engine::Config,
    pub mode: crate::daemon::service::ArchiveMode,
    pub hook: crate::hook::Inputs,
    pub(crate) lifecycle: crate::daemon::lifecycle::Inputs,
}
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub config_file: Option<PathBuf>,
    pub archive: Option<PathBuf>,
    pub listen: Option<SocketAddr>,
    pub auth_token_file: Option<PathBuf>,
}
impl Overrides {
    pub fn config_path(&self, paths: &Paths, cwd: &Path) -> PathBuf {
        self.config_file
            .as_ref()
            .map(|path| absolute(path.clone(), cwd))
            .unwrap_or_else(|| paths.config.join("polylogue/polylogue.toml"))
    }
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    archive: Option<PathBuf>,
    http: Option<HttpFile>,
    day: Option<DayFile>,
}
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct HttpFile {
    listen: Option<SocketAddr>,
    auth_token_file: Option<PathBuf>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DayFile {
    /// An IANA zone; the host's named zone when absent.
    zone: Option<String>,
    /// When a logical day starts; 04:00 when absent.
    start: Option<jiff::civil::Time>,
}
fn absolute(path: PathBuf, base: &Path) -> PathBuf {
    use std::path::Component;
    let joined = if path.is_absolute() {
        path
    } else {
        base.join(path)
    };
    let mut resolved = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            _ => resolved.push(component.as_os_str()),
        }
    }
    resolved
}
fn optional_file(path: &Path) -> Result<Option<String>, Error> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(Error::File {
            path: path.into(),
            source,
        }),
    }
}
fn deployment_file(
    paths: &Paths,
    overrides: &Overrides,
    cwd: &Path,
) -> Result<(PathBuf, FileConfig), Error> {
    let path = overrides.config_path(paths, cwd);
    let file = optional_file(&path)?
        .map(|text| {
            toml::from_str(&text)
                .map_err(|error| Error::Invalid(format!("{}: {error}", path.display())))
        })
        .transpose()?
        .unwrap_or_default();
    Ok((path, file))
}
impl Client {
    /// Loads the configuration: the deployment file, then the flags, then
    /// the environment where it is meaningful. This reads `polylogue.toml`
    /// and nothing else; `Startup::extend` adds the daemon's part from the
    /// same file.
    pub fn load(process: &ProcessInputs, overrides: Overrides) -> Result<Self, Error> {
        let cwd = process.cwd.clone().ok_or(Error::WorkingDirectory)?;
        let paths = Paths::resolve(&process.environment)?;
        let (config_file, file) = deployment_file(&paths, &overrides, &cwd)?;
        let base = config_file
            .parent()
            .ok_or_else(|| Error::Invalid("configuration path has no parent".into()))?
            .to_owned();
        let archive = paths.archive(&overrides, &cwd, file.archive.as_deref(), &base);
        let mcp_socket = if overrides.archive.is_none() && overrides.config_file.is_none() {
            process
                .environment
                .get_os("POLYLOGUE_MCP_SOCKET")
                .filter(|path| !path.is_empty())
                .map(PathBuf::from)
                .map(|path| {
                    if path.is_absolute() && path.file_name().is_some() {
                        Ok(path)
                    } else {
                        Err(Error::Invalid(
                            "POLYLOGUE_MCP_SOCKET must be an absolute socket path".into(),
                        ))
                    }
                })
                .transpose()?
        } else {
            None
        };
        let DayFile { zone, start } = file.day.unwrap_or(DayFile {
            zone: None,
            start: None,
        });
        let start = start.unwrap_or(jiff::civil::Time::constant(4, 0, 0, 0));
        let day = match zone {
            Some(zone) => jiff::tz::TimeZone::get(&zone)
                .map_err(|error| error.to_string())
                .and_then(|zone| DaySetting::new(zone, start).map_err(|error| error.to_string())),
            None => DaySetting::system(start).map_err(|error| error.to_string()),
        };
        Ok(Self {
            zsh_directory: process
                .environment
                .get_os("ZDOTDIR")
                .map(|path| absolute(path.into(), &cwd))
                .unwrap_or_else(|| paths.home.clone()),
            paths,
            archive,
            mcp_socket,
            clock: process.clock.clone(),
            output: process.output,
            pager: process
                .environment
                .text("PAGER")?
                .filter(|pager| !pager.trim().is_empty())
                .unwrap_or("less -FRX")
                .to_owned(),
            day,
            deployment: DeploymentFile {
                overrides,
                cwd,
                base,
                http: file.http.unwrap_or_default(),
            },
        })
    }
}
impl Startup {
    /// Adds what only the daemon and the commands that run the engine in
    /// this process need: the machine identity, the HTTP listener and its
    /// token, the archive mode from `sinex.toml` and the outlets in
    /// `signals.toml`.
    pub fn extend(client: Client, process: &ProcessInputs) -> Result<Self, Error> {
        let identity = process.identity.clone().map(Ok).unwrap_or_else(|| {
            let read_line = |path: &str| -> Result<String, Error> {
                let value = std::fs::read_to_string(path).map_err(|source| Error::File {
                    path: path.into(),
                    source,
                })?;
                if value.trim().is_empty() {
                    return Err(Error::Invalid(format!("identity file {path} is empty")));
                }
                Ok(value.trim().into())
            };
            Ok(Identity {
                machine: read_line("/etc/machine-id")?,
                host: read_line("/proc/sys/kernel/hostname")?,
            })
        })?;
        let workers = process
            .workers
            .ok_or_else(|| Error::Invalid("the process CPU allowance is unavailable".into()))?;
        let DeploymentFile {
            overrides,
            cwd,
            base,
            http,
        } = client.deployment.clone();
        let listen = process
            .environment
            .text("POLYLOGUE_LISTEN")?
            .map(|text| {
                text.parse::<SocketAddr>()
                    .map_err(|error| Error::Invalid(format!("POLYLOGUE_LISTEN: {error}")))
            })
            .transpose()?
            .or(overrides.listen)
            .or(http.listen)
            .unwrap_or(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8765));
        let token_file = process
            .environment
            .get_os("POLYLOGUE_AUTH_TOKEN_FILE")
            .map(|path| absolute(path.into(), &cwd))
            .or_else(|| overrides.auth_token_file.map(|path| absolute(path, &cwd)))
            .or_else(|| http.auth_token_file.map(|path| absolute(path, &base)));
        let token = token_file
            .map(|path| {
                let value = std::fs::read_to_string(&path).map_err(|source| Error::File {
                    path: path.clone(),
                    source,
                })?;
                let value = value.trim_end();
                if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_graphic()) {
                    return Err(Error::Invalid(format!(
                        "authentication token file {} must contain one nonempty ASCII bearer token",
                        path.display()
                    )));
                }
                Ok(BearerToken(value.into()))
            })
            .transpose()?;
        if !listen.ip().is_loopback() && token.is_none() {
            return Err(Error::Invalid(
                "a non-loopback listen address requires an authentication token file".into(),
            ));
        }
        let mode_path = if overrides.config_file.is_some() {
            base.join("sinex.toml")
        } else {
            client.archive.join("sinex.toml")
        };
        let mode = archive_mode(&mode_path);
        let standalone = matches!(mode, crate::daemon::service::ArchiveMode::Standalone);
        let tool_dirs = polylogue_sources::origin::variable_roots()
            .into_iter()
            .filter_map(|(name, _)| {
                process
                    .environment
                    .get_os(name)
                    .map(|value| (name.to_string(), absolute(value.into(), &cwd)))
            })
            .collect();
        let signals = SignalDeployment::read(&base.join("signals.toml"))?;
        let notification_bus_address = process
            .environment
            .text("DBUS_SESSION_BUS_ADDRESS")?
            .map(str::to_string);
        let engine = polylogue_engine::Config::new(
            client.paths.home.clone(),
            tool_dirs,
            identity.machine.clone(),
            identity.host.clone(),
            workers,
            process.clock.clone(),
        )
        .map_err(|error| Error::Invalid(error.to_string()))?
        .with_memory_budget(polylogue_store::archive::MemoryBudget::from_allowance(
            process.memory_bytes.unwrap_or(4 << 30),
        ));
        let config = Arc::new(Config {
            paths: client.paths.clone(),
            archive: client.archive.clone(),
            http: HttpConfig { listen, token },
            requests: polylogue_model::user::RequestMetadata::new(
                identity.machine.clone(),
                process.clock.clone(),
            ),
            identity,
            signals,
            notification_bus_address,
            embedding_credentials: crate::embeddings::Credentials::new(
                process
                    .environment
                    .text("VOYAGE_API_KEY")?
                    .map(str::to_string),
            ),
            standalone_day: standalone
                .then(|| client.day.clone().map_err(Error::Invalid))
                .transpose()?,
        });
        Ok(Self {
            config,
            engine,
            mode,
            lifecycle: crate::daemon::lifecycle::Inputs::from_environment(
                &process.environment,
                std::process::id(),
            )?,
            hook: crate::hook::Inputs {
                kitty_pid: process
                    .environment
                    .get_os("KITTY_PID")
                    .and_then(OsStr::to_str)
                    .and_then(|value| value.parse::<u32>().ok())
                    .filter(|pid| *pid > 0),
                kitty_window: process
                    .environment
                    .get_os("KITTY_WINDOW_ID")
                    .and_then(OsStr::to_str)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string),
            },
            client,
        })
    }
}
#[cfg(test)]
impl Config {
    /// A fixture configuration from explicit engine inputs. It never captures
    /// environment, opens deployment files or connects a notification bus.
    pub(crate) fn for_test(engine: &polylogue_engine::Config, archive: &Path) -> Arc<Self> {
        assert!(engine.home.is_absolute() && archive.is_absolute());
        let state = engine.home.join(".local/state");
        let paths = Paths {
            home: engine.home.clone(),
            config: engine.home.join(".config"),
            data: engine.home.join(".local/share"),
            cache: engine.home.join(".cache"),
            runtime: state.join("runtime"),
        };
        Arc::new(Self {
            paths,
            archive: archive.into(),
            http: HttpConfig {
                listen: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
                token: None,
            },
            identity: Identity {
                machine: engine.machine.clone(),
                host: engine.host.clone(),
            },
            requests: polylogue_model::user::RequestMetadata::new(
                engine.machine.clone(),
                engine.clock.clone(),
            ),
            signals: SignalDeployment::default(),
            notification_bus_address: None,
            embedding_credentials: crate::embeddings::Credentials::new(None),
            standalone_day: Some(
                DaySetting::new(
                    jiff::tz::TimeZone::UTC,
                    jiff::civil::Time::constant(4, 0, 0, 0),
                )
                .expect("named fixture day"),
            ),
        })
    }
}
#[cfg(test)]
impl Client {
    /// The client part of `Config::for_test`'s fixture.
    pub(crate) fn for_test(config: &Config, clock: WallClock) -> Self {
        Self {
            paths: config.paths.clone(),
            archive: config.archive.clone(),
            mcp_socket: None,
            clock,
            output: OutputConfig::new(false, false, true, None),
            pager: "cat".into(),
            zsh_directory: config.paths.home.clone(),
            day: Ok(config
                .standalone_day
                .clone()
                .expect("fixture configurations are standalone")),
            deployment: DeploymentFile {
                overrides: Overrides::default(),
                cwd: config.paths.home.clone(),
                base: config.paths.config.join("polylogue"),
                http: HttpFile::default(),
            },
        }
    }
}
#[cfg(test)]
impl Startup {
    pub(crate) fn for_test(engine: polylogue_engine::Config, archive: &Path) -> Self {
        let config = Config::for_test(&engine, archive);
        Self {
            client: Client::for_test(&config, engine.clock.clone()),
            config,
            engine,
            mode: crate::daemon::service::ArchiveMode::Standalone,
            lifecycle: crate::daemon::lifecycle::Inputs::from_environment(
                &EnvSnapshot::from_values([]),
                0,
            )
            .expect("empty service environment"),
            hook: crate::hook::Inputs {
                kitty_pid: None,
                kitty_window: None,
            },
        }
    }
}

/// The mode `sinex.toml` asks for. Primary mode is not built in this version
/// (`docs/primary-mode.md`): with the `sinex` feature a configuration asking
/// for it is refused at start; without it the archive is blocked, readable
/// with writes paused.
pub(crate) fn archive_mode(path: &Path) -> crate::daemon::service::ArchiveMode {
    use crate::daemon::service::ArchiveMode;
    #[derive(Deserialize)]
    struct Header {
        mode: String,
    }
    let result = optional_file(path).and_then(|text| match text {
        None => Ok(ArchiveMode::Standalone),
        Some(text) => {
            let header: Header =
                toml::from_str(&text).map_err(|error| Error::Invalid(error.to_string()))?;
            match header.mode.as_str() {
                "standalone" => Ok(ArchiveMode::Standalone),
                #[cfg(feature = "sinex")]
                "primary" => Ok(ArchiveMode::Primary),
                #[cfg(not(feature = "sinex"))]
                "primary" => Err(Error::Invalid(
                    "this build has no primary-mode support".into(),
                )),
                mode => Err(Error::Invalid(format!("unknown archive mode {mode:?}"))),
            }
        }
    });
    result.unwrap_or_else(|error| ArchiveMode::Blocked {
        configured: None,
        path: path.into(),
        error: error.to_string(),
    })
}

#[cfg(test)]
mod tests;
