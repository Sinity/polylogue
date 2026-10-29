//! The binary's one deployment/environment boundary. Grammar-only entry points
//! use ProcessInputs; loading deployment files happens after those early exits.
pub mod signals;
use clap::{Arg, ArgMatches, Command};
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
            "POLYLOGUE_AUTH_TOKEN_FILE",
            "PAGER",
            "NO_COLOR",
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
            polylogue_sources::ORIGINS
                .iter()
                .filter_map(|origin| origin.root.env),
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
    pub fn new(stdout_terminal: bool, stderr_terminal: bool, no_color: bool) -> Self {
        Self {
            stdout_terminal,
            stderr_terminal,
            no_color,
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
    pub cwd: Option<PathBuf>,
}
impl ProcessInputs {
    pub fn capture() -> Result<Self, Error> {
        let environment = EnvSnapshot::capture();
        let output = OutputConfig::new(
            std::io::stdout().is_terminal(),
            std::io::stderr().is_terminal(),
            environment.get_os("NO_COLOR").is_some(),
        );
        Ok(Self {
            environment,
            output,
            clock: WallClock::new(jiff::Timestamp::now),
            identity: None,
            workers: std::thread::available_parallelism().ok().map(usize::from),
            cwd: std::env::current_dir().ok(),
        })
    }
}
#[derive(Debug, Clone)]
pub struct Paths {
    pub home: PathBuf,
    pub config: PathBuf,
    pub data: PathBuf,
    pub state: PathBuf,
    pub cache: PathBuf,
    pub runtime: PathBuf,
    pub check_history: PathBuf,
}
impl Paths {
    pub fn resolve(env: &EnvSnapshot) -> Result<Self, Error> {
        let home = env
            .get_os("HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .ok_or_else(|| Error::Invalid("HOME must be an absolute path".into()))?;
        let xdg = |name, fallback: &str| -> Result<PathBuf, Error> {
            let value = env
                .get_os(name)
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
            check_history: state.join("devkit/history.sqlite"),
            home,
            config,
            data,
            state,
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
    pub fn completion_catalog(&self, archive: &Path) -> PathBuf {
        self.cache
            .join("polylogue")
            .join(crate::runtime_dir::archive_key(archive))
            .join("catalog")
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
    #[cfg(feature = "sinex")]
    pub primary_auth: polylogue_sinex::Authentication,
}
impl Config {
    pub fn standalone_day(&self) -> Result<&DaySetting, Error> {
        self.standalone_day.as_ref().ok_or(Error::SinexDay)
    }
}
pub struct Startup {
    pub config: Arc<Config>,
    pub engine: polylogue_engine::Config,
    pub mode: crate::daemon::service::ArchiveMode,
    pub clock: WallClock,
    pub output: OutputConfig,
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
    pub fn from_matches(matches: &ArgMatches) -> Self {
        Self {
            config_file: matches
                .try_get_one::<PathBuf>("config")
                .ok()
                .flatten()
                .cloned(),
            archive: matches
                .try_get_one::<PathBuf>("archive")
                .ok()
                .flatten()
                .cloned(),
            listen: matches
                .try_get_one::<SocketAddr>("listen")
                .ok()
                .flatten()
                .copied(),
            auth_token_file: matches
                .try_get_one::<PathBuf>("auth-token-file")
                .ok()
                .flatten()
                .cloned(),
        }
    }
}
pub fn augment_command(command: Command) -> Command {
    command
        .arg(
            Arg::new("archive")
                .long("archive")
                .global(true)
                .value_parser(clap::value_parser!(PathBuf))
                .help("Archive directory"),
        )
        .arg(
            Arg::new("config")
                .long("config")
                .global(true)
                .value_parser(clap::value_parser!(PathBuf))
                .help("Deployment configuration file"),
        )
        .arg(
            Arg::new("listen")
                .long("listen")
                .global(true)
                .value_parser(clap::value_parser!(SocketAddr))
                .help("HTTP listen address"),
        )
        .arg(
            Arg::new("auth-token-file")
                .long("auth-token-file")
                .global(true)
                .value_parser(clap::value_parser!(PathBuf))
                .help("HTTP bearer token file"),
        )
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    archive: Option<PathBuf>,
    http: Option<HttpFile>,
    day: Option<DayFile>,
    check_history: Option<PathBuf>,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct HttpFile {
    listen: Option<SocketAddr>,
    auth_token_file: Option<PathBuf>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DayFile {
    zone: String,
    start: jiff::civil::Time,
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
/// The grammar/completion boundary reads only the deployment file's archive
/// selection. It opens no archive and resolves no clock, identity or zone.
pub fn archive_location(process: &ProcessInputs, overrides: &Overrides) -> Result<PathBuf, Error> {
    let paths = Paths::resolve(&process.environment)?;
    let cwd = process
        .cwd
        .as_ref()
        .ok_or_else(|| Error::Invalid("the current directory is unavailable".into()))?;
    let (path, file) = deployment_file(&paths, overrides, cwd)?;
    let base = path
        .parent()
        .ok_or_else(|| Error::Invalid("configuration path has no parent".into()))?;
    Ok(paths.archive(overrides, cwd, file.archive.as_deref(), base))
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
impl Startup {
    pub fn load(process: &ProcessInputs, overrides: Overrides) -> Result<Self, Error> {
        let identity = process.identity.clone().map(Ok).unwrap_or_else(|| {
            let fact = |path: &str| -> Result<String, Error> {
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
                machine: fact("/etc/machine-id")?,
                host: fact("/proc/sys/kernel/hostname")?,
            })
        })?;
        let cwd = process
            .cwd
            .as_ref()
            .ok_or_else(|| Error::Invalid("the current directory is unavailable".into()))?;
        let workers = process
            .workers
            .ok_or_else(|| Error::Invalid("the process CPU allowance is unavailable".into()))?;
        let mut paths = Paths::resolve(&process.environment)?;
        let (config_file, file) = deployment_file(&paths, &overrides, cwd)?;
        let base = config_file
            .parent()
            .ok_or_else(|| Error::Invalid("configuration path has no parent".into()))?;
        let archive = paths.archive(&overrides, cwd, file.archive.as_deref(), base);
        if let Some(path) = file.check_history {
            paths.check_history = absolute(path, base);
        }
        let http = file.http.unwrap_or_default();
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
            .map(|path| absolute(path.into(), cwd))
            .or_else(|| overrides.auth_token_file.map(|path| absolute(path, cwd)))
            .or_else(|| http.auth_token_file.map(|path| absolute(path, base)));
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
        let (mode, primary_auth) = load_mode(&paths.config.join("polylogue/sinex.toml"))?;
        let standalone = matches!(mode, crate::daemon::service::ArchiveMode::Standalone);
        let standalone_day = if standalone {
            let day = file.day.unwrap_or(DayFile {
                zone: "UTC".into(),
                start: jiff::civil::Time::constant(4, 0, 0, 0),
            });
            let zone = jiff::tz::TimeZone::get(&day.zone)
                .map_err(|error| Error::Invalid(error.to_string()))?;
            Some(
                DaySetting::new(zone, day.start)
                    .map_err(|error| Error::Invalid(error.to_string()))?,
            )
        } else {
            #[cfg(feature = "sinex")]
            if file.day.is_some()
                && matches!(mode, crate::daemon::service::ArchiveMode::Primary { .. })
            {
                return Err(Error::SinexDay);
            }
            None
        };
        let tool_dirs = polylogue_sources::ORIGINS
            .iter()
            .filter_map(|origin| origin.root.env)
            .filter_map(|name| {
                process
                    .environment
                    .get_os(name)
                    .map(|value| (name.to_string(), absolute(value.into(), cwd)))
            })
            .collect();
        let signals = SignalDeployment::load(&paths.config.join("polylogue/signals.toml"))?;
        let notification_bus_address = process
            .environment
            .text("DBUS_SESSION_BUS_ADDRESS")?
            .map(str::to_string);
        let engine = polylogue_engine::Config::new(
            paths.home.clone(),
            tool_dirs,
            identity.machine.clone(),
            identity.host.clone(),
            workers,
            process.clock.clone(),
        )
        .map_err(|error| Error::Invalid(error.to_string()))?;
        let config = Arc::new(Config {
            paths,
            archive,
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
            standalone_day,
            #[cfg(feature = "sinex")]
            primary_auth,
        });
        #[cfg(not(feature = "sinex"))]
        let _ = primary_auth;
        Ok(Self {
            config,
            engine,
            mode,
            clock: process.clock.clone(),
            output: process.output,
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
            check_history: state.join("devkit/history.sqlite"),
            state,
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
            #[cfg(feature = "sinex")]
            primary_auth: polylogue_sinex::Authentication::Anonymous,
        })
    }
}
#[cfg(test)]
impl Startup {
    pub(crate) fn for_test(engine: polylogue_engine::Config, archive: &Path) -> Self {
        Self {
            config: Config::for_test(&engine, archive),
            clock: engine.clock.clone(),
            engine,
            mode: crate::daemon::service::ArchiveMode::Standalone,
            output: OutputConfig::new(false, false, true),
        }
    }
}

#[cfg(feature = "sinex")]
pub(crate) fn load_mode(
    path: &Path,
) -> Result<
    (
        crate::daemon::service::ArchiveMode,
        polylogue_sinex::Authentication,
    ),
    Error,
> {
    use crate::daemon::service::ArchiveMode;
    use polylogue_sinex::{Authentication, Config, Mode};
    let text = match optional_file(path) {
        Ok(Some(text)) => text,
        Ok(None) => return Ok((ArchiveMode::Standalone, Authentication::Anonymous)),
        Err(error) => {
            return Ok((
                ArchiveMode::Blocked {
                    configured: None,
                    path: path.into(),
                    error: error.to_string(),
                },
                Authentication::Unavailable {
                    reason: error.to_string(),
                },
            ));
        }
    };
    match Config::parse(&text) {
        Ok(mut config) if config.mode == Mode::Primary => {
            let base = path
                .parent()
                .ok_or_else(|| Error::Invalid("Sinex configuration path has no parent".into()))?;
            config.store = absolute(config.store, base);
            config.password_file = config.password_file.map(|path| absolute(path, base));
            let auth = match &config.password_file {
                None => Authentication::Anonymous,
                Some(file) => match std::fs::read_to_string(file) {
                    Ok(value) if !value.trim_end_matches(['\r', '\n']).is_empty() => {
                        Authentication::Password(value.trim_end_matches(['\r', '\n']).into())
                    }
                    Ok(_) => Authentication::Unavailable {
                        reason: "producer credential file is empty".into(),
                    },
                    Err(error) => Authentication::Unavailable {
                        reason: format!("producer credential file {}: {error}", file.display()),
                    },
                },
            };
            Ok((
                ArchiveMode::Primary {
                    config,
                    path: path.into(),
                },
                auth,
            ))
        }
        Ok(_) => Ok((ArchiveMode::Standalone, Authentication::Anonymous)),
        Err(error) => Ok((
            ArchiveMode::Blocked {
                configured: None,
                path: path.into(),
                error: error.to_string(),
            },
            Authentication::Unavailable {
                reason: error.to_string(),
            },
        )),
    }
}
#[cfg(not(feature = "sinex"))]
pub(crate) fn load_mode(path: &Path) -> Result<(crate::daemon::service::ArchiveMode, ()), Error> {
    #[derive(Deserialize)]
    struct Header {
        mode: String,
    }
    if let Some(text) = optional_file(path)? {
        let header: Header = toml::from_str(&text)
            .map_err(|error| Error::Invalid(format!("{}: {error}", path.display())))?;
        if header.mode != "standalone" {
            return Err(Error::Invalid(format!(
                "{} requires a build with primary-mode support",
                path.display()
            )));
        }
    }
    Ok((crate::daemon::service::ArchiveMode::Standalone, ()))
}

#[cfg(test)]
mod tests;
