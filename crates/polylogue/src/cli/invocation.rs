//! The one scan of a process's arguments: the global options, read from
//! their declarations, and what kind of invocation the arguments make.
//! `main`, completion and the daemon's classification of a socket peer all
//! read a command line through it.

use crate::commands::Command;
use crate::config::{EnvSnapshot, Overrides};
use std::ffi::{OsStr, OsString};

/// An option every invocation accepts, whatever its words.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GlobalOption {
    /// The long name, without its dashes.
    pub name: &'static str,
    /// The value's placeholder in help; `None` for a flag.
    pub value: Option<&'static str>,
    pub help: &'static str,
    pub owner: Owner,
}

/// Who reads a global option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Owner {
    /// The line and command parsers, which see it among the words.
    Line,
    /// The configuration loader; the scan takes it out of the words.
    Deployment(Deployment),
}

/// The deployment selections a command line may override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Deployment {
    Archive,
    Config,
    Listen,
    AuthTokenFile,
}

/// The global options. The clap command tree, the scan and the request
/// envelope are built from this table.
pub(crate) const GLOBAL_OPTIONS: [GlobalOption; 10] = [
    GlobalOption {
        name: "json",
        value: None,
        help: "Machine-readable output",
        owner: Owner::Line,
    },
    GlobalOption {
        name: "request",
        value: Some("JSON"),
        help: "The operation's complete typed argument object",
        owner: Owner::Line,
    },
    GlobalOption {
        name: "request-id",
        value: Some("ID"),
        help: "The request identity, kept across retries",
        owner: Owner::Line,
    },
    GlobalOption {
        name: "deadline",
        value: Some("MS|none"),
        help: "Milliseconds from when the daemon accepts the request, or none to lift the deadline",
        owner: Owner::Line,
    },
    GlobalOption {
        name: "cursor",
        value: Some("CURSOR"),
        help: "Continue from a previous page's `next`",
        owner: Owner::Line,
    },
    GlobalOption {
        name: "budget",
        value: Some("BUDGET"),
        help: "Tokens for a prepared context; defaults to its named pack",
        owner: Owner::Line,
    },
    GlobalOption {
        name: "archive",
        value: Some("DIR"),
        help: "Archive directory",
        owner: Owner::Deployment(Deployment::Archive),
    },
    GlobalOption {
        name: "config",
        value: Some("FILE"),
        help: "Deployment configuration file",
        owner: Owner::Deployment(Deployment::Config),
    },
    GlobalOption {
        name: "listen",
        value: Some("ADDRESS"),
        help: "HTTP listen address",
        owner: Owner::Deployment(Deployment::Listen),
    },
    GlobalOption {
        name: "auth-token-file",
        value: Some("FILE"),
        help: "HTTP bearer token file",
        owner: Owner::Deployment(Deployment::AuthTokenFile),
    },
];

impl GlobalOption {
    /// The option on the clap command tree, which gives it help and
    /// completion; its value is read from the scan or the parsed words.
    pub(crate) fn clap_arg(&self) -> clap::Arg {
        let arg = clap::Arg::new(self.name)
            .long(self.name)
            .global(true)
            .help(self.help);
        let Some(value) = self.value else {
            return arg.action(clap::ArgAction::SetTrue);
        };
        let arg = arg.value_name(value);
        match self.owner {
            Owner::Line => arg,
            Owner::Deployment(Deployment::Listen) => {
                arg.value_parser(clap::value_parser!(std::net::SocketAddr))
            }
            Owner::Deployment(_) => arg.value_parser(clap::value_parser!(std::path::PathBuf)),
        }
    }
}

impl Deployment {
    fn set(self, overrides: &mut Overrides, value: &OsStr) -> Result<(), String> {
        match self {
            Self::Archive => overrides.archive = Some(value.into()),
            Self::Config => overrides.config_file = Some(value.into()),
            Self::AuthTokenFile => overrides.auth_token_file = Some(value.into()),
            Self::Listen => {
                overrides.listen = Some(
                    value
                        .to_str()
                        .ok_or("--listen needs an address")?
                        .parse()
                        .map_err(|error| format!("--listen: {error}"))?,
                );
            }
        }
        Ok(())
    }
}

fn global(name: &str) -> Option<&'static GlobalOption> {
    GLOBAL_OPTIONS.iter().find(|option| option.name == name)
}

/// A command line, read once.
pub(crate) struct Invocation {
    /// The deployment options it names.
    pub overrides: Overrides,
    /// Every argument after the program name except the deployment
    /// options: the words the line and command parsers read.
    pub words: Vec<String>,
    /// `--json` stands among the words, so a refusal before any parser ran
    /// is printed for a program.
    pub json: bool,
    pub shape: Shape,
    /// Why the arguments cannot run: a word that is not UTF-8, or a
    /// deployment option without a valid value. Completion ignores it.
    pub problem: Option<String>,
}

pub(crate) enum Shape {
    /// A shell asking for completions (`COMPLETE` is set).
    Completion(Completion),
    /// Chrome starting the native-messaging host with an extension origin.
    NativeHost,
    /// A harness hook: `internal hook`, which exits 0 whatever happens.
    Hook,
    /// `help`, `--help` or `-h` as the first word.
    Help,
    /// A table command, `None` for words that name no row (`--version`,
    /// a group such as `dev`).
    Command(Option<&'static Command>),
    /// A line: a selection and its verbs, or nothing.
    Line,
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Completion(completion) => f.debug_tuple("Completion").field(completion).finish(),
            Self::NativeHost => f.write_str("NativeHost"),
            Self::Hook => f.write_str("Hook"),
            Self::Help => f.write_str("Help"),
            Self::Command(row) => f
                .debug_tuple("Command")
                .field(&row.map(|row| row.name))
                .finish(),
            Self::Line => f.write_str("Line"),
        }
    }
}

/// A completion callback: the words being completed and the cursor.
#[derive(Debug)]
pub(crate) struct Completion {
    pub arguments: Vec<OsString>,
    pub index: usize,
    /// What zsh has typed of the current word (`_POLYLOGUE_COMPLETE_PREFIX`).
    pub prefix: Option<String>,
}

/// Reads `arguments` (the program name first) once. `environment` decides
/// whether this is a completion callback; the daemon passes an empty one.
pub(crate) fn scan(arguments: &[OsString], environment: &EnvSnapshot) -> Invocation {
    let completion = environment.get_os("COMPLETE").is_some().then(|| {
        let separator = arguments.iter().position(|value| value == "--");
        let callback = separator
            .map(|separator| arguments[separator + 1..].to_vec())
            .unwrap_or_default();
        let index = environment
            .get_os("_CLAP_COMPLETE_INDEX")
            .and_then(OsStr::to_str)
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| callback.len().saturating_sub(1));
        let mut invocation = arguments[..separator.unwrap_or(arguments.len())].to_vec();
        invocation.extend(callback.iter().skip(1).cloned());
        let completion = Completion {
            arguments: callback,
            index,
            prefix: environment
                .get_os("_POLYLOGUE_COMPLETE_PREFIX")
                .and_then(OsStr::to_str)
                .map(str::to_owned),
        };
        (invocation, completion)
    });
    let (arguments, completion) = match completion {
        Some((invocation, completion)) => (invocation, Some(completion)),
        None => (arguments.to_vec(), None),
    };
    let mut overrides = Overrides::default();
    let mut words = Vec::new();
    let mut json = false;
    let mut problem = None;
    // The first words that are neither options nor their values: the
    // command they name.
    let mut leading: Vec<String> = Vec::new();
    let mut leading_ended = false;
    let mut rest = arguments.iter().skip(1);
    while let Some(argument) = rest.next() {
        let Some(word) = argument.to_str() else {
            problem.get_or_insert_with(|| "command arguments must be UTF-8".to_owned());
            words.push(argument.to_string_lossy().into_owned());
            continue;
        };
        if word == "--" {
            // The next argument is literal: a selection or an owed word.
            // It names no command, so the leading words end here.
            words.push(word.to_owned());
            leading_ended = true;
            if let Some(literal) = rest.next() {
                words.push(literal.to_string_lossy().into_owned());
            }
            continue;
        }
        let (name, assigned) = match word.strip_prefix("--") {
            Some(option) => match option.split_once('=') {
                Some((name, value)) => (name, Some(value)),
                None => (option, None),
            },
            None => ("", None),
        };
        match global(name) {
            Some(GlobalOption {
                owner: Owner::Deployment(deployment),
                ..
            }) => {
                let value = match assigned {
                    Some(value) => Some(OsStr::new(value)),
                    None => rest.next().map(OsString::as_os_str),
                };
                let set = match value {
                    Some(value) => deployment.set(&mut overrides, value),
                    None => Err(format!("--{name} needs a value")),
                };
                if let Err(error) = set {
                    problem.get_or_insert(error);
                }
            }
            Some(option) => {
                json |= option.name == "json";
                words.push(word.to_owned());
                if option.value.is_some()
                    && assigned.is_none()
                    && let Some(value) = rest.next()
                {
                    words.push(value.to_string_lossy().into_owned());
                }
            }
            None => {
                if !word.starts_with('-') && !leading_ended && leading.len() < 3 {
                    leading.push(word.to_owned());
                }
                words.push(word.to_owned());
            }
        }
    }
    let shape = match completion {
        Some(completion) => Shape::Completion(completion),
        None => shape(&words, &leading),
    };
    Invocation {
        overrides,
        words,
        json,
        shape,
        problem,
    }
}

fn shape(words: &[String], leading: &[String]) -> Shape {
    if let [only] = words
        && only.starts_with("chrome-extension://")
    {
        return Shape::NativeHost;
    }
    if words
        .first()
        .is_some_and(|word| matches!(word.as_str(), "help" | "--help" | "-h"))
    {
        return Shape::Help;
    }
    // The longest table row the leading words name; the line parser
    // decides the rest, such as `--help` after `--json` or a group word.
    let row = (1..=leading.len()).rev().find_map(|count| {
        crate::commands::command(&leading[..count].join(" "))
            .filter(|row| !matches!(row.cli, crate::commands::Cli::Line))
    });
    match row {
        Some(row) if row.name == "internal hook" => Shape::Hook,
        Some(row) => Shape::Command(Some(row)),
        None if words == ["--version"] || super::line::names_command(words) => Shape::Command(None),
        None => Shape::Line,
    }
}

#[cfg(test)]
mod tests;
