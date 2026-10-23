//! The command line, built from the command table. A bare query
//! (`polylogue repo:x since:7d`) runs the query language; subcommands cover
//! everything else. Every command takes `--json` for programs and agents.

mod changes;
pub mod local;
pub mod print;

use crate::commands::{Action, COMMANDS, Caller, Cli, Command, GROUPS, Kind, Surface, Values};
use crate::daemon::{Request, Response};
use clap::ArgAction;
use clap::parser::{ArgMatches, ValueSource};
use polylogue_store::Archive;
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::ExitCode;

/// The clap command tree: one subcommand per table command, grouped by
/// their first word, and the bare query as the root's own arguments.
pub fn command() -> clap::Command {
    let mut root = clap::Command::new("polylogue")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Search, read and measure your AI sessions")
        .arg(
            clap::Arg::new("json")
                .long("json")
                .global(true)
                .action(ArgAction::SetTrue)
                .help("Machine-readable output"),
        );
    let mut globals: Vec<&str> = Vec::new();
    for a in COMMANDS.iter().flat_map(|c| c.args).filter(|a| a.global) {
        if !globals.contains(&a.name) {
            globals.push(a.name);
            root = root.arg(clap_arg(a).global(true));
        }
    }
    for c in COMMANDS.iter() {
        match c.cli {
            Cli::Absent => {}
            Cli::NoSubcommand => {
                for a in c.args.iter().filter(|a| !a.global) {
                    root = root.arg(clap_arg(a));
                }
            }
            Cli::Subcommand | Cli::Hidden => {
                let sub = subcommand(c);
                root = match c.name.split_once(' ') {
                    None => root.subcommand(sub),
                    Some((group, _)) => {
                        if root.find_subcommand(group).is_none() {
                            let help = GROUPS
                                .iter()
                                .find(|g| g.name == group)
                                .map_or("", |g| g.help);
                            // A group whose commands are all hidden is hidden.
                            let hidden = COMMANDS
                                .iter()
                                .filter(|x| x.name.split_once(' ').is_some_and(|(g, _)| g == group))
                                .all(|x| x.cli == Cli::Hidden);
                            root = root.subcommand(
                                clap::Command::new(group)
                                    .about(help)
                                    .hide(hidden)
                                    .subcommand_required(true)
                                    .arg_required_else_help(true),
                            );
                        }
                        root.mut_subcommand(group, |g| g.subcommand(sub))
                    }
                };
            }
        }
    }
    root
}

fn subcommand(c: &Command) -> clap::Command {
    let word = c.name.rsplit(' ').next().unwrap_or(c.name);
    let mut sub = clap::Command::new(word)
        .about(c.help)
        .hide(c.cli == Cli::Hidden);
    for a in c.args.iter().filter(|a| !a.global) {
        sub = sub.arg(clap_arg(a));
    }
    sub
}

fn clap_arg(a: &crate::commands::Arg) -> clap::Arg {
    let mut arg = clap::Arg::new(a.name)
        .help(a.help)
        .value_name(a.name.to_uppercase());
    if !a.positional {
        arg = arg.long(a.name.replace('_', "-"));
        if let Some(s) = a.short {
            arg = arg.short(s);
        }
    }
    arg = match a.kind {
        Kind::Bool => arg.action(ArgAction::SetTrue),
        Kind::Integer => arg.value_parser(clap::value_parser!(u64)),
        Kind::Text => arg,
        Kind::Choice(choices) => arg.value_parser(clap::builder::PossibleValuesParser::new(
            choices.iter().copied(),
        )),
        Kind::Words => arg.num_args(1..),
        Kind::Rest => arg.num_args(1..).trailing_var_arg(true),
        Kind::List if a.positional => arg.num_args(1..),
        Kind::List => arg.action(ArgAction::Append).value_delimiter(','),
        Kind::FlagOrText => arg.num_args(0..=1),
    };
    if let Some(d) = a.default {
        arg = arg.default_value(d);
    }
    arg.required(a.required && a.default.is_none() && a.kind != Kind::Bool)
}

/// The table command the matches chose, and the matches holding its
/// arguments.
fn chosen(matches: &ArgMatches) -> Option<(&'static Command, &ArgMatches)> {
    let Some((word, sub)) = matches.subcommand() else {
        return COMMANDS
            .iter()
            .find(|c| c.cli == Cli::NoSubcommand)
            .map(|c| (c, matches));
    };
    let (name, leaf) = match sub.subcommand() {
        Some((second, leaf)) => (format!("{word} {second}"), leaf),
        None => (word.to_owned(), sub),
    };
    crate::commands::command(&name).map(|c| (c, leaf))
}

/// A command's values from its matches.
fn values(c: &Command, m: &ArgMatches) -> Values {
    let mut out = serde_json::Map::new();
    for a in c.args {
        let value = match a.kind {
            Kind::Bool => m.get_flag(a.name).then_some(Value::Bool(true)),
            Kind::Integer => m.get_one::<u64>(a.name).map(|n| Value::from(*n)),
            Kind::Text | Kind::Choice(_) => m.get_one::<String>(a.name).cloned().map(Value::String),
            Kind::Words | Kind::Rest => m
                .get_many::<String>(a.name)
                .map(|words| Value::String(words.cloned().collect::<Vec<_>>().join(" "))),
            Kind::List => m
                .get_many::<String>(a.name)
                .map(|vs| Value::Array(vs.cloned().map(Value::String).collect())),
            Kind::FlagOrText => {
                (m.value_source(a.name) == Some(ValueSource::CommandLine)).then(|| {
                    m.get_one::<String>(a.name)
                        .map_or(Value::Bool(true), |s| Value::String(s.clone()))
                })
            }
        };
        if let Some(v) = value {
            out.insert(a.name.to_owned(), v);
        }
    }
    Values(out)
}

/// The command and values a command line names, for tests.
#[cfg(test)]
pub fn parse(argv: &[&str]) -> Result<(&'static Command, Values), String> {
    let matches = command()
        .try_get_matches_from(argv)
        .map_err(|e| e.to_string())?;
    let (c, m) = chosen(&matches).ok_or("no command")?;
    Ok((c, values(c, m)))
}

pub fn run(matches: &ArgMatches) -> ExitCode {
    let json = matches.get_flag("json");
    let Some((cmd, m)) = chosen(matches) else {
        return fail(json, "bad_request", "no such command", 2);
    };
    let values = values(cmd, m);
    let build = match cmd.action {
        Action::Local(f) => return f(cmd, &values, json),
        Action::Request(build) => build,
        Action::JsonRequest(_) | Action::BytesRequest(_) => {
            return fail(json, "bad_request", "this command is HTTP-only", 2);
        }
    };
    let caller = Caller {
        surface: Surface::Cli,
        cwd: Some(current_dir()),
        json,
    };
    let request = match build(cmd, &values, &caller) {
        Ok(r) => r,
        Err(message) => return fail(json, "bad_request", &message, 2),
    };
    let root = Archive::default_root();
    if matches!(request, Request::Changes { .. }) {
        return match changes::send(&Archive::at(&root), &request, &mut std::io::stdout().lock()) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(e) => fail(true, "io", &e.to_string(), 1),
        };
    }
    let response = match send_when_ready(&Archive::at(&root), &request) {
        Ok(r) => r,
        Err(e) if is_down(&e) => {
            return fail(
                json,
                "daemon_required",
                &format!(
                    "no daemon is serving {}; start it with `polylogue daemon`",
                    root.display()
                ),
                crate::EXIT_DAEMON_REQUIRED,
            );
        }
        Err(e) => return fail(json, "io", &e.to_string(), 1),
    };
    match response {
        Response::Ok { data } => match cmd.output {
            Some(output) => output(&values, &data, json),
            None => {
                if json {
                    println!("{data}");
                } else {
                    print!("{}", (cmd.print)(&data));
                }
                ExitCode::SUCCESS
            }
        },
        Response::Accepted { request_id } => {
            println!("accepted {request_id}");
            ExitCode::SUCCESS
        }
        Response::Error {
            kind,
            message,
            detail,
        } => fail_detail(json, &kind, &message, detail, 1),
    }
}

/// `export`: the document written to `--out`, or to standard output.
pub fn write_export(values: &Values, data: &Value, json: bool) -> ExitCode {
    let out = values.text("out").map(PathBuf::from);
    let out = out.as_deref();
    let text = match data.get("content").and_then(|c| c.as_str()) {
        Some(content) => content.to_owned(),
        None => match serde_json::to_string_pretty(data) {
            Ok(t) => t,
            Err(e) => return fail(json, "internal", &e.to_string(), 1),
        },
    };
    match out {
        Some(path) => match std::fs::write(path, text.as_bytes()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => fail(json, "io", &format!("{}: {e}", path.display()), 1),
        },
        None => {
            print!("{text}");
            ExitCode::SUCCESS
        }
    }
}

fn current_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
}

/// `continue --exec`: replaces this process with the session's native
/// resume command; without `--exec`, the plan is printed.
pub fn resume_output(values: &Values, data: &Value, json: bool) -> ExitCode {
    if !values.flag("exec") {
        if json {
            println!("{data}");
        } else {
            print!("{}", print::resume(data));
        }
        return ExitCode::SUCCESS;
    }
    exec_plan(data)
}

fn exec_plan(data: &Value) -> ExitCode {
    use std::os::unix::process::CommandExt;
    let Some(native) = data.get("native").filter(|n| !n.is_null()) else {
        eprintln!(
            "polylogue: this session has no native resume; continue with context instead: {}",
            data["context_assisted"]["program"]
                .as_str()
                .map_or_else(String::new, |_| format!(
                    "polylogue takeover {}",
                    data["session"].as_str().unwrap_or("")
                ))
        );
        return ExitCode::from(1);
    };
    let program = native["program"].as_str().unwrap_or("");
    let args: Vec<&str> = native["args"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|a| a.as_str())
        .collect();
    let mut cmd = std::process::Command::new(program);
    cmd.args(&args);
    if let Some(dir) = native["cwd"].as_str() {
        cmd.current_dir(dir);
    }
    let err = cmd.exec();
    eprintln!("polylogue: could not run {program}: {err}");
    ExitCode::FAILURE
}

fn is_down(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
    )
}

/// Sends one request, waiting for a daemon that is still starting: one
/// that holds the archive lock but has not bound its socket yet (opening a
/// large archive takes seconds). With no daemon at all, the error is
/// returned at once.
pub fn send_when_ready(archive: &Archive, request: &Request) -> std::io::Result<Response> {
    when_ready(archive, || send(&archive.socket(), request))
}

fn when_ready<T>(
    archive: &Archive,
    mut call: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    let mut pause = std::time::Duration::from_millis(50);
    let mut told = false;
    loop {
        match call() {
            Err(e) if is_down(&e) && archive.lock_held() => {
                if !told {
                    eprintln!("polylogue: waiting for the daemon to finish opening the archive");
                    told = true;
                }
                std::thread::sleep(pause);
                pause = (pause * 2).min(std::time::Duration::from_millis(500));
            }
            other => return other,
        }
    }
}

/// Sends one request and reads its response.
pub fn send(socket: &std::path::Path, request: &Request) -> std::io::Result<Response> {
    if matches!(request, Request::Changes { .. }) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "the change feed requires its finite stream consumer",
        ));
    }
    let mut stream = UnixStream::connect(socket)?;
    let mut line = serde_json::to_string(request).map_err(std::io::Error::other)?;
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    let mut reply = String::new();
    BufReader::new(stream).read_line(&mut reply)?;
    serde_json::from_str(&reply).map_err(std::io::Error::other)
}

fn fail(json: bool, kind: &str, message: &str, code: u8) -> ExitCode {
    if json {
        println!(
            "{}",
            serde_json::json!({ "error": { "kind": kind, "message": message } })
        );
    } else {
        eprintln!("polylogue: {kind}: {message}");
    }
    ExitCode::from(code)
}

/// An error with structured detail: printed whole for programs; for a
/// person, the message and the corrected query when there is one.
fn fail_detail(
    json: bool,
    kind: &str,
    message: &str,
    detail: Option<serde_json::Value>,
    code: u8,
) -> ExitCode {
    if json {
        println!(
            "{}",
            serde_json::json!({ "error": { "kind": kind, "message": message, "detail": detail } })
        );
    } else {
        eprintln!("polylogue: {kind}: {message}");
        if let Some(s) = detail.as_ref().and_then(|d| d["suggestion"].as_str()) {
            eprintln!("polylogue: try: {s}");
        }
    }
    ExitCode::from(code)
}

#[cfg(test)]
mod tests;
