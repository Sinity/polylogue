//! The command line, built from the command table. A bare query
//! (`polylogue repo:x since:7d`) runs the query language; subcommands cover
//! everything else. Every command takes `--json` for programs and agents.

mod attachment;
pub mod changes;
pub(crate) mod completion;
pub mod local;
pub mod print;
pub(crate) mod schema;

use crate::commands::{Action, COMMANDS, Caller, Cli, Command, GROUPS, Kind, Surface, Values};
use crate::daemon::{Request, Response};
use clap::ArgAction;
use clap::parser::{ArgMatches, ValueSource};
use polylogue_store::Archive;
use serde_json::Value;
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
    root = root.arg(
        clap::Arg::new("timeout")
            .long("timeout")
            .global(true)
            .value_name("SECONDS")
            .value_parser(clap::value_parser!(u64).range(1..))
            .help(
                "Daemon read timeout in seconds (default: 30, or POLYLOGUE_READ_TIMEOUT_MS; export and schema infer: none)",
            ),
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
            Cli::NoSubcommand => {
                for a in c.args_for(Surface::Cli).filter(|a| !a.global) {
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
    for a in c.args_for(Surface::Cli).filter(|a| !a.global) {
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
    if a.name == "query" && matches!(a.kind, Kind::Words | Kind::Rest) {
        arg = arg.add(clap_complete::engine::ArgValueCompleter::new(
            completion::Query,
        ));
    }
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
    for a in c.args_for(Surface::Cli) {
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
    if let Some(seconds) = m.get_one::<u64>("timeout") {
        out.insert("timeout".into(), Value::from(*seconds));
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
        Action::Request(build) | Action::Stream { build, .. } => build,
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
    let timeout = matches
        .get_one::<u64>("timeout")
        .copied()
        .map(std::time::Duration::from_secs);
    if let Action::Stream { cli, .. } = cmd.action {
        return cli(&values, &request, timeout);
    }
    let root = Archive::default_root();
    let result = if cmd.register.is_some() {
        let deadline = timeout.map_or(command_surface::Deadline::Default, |duration| {
            command_surface::Deadline::Milliseconds(
                duration.as_millis().try_into().unwrap_or(u64::MAX),
            )
        });
        let call = crate::protocol::operations::request(
            cmd,
            values.clone(),
            values.request_id(),
            deadline,
        );
        crate::client::call_ready(&Archive::at(&root), &call)
    } else {
        send_with_timeout(&Archive::at(&root), &request, timeout).map(|response| response.answer())
    };
    let answer = match result {
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
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            ) =>
        {
            return fail(json, "deadline_exceeded", &e.to_string(), 7);
        }
        Err(e) => return fail(json, "io", &e.to_string(), 1),
    };
    let outcome_code = answer.exit_code();
    if json && cmd.output.is_none() {
        println!(
            "{}",
            polylogue_render::terminal::json(
                &serde_json::to_value(&answer).expect("answer serializes"),
                false
            )
        );
        return ExitCode::from(outcome_code);
    }
    let response: Response = answer.into();
    match response {
        Response::Ok { data, .. } => {
            if !json && matches!(data["shape"].as_str(), Some("rows" | "matches")) {
                crate::row_numbers::remember(&root, &crate::row_numbers::row_ids(&data["rows"]));
            }
            match cmd.output {
                Some(output) => output(&values, &data, json),
                None => {
                    if json {
                        println!(
                            "{}",
                            polylogue_render::terminal::json(
                                &serde_json::to_value(crate::protocol::answer(data.clone()))
                                    .expect("answer serializes"),
                                false
                            )
                        );
                    } else {
                        print!("{}", (cmd.print)(&data));
                    }
                    ExitCode::from(outcome_code)
                }
            }
        }
        Response::Error {
            kind,
            message,
            detail,
        } => {
            let code = crate::daemon::execution::exit_code(&kind);
            fail_detail(json, &kind, &message, detail, code)
        }
    }
}

/// `export`: the document written to `--out`, or to standard output.
pub fn write_export(values: &Values, data: &Value, json: bool) -> ExitCode {
    let out = values.text("out").map(PathBuf::from);
    let out = out.as_deref();
    let text = match data.get("content").and_then(|c| c.as_str()) {
        Some(content) => content.to_owned(),
        None => polylogue_render::terminal::json(data, true),
    };
    match out {
        Some(path) => match std::fs::write(path, text.as_bytes()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => fail(json, "io", &format!("{}: {e}", path.display()), 1),
        },
        None => {
            print!("{}", polylogue_render::terminal::stdout_text(&text));
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
            println!("{}", polylogue_render::terminal::json(data, false));
        } else {
            print!("{}", print::resume(data));
        }
        return ExitCode::SUCCESS;
    }
    exec_plan(data)
}

fn exec_plan(data: &Value) -> ExitCode {
    use std::os::unix::process::CommandExt;
    let Some(native) = data.get("harness_command").filter(|n| !n.is_null()) else {
        eprintln!(
            "{}",
            polylogue_render::terminal::escape(&format!(
                "polylogue: this session has no native resume; continue with context instead: {}",
                data["context_assisted"]["program"]
                    .as_str()
                    .map_or_else(String::new, |_| format!(
                        "polylogue takeover {}",
                        data["session"].as_str().unwrap_or("")
                    ))
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
    eprintln!(
        "{}",
        polylogue_render::terminal::escape(&format!("polylogue: could not run {program}: {err}"))
    );
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
fn send_with_timeout(
    archive: &Archive,
    request: &Request,
    timeout: Option<std::time::Duration>,
) -> std::io::Result<Response> {
    let deadline_ms = crate::daemon::execution::request_deadline(
        request,
        timeout.map(crate::daemon::execution::deadline),
    );
    crate::client::send_when_ready(archive, request, deadline_ms)
}

pub(crate) fn wait_for_ready<T>(
    archive: &Archive,
    deadline_ms: Option<i64>,
    mut call: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?
        .block_on(when_ready(archive, deadline_ms, || {
            std::future::ready(call())
        }))
}

pub(crate) async fn when_ready<T, F: std::future::Future<Output = std::io::Result<T>>>(
    archive: &Archive,
    deadline_ms: Option<i64>,
    mut call: impl FnMut() -> F,
) -> std::io::Result<T> {
    let mut pause = std::time::Duration::from_millis(50);
    let mut told = false;
    let started = std::time::Instant::now();
    let duration =
        deadline_ms.map(|deadline_ms| crate::daemon::execution::remaining(Some(deadline_ms)));
    loop {
        match call().await {
            Err(e) if is_down(&e) && archive.lock_held() => {
                if duration.is_some_and(|duration| started.elapsed() >= duration) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "the daemon did not start before the deadline",
                    ));
                }
                if !told {
                    eprintln!("polylogue: waiting for the daemon to finish opening the archive");
                    told = true;
                }
                tokio::time::sleep(pause).await;
                pause = (pause * 2).min(std::time::Duration::from_millis(500));
            }
            other => return other,
        }
    }
}

pub(crate) fn fail(json: bool, kind: &str, message: &str, code: u8) -> ExitCode {
    fail_detail(json, kind, message, None, code)
}

/// An error with structured detail: printed whole for programs; for a
/// person, the message and the corrected query when there is one.
fn fail_detail(
    json: bool,
    kind: &str,
    message: &str,
    detail: Option<serde_json::Value>,
    _code: u8,
) -> ExitCode {
    if json {
        println!(
            "{}",
            polylogue_render::terminal::json(
                &serde_json::to_value(Response::Error {
                    kind: kind.into(),
                    message: message.into(),
                    detail: detail.clone()
                })
                .expect("failure serializes"),
                false
            )
        );
    } else {
        eprintln!(
            "{}",
            polylogue_render::terminal::escape(&format!("polylogue: {kind}: {message}"))
        );
        if let Some(s) = detail.as_ref().and_then(|d| d["suggestion"].as_str()) {
            eprintln!(
                "{}",
                polylogue_render::terminal::escape(&format!("polylogue: try: {s}"))
            );
        }
    }
    ExitCode::from(crate::protocol::kind(kind).exit_code())
}

#[cfg(test)]
mod tests;
