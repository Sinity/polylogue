//! The command line, built from the command table. A bare query
//! (`polylogue repo:x since:7d`) runs the query language; subcommands cover
//! everything else. Every command takes `--json` for programs and agents.

pub(crate) mod attachment;
pub mod changes;
pub(crate) mod completion;
pub(crate) mod follow;
pub mod local;
pub mod print;
pub(crate) mod schema;
pub(crate) mod staged_file;
pub(crate) mod write_out;

use crate::commands::{COMMANDS, Cli, Command, GROUPS, Kind, Run, Surface, Values};
use clap::ArgAction;
use clap::parser::{ArgMatches, ValueSource};
use command_surface::Answer;
use polylogue_store::Archive;
use serde_json::Value;
use std::process::ExitCode;

fn version() -> String {
    let revision = option_env!("POLYLOGUE_REVISION").unwrap_or("unavailable");
    format!("{} (revision {revision})", env!("CARGO_PKG_VERSION"))
}

/// The clap command tree: one subcommand per table command, grouped by
/// their first word, and the bare query as the root's own arguments.
pub fn command() -> clap::Command {
    let mut root = clap::Command::new("polylogue")
        .color(clap::ColorChoice::Never)
        .styles(clap::builder::Styles::plain())
        .version(version())
        .about("Search, read and measure your AI sessions")
        .arg(
            clap::Arg::new("json")
                .long("json")
                .global(true)
                .action(ArgAction::SetTrue)
                .help("Machine-readable output"),
        );
    root = root
        .arg(
            clap::Arg::new("request")
                .long("request")
                .global(true)
                .value_name("JSON")
                .help("The operation's complete typed argument object"),
        )
        .arg(
            clap::Arg::new("request_id")
                .long("request-id")
                .global(true)
                .value_name("ID")
                .help("The durable request identity"),
        )
        .arg(
            clap::Arg::new("deadline")
                .long("deadline")
                .global(true)
                .value_name("MS|none")
                .help("Milliseconds from when the daemon accepts the request, or none to lift the deadline"),
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
                root = root.subcommand(subcommand(c).name("find"));
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
    crate::config::augment_command(root)
}

fn subcommand(c: &Command) -> clap::Command {
    let word = c.name.rsplit(' ').next().unwrap_or(c.name);
    let mut sub = clap::Command::new(word)
        .about(c.help)
        .hide(c.cli == Cli::Hidden);
    for a in c.args_for(Surface::Cli).filter(|a| !a.global) {
        let mut arg = clap_arg(a);
        if c.name == "setting" && matches!(a.name, "key" | "value") {
            arg = arg.add(clap_complete::engine::ArgValueCompleter::new(
                completion::Setting {
                    key: a.name == "key",
                },
            ));
        }
        sub = sub.arg(arg);
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
        Kind::Json => {
            arg.value_parser(|text: &str| serde_json::from_str::<serde_json::Value>(text))
        }
        Kind::Choice(choices) => arg.value_parser(clap::builder::PossibleValuesParser::new(
            choices.iter().copied(),
        )),
        Kind::Words => arg.num_args(1..),
        Kind::Rest => arg.num_args(1..),
        Kind::List if a.positional => arg.num_args(1..),
        Kind::List => arg.action(ArgAction::Append).value_delimiter(','),
    };
    if a.name == "budget" {
        arg = arg.value_parser(|value: &str| parse_budget(Some(value)).map(|_| value.to_owned()));
    }
    if a.name == "query" && matches!(a.kind, Kind::Words | Kind::Rest) {
        arg = arg.add(clap_complete::engine::ArgValueCompleter::new(
            completion::Query,
        ));
    }
    if let Some(d) = a.default {
        arg = arg.default_value(d);
    }
    if a.required && a.default.is_none() && a.kind != Kind::Bool {
        arg.required_unless_present("request")
    } else {
        arg
    }
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
    if word == "find" {
        return crate::commands::command("query").map(|command| (command, sub));
    }
    let (name, leaf) = match sub.subcommand() {
        Some((second, leaf)) => (format!("{word} {second}"), leaf),
        None => (word.to_owned(), sub),
    };
    crate::commands::command(&name).map(|c| (c, leaf))
}

/// Only configured local commands that construct an Engine need its binary identity.
pub(crate) fn needs_engine(
    matches: &ArgMatches,
    mode: &crate::daemon::service::ArchiveMode,
) -> bool {
    chosen(matches).is_some_and(|(command, _)| match command.name {
        "daemon" => mode.needs_engine(),
        "demo tour" => true,
        _ => false,
    })
}

/// A command's values from its matches.
fn values(c: &Command, m: &ArgMatches) -> Values {
    let mut out = serde_json::Map::new();
    for a in c.args_for(Surface::Cli) {
        let value = match a.kind {
            Kind::Bool => m.get_flag(a.name).then_some(Value::Bool(true)),
            Kind::Integer => m.get_one::<u64>(a.name).map(|n| Value::from(*n)),
            Kind::Json => m.get_one::<Value>(a.name).cloned(),
            Kind::Text | Kind::Choice(_) => m.get_one::<String>(a.name).cloned().map(Value::String),
            Kind::Words | Kind::Rest => m
                .get_many::<String>(a.name)
                .map(|words| Value::String(words.cloned().collect::<Vec<_>>().join(" "))),
            Kind::List => m
                .get_many::<String>(a.name)
                .map(|vs| Value::Array(vs.cloned().map(Value::String).collect())),
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

/// The command a command line names and the request arguments it sends,
/// prepared in `cwd`, for tests.
#[cfg(test)]
pub fn request_arguments(
    argv: &[&str],
    cwd: &std::path::Path,
) -> Result<(&'static Command, Values, serde_json::Map<String, Value>), String> {
    let matches = command()
        .try_get_matches_from(argv)
        .map_err(|e| e.to_string())?;
    let (c, m) = chosen(&matches).ok_or("no command")?;
    let values = values(c, m);
    let mut arguments = arguments(c, &values, m);
    if let Some(prepare) = c.prepare {
        prepare(&values, cwd, &mut arguments)?;
    }
    Ok((c, values, arguments))
}

fn parse_text(error: &clap::Error, output: crate::config::OutputConfig) -> String {
    let color = if error.use_stderr() {
        output.stderr_color()
    } else {
        output.stdout_color()
    };
    polylogue_render::terminal::heading(&error.render().to_string(), color)
}

pub(crate) fn parse_error(error: clap::Error, output: crate::config::OutputConfig) -> ExitCode {
    let text = parse_text(&error, output);
    if error.use_stderr() {
        eprint!("{text}");
    } else {
        print!("{text}");
    }
    ExitCode::from(u8::try_from(error.exit_code()).expect("Clap exit code fits u8"))
}

pub fn run_local(matches: &ArgMatches, output: crate::config::OutputConfig) -> Option<ExitCode> {
    let (command, matches) = chosen(matches)?;
    let Run::Local(run) = command.run else {
        return None;
    };
    Some(run(
        command,
        &values(command, matches),
        output,
        matches.get_flag("json"),
    ))
}

/// A request's token budget as the command line writes it.
pub(crate) fn parse_budget(value: Option<&str>) -> Result<command_surface::TokenBudget, String> {
    match value {
        None => Ok(command_surface::TokenBudget::Default),
        Some("none") => Ok(command_surface::TokenBudget::Unlimited),
        Some(value) => value
            .parse::<u32>()
            .map(command_surface::TokenBudget::Tokens)
            .map_err(|_| "budget must be a token count or none".into()),
    }
}

/// The request arguments the command line names: the operation's own
/// arguments given on it, without those that act on the calling machine.
fn arguments(
    command: &Command,
    values: &Values,
    matches: &ArgMatches,
) -> serde_json::Map<String, Value> {
    values
        .0
        .iter()
        .filter(|(name, _)| {
            matches.value_source(name) == Some(ValueSource::CommandLine)
                && command.args.iter().any(|argument| {
                    argument.name == name.as_str()
                        && !argument.cli_only
                        && !matches!(argument.name, "cursor" | "budget")
                })
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

pub fn run(
    matches: &ArgMatches,
    startup: &crate::config::Startup,
    cwd: &std::path::Path,
) -> ExitCode {
    let policy = startup.output;
    let json = matches.get_flag("json");
    let Some((cmd, m)) = chosen(matches) else {
        return fail(policy, json, "bad_request", "no such command", 2);
    };
    let mut values = values(cmd, m);
    values.0.insert("json".into(), Value::Bool(json));
    match cmd.run {
        Run::Local(run) => return run(cmd, &values, policy, json),
        Run::ConfiguredLocal(run) => return run(cmd, &values, startup, json),
        Run::Daemon(_) | Run::Delivered { .. } | Run::Feed { .. } => {}
    }
    let raw = m.get_one::<String>("request");
    if raw.is_some()
        && cmd.args_for(Surface::Cli).any(|argument| {
            !matches!(argument.name, "cursor" | "budget")
                && m.value_source(argument.name) == Some(ValueSource::CommandLine)
        })
    {
        return fail(
            policy,
            json,
            "bad_request",
            "--request cannot be mixed with operation arguments",
            2,
        );
    }
    let deadline = match m.get_one::<String>("deadline").map(String::as_str) {
        None => command_surface::Deadline::Default,
        Some("none") => command_surface::Deadline::None,
        Some(value) => match value.parse() {
            Ok(milliseconds) => command_surface::Deadline::Milliseconds(milliseconds),
            Err(_) => {
                return fail(
                    policy,
                    json,
                    "bad_request",
                    "deadline must be milliseconds or none",
                    2,
                );
            }
        },
    };
    let request_id = m
        .get_one::<String>("request_id")
        .cloned()
        .unwrap_or_else(polylogue_store::user_writes::new_request_id);
    let budget_tokens = match parse_budget(m.get_one::<String>("budget").map(String::as_str)) {
        Ok(tokens) => tokens,
        Err(error) => return fail(policy, json, "bad_request", &error, 2),
    };
    let cursor = values.text("cursor");
    let arguments = match raw {
        Some(raw) => match serde_json::from_str::<Value>(raw) {
            Ok(Value::Object(arguments)) => {
                // Output options may read the arguments, while the request
                // keeps exactly what the caller sent.
                values = Values(arguments.clone());
                values.0.insert("json".into(), Value::Bool(json));
                arguments
            }
            Ok(_) => {
                return fail(
                    policy,
                    json,
                    "bad_request",
                    "operation arguments must be an object",
                    2,
                );
            }
            Err(error) => return fail(policy, json, "bad_request", &error.to_string(), 2),
        },
        None => {
            let mut arguments = arguments(cmd, &values, m);
            if let Some(prepare) = cmd.prepare
                && let Err(message) = prepare(&values, cwd, &mut arguments)
            {
                return fail(policy, json, "bad_request", &message, 2);
            }
            arguments
        }
    };
    let call = command_surface::Request {
        operation: cmd.name.into(),
        arguments: Value::Object(arguments),
        request_id,
        deadline,
        budget_tokens,
        cursor,
        fresh: false,
        format: Default::default(),
    };
    if let Run::Delivered { deliver, .. } | Run::Feed { deliver, .. } = cmd.run {
        return deliver(&values, startup, &call);
    }
    let root = startup.config.archive.clone();
    let answer = match crate::client::call_ready(&Archive::at(&root), &call) {
        Ok(r) => r,
        Err(e) if is_down(&e) => {
            return fail(
                policy,
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
            return fail(policy, json, "deadline_exceeded", &e.to_string(), 7);
        }
        Err(e) => return fail(policy, json, "io", &e.to_string(), 1),
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
    if let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        envelope: command_surface::Envelope { notes, .. },
        ..
    } = &answer
    {
        return fail_with(policy, json, error, notes);
    }
    // JSON passes the answer through as received; anything that reads the
    // value first checks it is the operation's declared type.
    if let Err(failure) = crate::commands::declarations().validate_answer(cmd.name, &answer) {
        let mismatch = crate::protocol::protocol_mismatch(cmd.name, failure.message);
        return fail_with(policy, json, &mismatch, &[]);
    }
    let data = answer.value().expect("a non-error answer has a value");
    if !json && matches!(data["shape"].as_str(), Some("rows" | "matches")) {
        crate::row_numbers::remember(&root, &crate::row_numbers::row_ids(&data["rows"]));
    }
    match cmd.output {
        Some(output) => output(&values, &answer, startup, json),
        None => {
            let style = if policy.stdout_color() {
                command_surface::TextStyle::Ansi
            } else {
                command_surface::TextStyle::Plain
            };
            match crate::commands::declarations().render_text(cmd.name, &answer, style) {
                Ok(text) => print!("{text}"),
                Err(failure) => return fail_with(policy, json, &failure, &[]),
            }
            notes(
                &crate::protocol::notes(&answer).cloned().collect::<Vec<_>>(),
                policy,
            );
            ExitCode::from(outcome_code)
        }
    }
}

/// Notes go to standard error, so standard output stays the value alone.
fn notes(notes: &[String], output: crate::config::OutputConfig) {
    for note in notes {
        eprint!(
            "{}",
            print::diagnostic("note", note, None, output.stderr_color())
        );
    }
}

/// `status`: the whole answer, or with `--line` the one line a status bar
/// shows (`3 running · 1 waiting 7m`).
pub fn status_output(
    values: &Values,
    answer: &Answer<Value>,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    let data = answer
        .value()
        .expect("output handlers receive answers with values");
    if !values.flag("line") {
        if json {
            println!(
                "{}",
                polylogue_render::terminal::json(
                    &serde_json::to_value(answer).expect("answer serializes"),
                    false
                )
            );
        } else {
            print!("{}", print::json(data, startup.output.stdout_color()));
        }
        return ExitCode::from(answer.exit_code());
    }
    let running: crate::protocol::status::Running =
        match serde_json::from_value(data["running"].clone()) {
            Ok(running) => running,
            Err(error) => {
                return fail(startup.output, json, "internal", &error.to_string(), 1);
            }
        };
    let line = running.line(startup.clock.now().as_millisecond());
    if json {
        println!("{}", serde_json::json!({ "line": line }));
    } else {
        println!("{line}");
    }
    ExitCode::from(answer.exit_code())
}

/// `continue --exec`: replaces this process with the session's native
/// resume command; without `--exec`, the plan is printed.
pub fn resume_output(
    values: &Values,
    answer: &Answer<Value>,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    let data = answer
        .value()
        .expect("output handlers receive answers with values");
    if !values.flag("exec") {
        if json {
            println!("{}", polylogue_render::terminal::json(data, false));
        } else {
            print!("{}", print::resume(data, startup.output.stdout_color()));
        }
        return ExitCode::SUCCESS;
    }
    exec_plan(data, startup.output)
}

fn exec_plan(data: &Value, output: crate::config::OutputConfig) -> ExitCode {
    use std::os::unix::process::CommandExt;
    let Some(native) = data.get("harness_command").filter(|n| !n.is_null()) else {
        eprint!(
            "{}",
            print::diagnostic(
                "",
                &format!(
                    "this session has no native resume; continue with context instead: {}",
                    data["context_assisted"]["program"].as_str().map_or_else(
                        String::new,
                        |_| format!(
                            "polylogue takeover {}",
                            data["session"].as_str().unwrap_or("")
                        )
                    )
                ),
                None,
                output.stderr_color()
            )
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
    eprint!(
        "{}",
        print::diagnostic(
            "",
            &format!("could not run {program}: {err}"),
            None,
            output.stderr_color()
        )
    );
    ExitCode::FAILURE
}

fn is_down(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
    )
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

/// A failure made here, before or instead of a daemon answer.
pub(crate) fn fail(
    output: crate::config::OutputConfig,
    json: bool,
    code: &str,
    message: &str,
    _exit: u8,
) -> ExitCode {
    fail_with(output, json, &crate::protocol::failure(code, message), &[])
}

/// One printer for every failure: the whole answer for programs; for a
/// person, the code, the message, the corrected query and the notes.
fn fail_with(
    output: crate::config::OutputConfig,
    json: bool,
    failure: &command_surface::Failure,
    notes: &[String],
) -> ExitCode {
    if json {
        let answer = command_surface::Answer::<Value> {
            outcome: command_surface::Outcome::Error(failure.clone()),
            envelope: command_surface::Envelope {
                notes: notes.to_vec(),
                ..Default::default()
            },
        };
        println!(
            "{}",
            polylogue_render::terminal::json(
                &serde_json::to_value(answer).expect("failure serializes"),
                false
            )
        );
    } else {
        let code = failure.code.clone().unwrap_or_else(|| {
            serde_json::to_value(failure.kind)
                .ok()
                .and_then(|kind| kind.as_str().map(str::to_owned))
                .unwrap_or_default()
        });
        eprint!(
            "{}",
            print::diagnostic(
                &code,
                &failure.message,
                crate::protocol::suggestion_text(failure).as_deref(),
                output.stderr_color()
            )
        );
        for candidate in failure.candidates() {
            eprintln!("  {}", polylogue_render::terminal::escape(candidate));
        }
        self::notes(notes, output);
    }
    ExitCode::from(failure.kind.exit_code())
}

#[cfg(test)]
mod tests;
