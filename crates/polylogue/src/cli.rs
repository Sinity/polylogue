//! The command line, built from the command table. A bare query
//! (`polylogue repo:x since:7d`) runs the query language; subcommands cover
//! everything else. Every command takes `--json` for programs and agents.

use polylogue_model::JsonExt as _;
pub(crate) mod attachment;
pub mod changes;
pub(crate) mod completion;
pub(crate) mod follow;
pub mod local;
pub mod print;
pub(crate) mod schema;
pub(crate) mod staged_file;
pub(crate) mod write_out;

use crate::commands::{COMMANDS, Cli, Command, GROUPS, Run, Surface};
use clap::ArgAction;
use clap::parser::{ArgMatches, ValueSource};
use command_surface::{Answer, ArgumentForm, CliArgument, ParsedCommand};
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
                .help("The request identity, kept across retries"),
        )
        .arg(
            clap::Arg::new("deadline")
                .long("deadline")
                .global(true)
                .value_name("MS|none")
                .help("Milliseconds from when the daemon accepts the request, or none to lift the deadline"),
        );
    // Each continuation starts a client: project each command once, not once per flag.
    let projected: Vec<_> = COMMANDS.iter().map(arguments_of).collect();
    let mut globals: Vec<&str> = Vec::new();
    for (command, arguments) in COMMANDS.iter().zip(&projected) {
        for a in command.args.iter().filter(|a| a.global) {
            if !globals.contains(&a.name) {
                globals.push(a.name);
                let argument = arguments
                    .iter()
                    .find(|argument| argument.spec.name == a.name)
                    .expect("global argument is projected");
                root = root.arg(clap_arg(argument).global(true));
            }
        }
    }
    for (c, arguments) in COMMANDS.iter().zip(&projected) {
        match c.cli {
            Cli::NoSubcommand => {
                for a in c.args_for(Surface::Cli).filter(|a| !a.global) {
                    root = root.arg(clap_arg(
                        arguments
                            .iter()
                            .find(|argument| argument.spec.name == a.name)
                            .expect("root argument is projected"),
                    ));
                }
                root = root.subcommand(subcommand(c, arguments).name("find"));
            }
            Cli::Subcommand | Cli::Hidden => {
                let sub = subcommand(c, arguments);
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

fn subcommand(c: &Command, arguments: &[CliArgument]) -> clap::Command {
    let word = c.name.rsplit(' ').next().unwrap_or(c.name);
    let mut sub = clap::Command::new(word)
        .about(c.help)
        .hide(c.cli == Cli::Hidden);
    for a in c.args_for(Surface::Cli).filter(|a| !a.global) {
        let mut arg = clap_arg(
            arguments
                .iter()
                .find(|argument| argument.spec.name == a.name)
                .expect("command argument is projected"),
        );
        if let Some(value) = a.default {
            arg = arg.default_value(value);
        }
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

/// The product keeps lexical exposure and completion on its command row;
/// value kinds, choices and requiredness come from the shared input schema.
pub(crate) fn arguments_of(command: &Command) -> Vec<CliArgument> {
    let declared = crate::commands::declarations()
        .cli_arguments(command.name)
        .expect("declared CLI arguments");
    command
        .args
        .iter()
        .map(|argument| {
            if matches!(argument.name, "cursor" | "budget") {
                return CliArgument {
                    spec: command_surface::ArgumentSpec::new(
                        argument.name,
                        ArgumentForm::Named,
                        argument.help,
                    ),
                    kind: command_surface::ValueKind::Text,
                    required: false,
                    choices: vec![],
                };
            }
            let mut projected = declared
                .iter()
                .find(|item| item.spec.name == argument.name)
                .expect("CLI field belongs to its input")
                .clone();
            // A named measure's series come from that measure's domain catalog;
            // its shared request type also serves measures with free-text options.
            if argument.name == "metric"
                && projected.choices.is_empty()
                && let Some(name) = command.name.strip_prefix("measure ")
                && let Some(measure) = crate::daemon::measure::MEASURES
                    .iter()
                    .find(|measure| measure.name == name)
            {
                projected.choices = measure
                    .metrics
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect();
            }
            if projected.choices.is_empty() {
                if command.name == "explain" && argument.name == "subject" {
                    projected.choices = crate::explain::SUBJECTS
                        .iter()
                        .map(|subject| (*subject).to_owned())
                        .collect();
                } else if (command.name == "start" && argument.name == "harness")
                    || (command.name == "hand-off" && argument.name == "to")
                {
                    projected.choices = polylogue_model::live::channel::HARNESSES
                        .iter()
                        .filter(|harness| harness.host.is_some())
                        .map(|harness| harness.name.to_owned())
                        .collect();
                }
            }
            projected
        })
        .collect()
}

fn clap_arg(a: &CliArgument) -> clap::Arg {
    let positional = matches!(
        a.spec.form,
        ArgumentForm::Positional | ArgumentForm::Words | ArgumentForm::Rest
    );
    let mut arg = clap::Arg::new(a.spec.name.clone())
        .help(a.spec.help.clone())
        .value_name(a.spec.name.to_uppercase());
    if !positional {
        arg = arg.long(a.spec.name.replace('_', "-"));
        if let Some(short) = a.spec.short {
            arg = arg.short(short);
        }
    }
    arg = match a.spec.form {
        ArgumentForm::Flag => arg.action(ArgAction::SetTrue),
        ArgumentForm::Words | ArgumentForm::Rest => arg.num_args(1..),
        ArgumentForm::Repeated => arg.action(ArgAction::Append),
        _ => arg,
    };
    if let Some(delimiter) = a.spec.value_delimiter {
        arg = arg.value_delimiter(delimiter);
    }
    if !a.choices.is_empty() {
        arg = arg.value_parser(clap::builder::PossibleValuesParser::new(a.choices.clone()));
    }
    if a.spec.name == "query" && matches!(a.spec.form, ArgumentForm::Words | ArgumentForm::Rest) {
        arg = arg.add(clap_complete::engine::ArgValueCompleter::new(
            completion::Query,
        ));
    }
    if a.required && a.spec.form != ArgumentForm::Flag {
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

/// Clap owns presentation, OS-valued startup options and lexical placement.
/// The shared parser alone turns the accepted words into values and an envelope.
fn parsed(c: &Command, m: &ArgMatches) -> Result<ParsedCommand, String> {
    let mut words = c.cli_words();
    for name in [
        "json",
        "request",
        "request_id",
        "deadline",
        "cursor",
        "budget",
    ] {
        if name == "json" {
            if m.get_flag(name) {
                words.push("--json".into());
            }
        } else if let Some(value) = m.try_get_one::<String>(name).ok().flatten() {
            words.extend([format!("--{}", name.replace('_', "-")), value.clone()]);
        }
    }
    if m.get_one::<String>("request").is_some()
        && c.args_for(Surface::Cli).any(|argument| {
            !matches!(argument.name, "cursor" | "budget")
                && m.value_source(argument.name) == Some(ValueSource::CommandLine)
        })
    {
        return Err("--request cannot be mixed with operation arguments".into());
    }
    let arguments = arguments_of(c);
    // Options precede positionals so a Rest field keeps its literal words.
    for positional in [false, true] {
        for projected in arguments
            .iter()
            .filter(|argument| !matches!(argument.spec.name.as_str(), "cursor" | "budget"))
        {
            let name = projected.spec.name.as_str();
            let is_positional = matches!(
                projected.spec.form,
                ArgumentForm::Positional | ArgumentForm::Words | ArgumentForm::Rest
            );
            if is_positional != positional || m.value_source(name) != Some(ValueSource::CommandLine)
            {
                continue;
            }
            if projected.spec.form == ArgumentForm::Flag {
                if m.get_flag(name) {
                    words.push(format!("--{}", projected.spec.name.replace('_', "-")));
                }
                continue;
            }
            let Some(raw) = m.get_raw(name) else {
                continue;
            };
            let raw: Vec<String> = raw
                .map(|value| {
                    value
                        .to_str()
                        .expect("operation arguments are UTF-8")
                        .to_owned()
                })
                .collect();
            if positional {
                if matches!(
                    projected.spec.form,
                    ArgumentForm::Words | ArgumentForm::Rest
                ) {
                    words.push("--".into());
                }
                words.extend(raw);
            } else {
                for value in raw {
                    words.extend([
                        format!("--{}", projected.spec.name.replace('_', "-")),
                        value,
                    ]);
                }
            }
        }
    }
    crate::commands::declarations()
        .parse_cli(&words, polylogue_store::user_writes::new_request_id())
        .map_err(|failure| failure.message)
}

/// The command and values a command line names, for tests.
#[cfg(test)]
pub fn parse(argv: &[&str]) -> Result<(&'static Command, ParsedCommand), String> {
    let matches = command()
        .try_get_matches_from(argv)
        .map_err(|e| e.to_string())?;
    let (c, m) = chosen(&matches).ok_or("no command")?;
    Ok((c, parsed(c, m)?))
}

/// The command a command line names and the request arguments it sends,
/// prepared in `cwd`, for tests.
#[cfg(test)]
pub fn request_arguments(
    argv: &[&str],
    cwd: &std::path::Path,
) -> Result<
    (
        &'static Command,
        ParsedCommand,
        serde_json::Map<String, Value>,
    ),
    String,
> {
    let matches = command()
        .try_get_matches_from(argv)
        .map_err(|e| e.to_string())?;
    let (c, m) = chosen(&matches).ok_or("no command")?;
    let values = parsed(c, m)?;
    let mut arguments = values
        .request
        .arguments
        .as_object()
        .ok_or("operation arguments must be an object")?
        .clone();
    if let Some(prepare) = c.prepare {
        prepare(&values.calling_process, cwd, &mut arguments)?;
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
    let Run::Local(run, _, validate) = command.run else {
        return None;
    };
    let parsed = match parsed(command, matches) {
        Ok(parsed) => parsed,
        Err(error) => {
            return Some(fail(
                output,
                matches.get_flag("json"),
                polylogue_model::Code::BadRequest,
                &error,
                2,
            ));
        }
    };
    if let Err(error) = validate(&parsed.request.arguments) {
        return Some(fail(
            output,
            parsed.json,
            polylogue_model::Code::BadRequest,
            &error,
            2,
        ));
    }
    Some(run(command, &parsed, output, parsed.json))
}

pub fn run(
    matches: &ArgMatches,
    startup: &crate::config::Startup,
    cwd: &std::path::Path,
) -> ExitCode {
    let policy = startup.output;
    let json = matches.get_flag("json");
    let Some((cmd, m)) = chosen(matches) else {
        return fail(
            policy,
            json,
            polylogue_model::Code::BadRequest,
            "no such command",
            2,
        );
    };
    let mut values = match parsed(cmd, m) {
        Ok(parsed) => parsed,
        Err(error) => return fail(policy, json, polylogue_model::Code::BadRequest, &error, 2),
    };
    match cmd.run {
        Run::Local(run, _, validate) => {
            return match validate(&values.request.arguments) {
                Ok(()) => run(cmd, &values, policy, json),
                Err(error) => fail(policy, json, polylogue_model::Code::BadRequest, &error, 2),
            };
        }
        Run::ConfiguredLocal(run, _, validate) => {
            return match validate(&values.request.arguments) {
                Ok(()) => run(cmd, &values, startup, json),
                Err(error) => fail(policy, json, polylogue_model::Code::BadRequest, &error, 2),
            };
        }
        Run::Daemon(_) | Run::Delivered { .. } | Run::Feed { .. } => {}
    }
    if m.get_one::<String>("request").is_none()
        && let Some(prepare) = cmd.prepare
    {
        let Some(arguments) = values.request.arguments.as_object_mut() else {
            return fail(
                policy,
                json,
                polylogue_model::Code::BadRequest,
                "operation arguments must be an object",
                2,
            );
        };
        if let Err(message) = prepare(&values.calling_process, cwd, arguments) {
            return fail(policy, json, polylogue_model::Code::BadRequest, &message, 2);
        }
    }
    let call = &values.request;
    if let Run::Feed {
        registration,
        deliver,
    } = cmd.run
    {
        if let Err(failure) = registration.admit(call) {
            return fail(
                policy,
                json,
                polylogue_model::Code::BadRequest,
                &failure.error.message,
                2,
            );
        }
        return deliver(&values, startup, call);
    }
    if let Run::Delivered { deliver, .. } = cmd.run {
        return deliver(&values, startup, call);
    }
    let root = startup.config.archive.clone();
    let answer = match crate::client::call_ready(&Archive::at(&root), call) {
        Ok(r) => r,
        Err(e) if is_down(&e) => {
            return fail(
                policy,
                json,
                polylogue_model::Code::DaemonRequired,
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
            return fail(
                policy,
                json,
                polylogue_model::Code::DeadlineExceeded,
                &e.to_string(),
                7,
            );
        }
        Err(e) => return fail(policy, json, polylogue_model::Code::Io, &e.to_string(), 1),
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
        crate::row_numbers::remember(
            &startup.config.paths,
            &root,
            &crate::row_numbers::row_ids(&data["rows"]),
        );
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
    values: &ParsedCommand,
    answer: &Answer<Value>,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    let data = answer
        .value()
        .expect("output handlers receive answers with values");
    if !values.request.arguments.bool("line").unwrap_or(false) {
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
                return fail(
                    startup.output,
                    json,
                    polylogue_model::Code::Internal,
                    &error.to_string(),
                    1,
                );
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

/// `continue --exec`: replaces this process with the harness's own
/// resume command; without `--exec`, the plan is printed.
pub fn resume_output(
    values: &ParsedCommand,
    answer: &Answer<Value>,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    let data = answer
        .value()
        .expect("output handlers receive answers with values");
    if !values.calling_process.bool("exec").unwrap_or(false) {
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
    let Some(resume) = data.get("harness_command").filter(|n| !n.is_null()) else {
        eprint!(
            "{}",
            print::diagnostic(
                "",
                &format!(
                    "this session has no harness resume; continue with context instead: {}",
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
    let program = resume["program"].as_str().unwrap_or("");
    let args: Vec<&str> = resume["args"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|a| a.as_str())
        .collect();
    // The harness reopens the session from its resume directory; the CLI's
    // own directory would reopen nothing or the wrong project.
    let dir = match resume["cwd"].as_str() {
        Some(dir) if std::path::Path::new(dir).is_dir() => Ok(dir),
        Some(dir) => Err(format!("directory_missing: {dir} does not exist here")),
        None => Err("directory_unknown: the session recorded no directory to resume in".into()),
    };
    let dir = match dir {
        Ok(dir) => dir,
        Err(reason) => {
            eprint!(
                "{}",
                print::diagnostic(
                    "",
                    &format!(
                        "{reason}; continue with context instead: polylogue takeover {}",
                        data["session"].as_str().unwrap_or("")
                    ),
                    None,
                    output.stderr_color()
                )
            );
            return ExitCode::from(1);
        }
    };
    let mut cmd = std::process::Command::new(program);
    cmd.args(&args).current_dir(dir);
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
    code: polylogue_model::Code,
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
