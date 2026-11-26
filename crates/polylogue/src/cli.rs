//! The command line, built from the command table. A bare query
//! (`polylogue repo:x since:7d`) runs the query language; subcommands cover
//! everything else. Every command takes `--json` for programs and agents.

pub(crate) mod attachment;
pub(crate) mod calling_process;
pub mod changes;
pub(crate) mod completion;
pub(crate) mod follow;
pub(crate) mod line;
pub mod local;
pub mod picker;
pub mod print;
pub(crate) mod run_line;
pub(crate) mod schema;
pub(crate) mod staged_file;
pub(crate) mod status_view;
pub(crate) mod write_out;

use crate::commands::{COMMANDS, Cli, Command, GROUPS, Run};
use clap::ArgAction;
use clap::parser::{ArgMatches, ValueSource};
use command_surface::{Answer, ArgumentForm, CliArgument, ParsedCommand, TextStyle};
use polylogue_store::Archive;
use serde::Deserialize as _;
use serde_json::Value;
use std::process::ExitCode;

fn version() -> String {
    let revision = option_env!("POLYLOGUE_REVISION").unwrap_or("unavailable");
    format!("{} (revision {revision})", env!("CARGO_PKG_VERSION"))
}

/// The global options that take their value as the next word. A reader of a
/// raw command line (the hook and daemon peer checks) skips each with its value.
pub(crate) const GLOBAL_VALUED_OPTIONS: [&str; 9] = [
    "--deadline",
    "--archive",
    "--config",
    "--listen",
    "--auth-token-file",
    "--request",
    "--request-id",
    "--cursor",
    "--budget",
];

/// The clap command tree: one subcommand per table command, grouped by
/// their first word, and the bare query as the root's own arguments.
pub fn command() -> clap::Command {
    let mut root = clap::Command::new("polylogue")
        .disable_help_subcommand(true)
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
        )
        .arg(
            clap::Arg::new("cursor")
                .long("cursor")
                .global(true)
                .value_name("CURSOR")
                .help("Continue from a previous page's `next`"),
        )
        .arg(
            clap::Arg::new("budget")
                .long("budget")
                .global(true)
                .value_name("BUDGET")
                .help("Tokens for a prepared context; defaults to its named pack"),
        );
    for c in COMMANDS.iter() {
        match c.cli {
            Cli::Line => continue,
            Cli::Subcommand | Cli::Hidden => {
                let words: Vec<&str> = c.name.split(' ').collect();
                root = place(root, "", &words[..words.len() - 1], subcommand(c));
            }
        }
    }
    root = root.subcommand_precedence_over_arg(true).arg(
        clap::Arg::new("selection")
            .value_name("SELECTION")
            .help("One quoted query or reference")
            .add(clap_complete::engine::ArgValueCompleter::new(
                completion::Query,
            )),
    );
    crate::config::augment_command(root)
}

/// Places a row's subcommand under its group words (`dev schema infer`
/// under `dev`, then `schema`), creating each group on first use; a group
/// whose rows are all hidden is hidden.
fn place(parent: clap::Command, above: &str, groups: &[&str], sub: clap::Command) -> clap::Command {
    let Some((group, below)) = groups.split_first() else {
        return parent.subcommand(sub);
    };
    let path = if above.is_empty() {
        (*group).to_owned()
    } else {
        format!("{above} {group}")
    };
    let mut parent = parent;
    if parent.find_subcommand(group).is_none() {
        let help = GROUPS
            .iter()
            .find(|g| g.name == path)
            .map_or("", |g| g.help);
        let hidden = COMMANDS
            .iter()
            .filter(|x| x.name.starts_with(&format!("{path} ")))
            .all(|x| x.cli.hidden());
        parent = parent.subcommand(
            clap::Command::new(group.to_string())
                .about(help)
                .hide(hidden)
                .subcommand_required(true)
                .arg_required_else_help(true),
        );
    }
    parent.mut_subcommand(group, |g| place(g, &path, below, sub))
}

fn subcommand(c: &Command) -> clap::Command {
    let word = c.name.rsplit(' ').next().unwrap_or(c.name);
    let mut sub = clap::Command::new(word)
        .about(c.help)
        .hide(c.cli.hidden())
        .args(row_args(c));
    if !c.examples.is_empty() {
        sub = sub.after_help(format!("Examples:\n  {}", c.examples.join("\n  ")));
    }
    sub
}

/// A row's clap arguments: help from the field's doc comment, the default
/// shown from its declared serde default.
fn row_args(c: &Command) -> Vec<clap::Arg> {
    shown_arguments(c)
        .into_iter()
        .map(|(argument, property)| {
            let mut arg = clap_arg(&argument);
            match property.and_then(|property| property.get("default")) {
                Some(Value::String(text)) if !text.is_empty() => {
                    arg = arg.default_value(text.clone());
                }
                Some(Value::Number(number)) => arg = arg.default_value(number.to_string()),
                _ => {}
            }
            if c.name == "set" && matches!(argument.spec.name.as_str(), "key" | "value") {
                arg = arg.add(clap_complete::engine::ArgValueCompleter::new(
                    completion::Setting {
                        key: argument.spec.name == "key",
                    },
                ));
            }
            arg
        })
        .collect()
}

/// A row's arguments on the command line: its declared request fields and
/// calling-process options, with the product's choice lists the input
/// schema cannot carry.
pub(crate) fn arguments_of(command: &Command) -> Vec<CliArgument> {
    shown_arguments(command)
        .into_iter()
        .map(|(argument, _)| argument)
        .collect()
}

/// Each shown argument with its input-schema property; calling-process
/// options have none.
fn shown_arguments(command: &Command) -> Vec<(CliArgument, Option<&'static Value>)> {
    let declarations = crate::commands::declarations();
    let properties = declarations
        .descriptions()
        .iter()
        .find(|description| description.spec.name == command.name)
        .and_then(|description| description.input_schema.get("properties"));
    let globals = declarations.cli_globals();
    declarations
        .cli_arguments(command.name)
        .expect("declared CLI arguments")
        .into_iter()
        .filter(|argument| {
            !globals
                .iter()
                .any(|global| global.name == argument.spec.name)
        })
        .map(|mut argument| {
            let property = properties
                .filter(|_| !argument.spec.calling_process)
                .and_then(|properties| properties.get(&argument.spec.name));
            if argument.choices.is_empty() {
                let name = argument.spec.name.as_str();
                if command.name == "start" && name == "harness" {
                    argument.choices = polylogue_model::live::channel::HARNESSES
                        .iter()
                        .filter(|harness| harness.host.is_some())
                        .map(|harness| harness.name.to_owned())
                        .collect();
                }
            }
            (argument, property)
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
    let (word, sub) = matches.subcommand()?;
    let mut name = word.to_owned();
    let mut leaf = sub;
    while let Some((next, below)) = leaf.subcommand() {
        name = format!("{name} {next}");
        leaf = below;
    }
    crate::commands::command(&name).map(|c| (c, leaf))
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
    let arguments = arguments_of(c);
    if m.get_one::<String>("request").is_some()
        && arguments
            .iter()
            .any(|argument| m.value_source(&argument.spec.name) == Some(ValueSource::CommandLine))
    {
        return Err("--request cannot be mixed with operation arguments".into());
    }
    // Options precede positionals so a Rest field keeps its literal words.
    for positional in [false, true] {
        for projected in &arguments {
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
/// for tests.
#[cfg(test)]
pub fn request_arguments(
    argv: &[&str],
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
    let arguments = values
        .request
        .arguments
        .as_object()
        .ok_or("operation arguments must be an object")?
        .clone();
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
            ));
        }
    };
    if let Err(error) = validate(&parsed.request.arguments) {
        return Some(fail(
            output,
            parsed.json,
            polylogue_model::Code::BadRequest,
            &error,
        ));
    }
    Some(run(command, &parsed, output, parsed.json))
}

pub fn run(matches: &ArgMatches, startup: &crate::config::Startup) -> ExitCode {
    let policy = startup.output;
    let json = matches.get_flag("json");
    let Some((cmd, m)) = chosen(matches) else {
        return fail(
            policy,
            json,
            polylogue_model::Code::BadRequest,
            "no such command",
        );
    };
    let values = match parsed(cmd, m) {
        Ok(parsed) => parsed,
        Err(error) => return fail(policy, json, polylogue_model::Code::BadRequest, &error),
    };
    match cmd.run {
        Run::Local(run, _, validate) => {
            return match validate(&values.request.arguments) {
                Ok(()) => run(cmd, &values, policy, json),
                Err(error) => fail(policy, json, polylogue_model::Code::BadRequest, &error),
            };
        }
        Run::ConfiguredLocal(run, _, validate) => {
            return match validate(&values.request.arguments) {
                Ok(()) => run(cmd, &values, startup, json),
                Err(error) => fail(policy, json, polylogue_model::Code::BadRequest, &error),
            };
        }
        Run::Daemon(_) => {}
    }
    let call = &values.request;
    // The request is checked here as the daemon checks it, so a refusal
    // needs no daemon.
    if let Err(failure) = crate::daemon::handler::policy(
        cmd,
        &call.arguments,
        polylogue_model::authorship::Channel::Cli,
    )
    .and_then(|row_policy| crate::daemon::handler::refused_options(cmd, &row_policy, call))
    {
        return fail_with(policy, json, &failure.error, &[]);
    }
    if let Some(deliver) = cmd.deliver {
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
            );
        }
        Err(e) => return fail(policy, json, polylogue_model::Code::Io, &e.to_string()),
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
    match cmd.output {
        Some(output) => output(&values, &answer, startup, json),
        None => {
            match crate::commands::declarations().render_text(cmd.name, &answer, style(policy)) {
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

/// How text for standard output is styled under `output`.
fn style(output: crate::config::OutputConfig) -> TextStyle {
    if output.stdout_color() {
        TextStyle::Ansi
    } else {
        TextStyle::Plain
    }
}

/// `status`: the view, the whole answer with `--json`, or with `--line` the
/// one line a status bar shows (`3 running · 1 waiting 7m`). `--watch`
/// asks the daemon again every two seconds and redraws the view.
pub fn status_output(
    values: &ParsedCommand,
    answer: &Answer<Value>,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    let arguments = match crate::protocol::status::Arguments::deserialize(&values.request.arguments)
    {
        Ok(arguments) => arguments,
        Err(error) => {
            return fail(
                startup.output,
                json,
                polylogue_model::Code::Internal,
                &error.to_string(),
            );
        }
    };
    if arguments.watch {
        return watch_status(values, &arguments, answer, startup, json);
    }
    let mut status = match status_answer(answer) {
        Ok(status) => status,
        Err(error) => {
            return fail(
                startup.output,
                json,
                polylogue_model::Code::Internal,
                &error.to_string(),
            );
        }
    };
    if !arguments.line {
        if json {
            println!(
                "{}",
                polylogue_render::terminal::json(
                    &serde_json::to_value(answer).expect("answer serializes"),
                    false
                )
            );
        } else {
            let verification = status.verification.take();
            print!("{}", status_view(&status, answer, startup, &arguments));
            if let Some(found) = verification {
                print!("{}", print::verification(&found, style(startup.output)));
            }
        }
        return ExitCode::from(answer.exit_code());
    }
    let line = status.running.line(startup.clock.now().as_millisecond());
    if json {
        println!("{}", serde_json::json!({ "line": line }));
    } else {
        println!("{line}");
    }
    ExitCode::from(answer.exit_code())
}

fn status_answer(answer: &Answer<Value>) -> serde_json::Result<crate::protocol::status::Answer> {
    let data = answer
        .value()
        .expect("output handlers receive answers with values");
    crate::protocol::status::Answer::deserialize(data)
}

/// The status view of `status`, with the answer's degraded reasons and
/// notes, at the terminal's width.
fn status_view(
    status: &crate::protocol::status::Answer,
    answer: &Answer<Value>,
    startup: &crate::config::Startup,
    arguments: &crate::protocol::status::Arguments,
) -> String {
    let mut problems = match &answer.outcome {
        command_surface::Outcome::Degraded { missing, .. } => missing.clone(),
        _ => Vec::new(),
    };
    problems.extend(answer.envelope.notes.iter().cloned());
    let zone = startup
        .config
        .standalone_day()
        .map(|day| day.zone().clone())
        .unwrap_or(jiff::tz::TimeZone::UTC);
    let width = if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        // A terminal that reports no size (a pseudo-terminal nobody sized)
        // gets the same width as no terminal.
        crossterm::terminal::size()
            .ok()
            .filter(|(columns, _)| *columns > 0)
            .map_or(100, |(columns, _)| usize::from(columns))
    } else {
        100
    };
    status_view::render(
        status,
        &problems,
        &status_view::View {
            style: style(startup.output),
            width,
            now_ms: startup.clock.now().as_millisecond(),
            zone: &zone,
            whole_log: arguments.log,
        },
    )
}

/// Redraws the status view every two seconds until interrupted. Each frame
/// is a new request; the daemon keeps nothing between them.
fn watch_status(
    values: &ParsedCommand,
    arguments: &crate::protocol::status::Arguments,
    first: &Answer<Value>,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    if json || style(startup.output) != TextStyle::Ansi {
        return fail(
            startup.output,
            json,
            polylogue_model::Code::BadRequest,
            "--watch redraws the view on a color terminal; without one, run `polylogue status` as often as needed",
        );
    }
    let archive = Archive::at(&startup.config.archive);
    let mut answer = first.clone();
    loop {
        let frame = match &answer.outcome {
            command_surface::Outcome::Error(error) => {
                print::diagnostic("", &error.message, None, startup.output.stdout_color())
            }
            _ => match status_answer(&answer) {
                Ok(status) => status_view(&status, &answer, startup, arguments),
                Err(error) => error.to_string(),
            },
        };
        print!("\x1b[H\x1b[2J{frame}");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        std::thread::sleep(std::time::Duration::from_secs(2));
        answer = match crate::client::call_ready(&archive, &values.request) {
            Ok(answer) => answer,
            Err(error) => {
                return fail(
                    startup.output,
                    json,
                    polylogue_model::Code::DaemonRequired,
                    &error.to_string(),
                );
            }
        };
    }
}

fn run_plan(
    plan: &crate::resume::Plan,
    output: crate::config::OutputConfig,
    replace: bool,
) -> ExitCode {
    use std::os::unix::process::CommandExt;
    let Some(resume) = &plan.harness_command else {
        eprint!(
            "{}",
            print::diagnostic(
                "",
                &format!(
                    "this session has no harness resume; continue with context instead: {}",
                    print::command_line(&plan.context_assisted)
                ),
                None,
                output.stderr_color()
            )
        );
        return ExitCode::from(1);
    };
    // The harness reopens the session from its resume directory; the CLI's
    // own directory would reopen nothing or the wrong project.
    let dir = match resume.cwd.as_deref() {
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
                        "{reason}; continue with context instead: {}",
                        print::command_line(&plan.context_assisted)
                    ),
                    None,
                    output.stderr_color()
                )
            );
            return ExitCode::from(1);
        }
    };
    let mut cmd = std::process::Command::new(&resume.program);
    cmd.args(&resume.args).current_dir(dir);
    if !replace {
        use std::os::unix::process::ExitStatusExt as _;
        return match cmd.status() {
            // A harness killed by a signal exits as a shell reports it: 128 + signal.
            Ok(status) => ExitCode::from(
                status
                    .code()
                    .or_else(|| status.signal().map(|signal| 128 + signal))
                    .and_then(|code| u8::try_from(code).ok())
                    .unwrap_or(1),
            ),
            Err(error) => fail(output, false, polylogue_model::Code::Io, &error.to_string()),
        };
    }
    let err = cmd.exec();
    eprint!(
        "{}",
        print::diagnostic(
            "",
            &format!("could not run {}: {err}", resume.program),
            None,
            output.stderr_color()
        )
    );
    ExitCode::FAILURE
}

pub(crate) fn is_down(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
    )
}

/// Runs `call` (a connection), waiting while the archive's daemon is still
/// opening it, until `deadline_ms`.
pub(crate) fn when_ready<T>(
    archive: &Archive,
    deadline_ms: Option<i64>,
    mut call: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    let mut told = false;
    let until = deadline_ms.map(|deadline_ms| {
        std::time::Instant::now() + crate::daemon::execution::remaining(Some(deadline_ms))
    });
    loop {
        match call() {
            Err(e) if is_down(&e) && archive.lock_held() => {
                if !told {
                    eprintln!("polylogue: waiting for the daemon to finish opening the archive");
                    told = true;
                }
                crate::client::wait_for_socket(&archive.socket(), until)?;
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
) -> ExitCode {
    fail_with(output, json, &crate::protocol::failure(code, message), &[])
}

/// One printer for every failure: the whole answer for programs; for a
/// person, the code, the message, the corrected query and the notes.
pub(crate) fn fail_with(
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
