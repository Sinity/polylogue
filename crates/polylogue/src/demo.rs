//! A guided tour through ordinary commands over a separate Morrow archive.

mod files;

use crate::commands::{self, Caller, Command, Surface, Values};
use crate::daemon::service;
use polylogue_engine::Config;
use serde::Serialize;
use serde_json::{Value, json};
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

const PARSER: &str = "claude-code:7f3a2c10-0002-4000-8000-000000000002";
const RESUME: &str = "claude-code:7f3a2c10-0004-4000-8000-000000000004";

#[derive(Serialize)]
struct Step {
    title: String,
    description: String,
    command: String,
    command_line: String,
    arguments: Value,
    result: Value,
    seconds: f64,
    text: String,
}

#[derive(Serialize)]
struct Tour {
    preparation_seconds: f64,
    first_query_seconds: f64,
    steps: Vec<Step>,
    transcript: String,
}

pub fn tour(
    _: &Command,
    values: &Values,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    let directory = values.text("directory").unwrap_or_default();
    match run(Path::new(&directory), startup) {
        Ok(tour) => {
            if json {
                println!(
                    "{}",
                    polylogue_render::terminal::json(
                        &serde_json::to_value(&tour).expect("serialize tour"),
                        true,
                    )
                );
            } else {
                print!("{}", polylogue_render::terminal::escape(&tour.transcript));
            }
            ExitCode::SUCCESS
        }
        Err(error) => crate::cli::fail(startup.output, json, "demo", &error, 1),
    }
}

fn run(directory: &Path, startup: &crate::config::Startup) -> Result<Tour, String> {
    let preparing = Instant::now();
    // Creation is exclusive: a tour never changes an existing directory.
    fs::create_dir(directory).map_err(|e| format!("create a new tour directory: {e}"))?;
    // Keep a declared short alias usable by the daemon's Unix socket.
    // Archive identity is resolved separately by its owning configuration.
    let directory = std::path::absolute(directory).map_err(|e| e.to_string())?;
    let home = directory.join("home");
    for (name, bytes) in files::FILES {
        let path = home.join(name);
        fs::create_dir_all(path.parent().ok_or("example file has no parent")?)
            .map_err(|e| e.to_string())?;
        fs::write(path, bytes).map_err(|e| e.to_string())?;
    }
    let config = Config {
        clock: startup.clock.clone(),
        home,
        tool_dirs: Default::default(),
        workers: 2,
        reader_memory_bytes: 64 << 20,
        bytes_in_flight: 64 << 20,
        machine: startup.engine.machine.clone(),
        host: startup.engine.host.clone(),
        chunk: 64 << 20,
        live_every: Duration::from_secs(2),
        reads_sources: true,
    };
    let day = query_language::time::DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::constant(0, 0, 0, 0),
    )
    .map_err(|e| e.to_string())?;
    let deployment = startup
        .config
        .standalone_for_archive(directory.join("archive"), day)
        .map_err(|e| e.to_string())?;
    let mut runtime = service::start(
        deployment,
        config,
        service::ArchiveMode::Standalone,
        polylogue_engine::Lifetime::UntilStopped,
        startup.stopped_attempt,
    )
    .map_err(|e| e.to_string())?;
    // Tour reads need the bundled SDK files as well as the explicit exports.
    // An import answers for its own inputs, independently of background work.
    loop {
        let status = runtime.daemon.status.lock().map_err(|e| e.to_string())?;
        if status.passes > 0 && !status.busy {
            if !status.last_pass.errors.is_empty() {
                return Err(status.last_pass.errors.join("; "));
            }
            break;
        }
        drop(status);
        if let Some(failure) = runtime.worker_failure() {
            return Err(failure.into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let invoke = |command: &str, arguments: Value| -> Result<Value, String> {
        let declaration =
            commands::command(command).ok_or_else(|| format!("no command {command}"))?;
        let authorship = polylogue_model::authorship::compute_authorship(
            polylogue_model::authorship::Channel::Cli,
            &Default::default(),
            None,
        );
        let invocation = runtime.daemon.invocation(&authorship);
        let limit = crate::daemon::execution::limit(None, Default::default());
        let answer = crate::daemon::handler::local(
            declaration.name,
            arguments,
            &crate::daemon::handler::Local {
                daemon: &runtime.daemon,
                invocation: &invocation,
                caller: &Caller {
                    surface: Surface::Cli,
                    cwd: Some(directory.clone()),
                },
                limit: &limit,
                request_id: &polylogue_store::user_writes::new_request_id(),
                cursor: None,
            },
        );
        crate::protocol::into_value(answer).map_err(|failure| {
            format!(
                "{command}: {}: {}",
                failure.code.as_deref().unwrap_or_default(),
                failure.message
            )
        })
    };
    invoke(
        "import",
        json!({"paths": ["home/exports/chatgpt/conversations.json", "home/exports/claude-ai/conversations.json"]}),
    )?;
    let preparation_seconds = preparing.elapsed().as_secs_f64();
    let mut steps = Vec::new();
    let mut step =
        |title: &str, description: &str, command: &str, arguments: Value| -> Result<(), String> {
            let started = Instant::now();
            let result = invoke(command, arguments.clone())?;
            let seconds = started.elapsed().as_secs_f64();
            let declaration = commands::command(command).ok_or("missing command")?;
            let text = if command == "query" {
                crate::cli::print::query_plain(&result)
            } else {
                commands::declarations()
                    .render_text(
                        declaration.name,
                        &command_surface::Answer::ok(result.clone()),
                        command_surface::TextStyle::Plain,
                    )
                    .map_err(|failure| failure.message)?
            };
            steps.push(Step {
                title: title.into(),
                description: description.into(),
                command: command.into(),
                command_line: command_line(declaration, &arguments),
                arguments,
                result,
                seconds,
                text,
            });
            Ok(())
        };
    step(
        "Find the date decision",
        "Search human messages for chrono. Morrow has two matches: the parser decision and a README request.",
        "query",
        json!({"query": "messages where written_by:human chrono"}),
    )?;
    step(
        "Read the parser session",
        "Read the session around the decision, including the assistant's account of its work.",
        "read",
        json!({"target": format!("session:{PARSER}"), "format": "markdown"}),
    )?;
    step(
        "Check recorded tool failures",
        "Structured outcomes record two failures: cargo test exited 101 and cargo clippy exited 1. An assistant's claim is prose; the outcome comes from the tool result.",
        "query",
        json!({"query": "tool_calls where outcome:error"}),
    )?;
    step(
        "Follow parent and child work",
        "The parser session has a subagent and a later continuation. Include every session kind when following its children.",
        "query",
        json!({"query": format!("sessions where kind:* parent:{PARSER}")}),
    )?;
    step(
        "Read the continuation with its parents",
        "Inherited messages appear with the continuation when requested.",
        "read",
        json!({"target": format!("session:{RESUME}"), "show": ["inherited"], "format": "markdown"}),
    )?;
    step(
        "Keep unknown costs visible",
        "Four top-level sessions have unknown total cost, including both Codex sessions. Recorded token observations alone do not establish complete costs.",
        "query",
        json!({"query": "sessions where cost:unknown"}),
    )?;
    step(
        "Inspect usage",
        "Usage and cost come from the current catalog. The historical example price table is a separate illustration; unknown figures stay unknown.",
        "measure usage",
        json!({"query": "sessions where kind:*"}),
    )?;
    step(
        "Add a decision note",
        "Write a note only in this tour's archive.",
        "assert",
        json!({"assertions": [{"kind":"decision","value":"Use chrono for date parsing.","links":[{"role":"about","reference":format!("session:{PARSER}")}]}]}),
    )?;
    step(
        "Read the note",
        "Notes can be retrieved independently of the conversation.",
        "query",
        json!({"query":format!("assertions where about:\"session:{PARSER}\"")}),
    )?;
    step(
        "Prepare takeover context",
        "Takeover carries the session's prompts and last answer for the next reader. It names the assertion without quoting it, because another author wrote it and the operator has not accepted it; the previous step still reads it.",
        "takeover",
        json!({"session": PARSER}),
    )?;
    let first_query_seconds = steps.first().ok_or("tour has no steps")?.seconds;
    let mut transcript = format!(
        "# Morrow tour\n\nSynthetic work on a date parser. Preparation: {preparation_seconds:.3}s. First query: {first_query_seconds:.3}s.\n\n"
    );
    for step in &steps {
        use std::fmt::Write;
        writeln!(
            transcript,
            "## {}\n\n{}\n\n{}\n\n{}\n",
            step.title, step.description, step.command_line, step.text
        )
        .map_err(|e| e.to_string())?;
    }
    runtime.shutdown().map_err(|e| e.to_string())?;
    fs::write(directory.join("transcript.md"), &transcript).map_err(|e| e.to_string())?;
    Ok(Tour {
        preparation_seconds,
        first_query_seconds,
        steps,
        transcript,
    })
}

fn command_line(command: &Command, arguments: &Value) -> String {
    let quote = polylogue_render::shell_quote;
    let mut words = vec!["polylogue".to_owned()];
    if command.name != "query" {
        words.push(command.name.to_owned());
    }
    for arg in command.args {
        let value = &arguments[arg.name];
        if value.is_null() || value == false {
            continue;
        }
        if !arg.positional {
            words.push(format!("--{}", arg.name.replace('_', "-")));
        }
        if value == true {
            continue;
        }
        if let Some(values) = value.as_array() {
            words.extend(values.iter().map(|v| quote(v.as_str().unwrap_or(""))));
        } else {
            words.push(quote(value.as_str().unwrap_or("")));
        }
    }
    words.join(" ")
}
