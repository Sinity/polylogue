//! A guided tour through ordinary commands over a separate Morrow archive.

mod files;

use crate::commands::{self, Action, Caller, Command, Surface, Values};
use crate::daemon::{Response, service};
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

pub fn tour(_: &Command, values: &Values, json: bool) -> ExitCode {
    let directory = values.text("directory").unwrap_or_default();
    match run(Path::new(&directory)) {
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
        Err(error) => crate::cli::fail(json, "demo", &error, 1),
    }
}

fn run(directory: &Path) -> Result<Tour, String> {
    let preparing = Instant::now();
    // Creation is exclusive: a tour never changes an existing directory.
    fs::create_dir(directory).map_err(|e| format!("create a new tour directory: {e}"))?;
    let directory = directory.canonicalize().map_err(|e| e.to_string())?;
    let home = directory.join("home");
    for (name, bytes) in files::FILES {
        let path = home.join(name);
        fs::create_dir_all(path.parent().ok_or("example file has no parent")?)
            .map_err(|e| e.to_string())?;
        fs::write(path, bytes).map_err(|e| e.to_string())?;
    }
    let environment = Config::from_env();
    let config = Config {
        home,
        tool_dirs: Default::default(),
        workers: 2,
        reader_memory_bytes: 64 << 20,
        bytes_in_flight: 64 << 20,
        machine: environment.machine,
        host: environment.host,
        chunk: 64 << 20,
        rescan_every: Duration::from_secs(60),
        live_every: Duration::from_secs(2),
    };
    let mut runtime = service::start(
        &directory.join("archive"),
        config,
        service::ArchiveMode::Standalone,
    )
    .map_err(|e| e.to_string())?;
    let invoke = |command: &str, arguments: Value| -> Result<Value, String> {
        let declaration =
            commands::command(command).ok_or_else(|| format!("no command {command}"))?;
        let values = Values::from_json(
            declaration,
            Surface::Cli,
            arguments
                .as_object()
                .ok_or("arguments must be an object")?
                .clone(),
        )?;
        let Action::Request(build) = declaration.action else {
            return Err(format!("{command} must use the daemon"));
        };
        let request = build(
            declaration,
            &values,
            &Caller {
                surface: Surface::Cli,
                cwd: Some(directory.clone()),
                json: true,
            },
        )?;
        match service::handle(request, &runtime.daemon) {
            Response::Ok { data } => Ok(data),
            Response::Error { kind, message, .. } => Err(format!("{command}: {kind}: {message}")),
        }
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
                (declaration.print)(&result)
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
        json!({"session": PARSER, "format": "markdown"}),
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
        json!({"session": RESUME, "with_parents": true, "format": "markdown"}),
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
        "note add",
        json!({"targets": [format!("session:{PARSER}")], "kind": "decision", "body": "Use chrono for date parsing."}),
    )?;
    step(
        "Read the note",
        "Notes can be retrieved independently of the conversation.",
        "note list",
        json!({"targets": [format!("session:{PARSER}")]}),
    )?;
    step(
        "Prepare takeover context",
        "Takeover combines the session's work and notes for the next reader.",
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
