//! A guided tour through everyday commands over a separate Morrow archive.

use polylogue_model::JsonExt as _;
mod files;

use crate::commands::{self, Command};
use crate::daemon::service;
use polylogue_engine::Config;
use serde::{Deserialize as _, Serialize};
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
    values: &command_surface::ParsedCommand,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    let directory = values
        .request
        .arguments
        .str("directory")
        .map(str::to_owned)
        .unwrap_or_default();
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
        Err(error) => crate::cli::fail(
            startup.client.output,
            json,
            polylogue_model::Code::Demo,
            &error,
        ),
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
        clock: startup.client.clock.clone(),
        home,
        tool_dirs: Default::default(),
        workers: 2,
        memory_budget: Default::default(),
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
    )
    .map_err(|e| e.to_string())?;
    // Tour reads need the bundled SDK files as well as the explicit exports.
    // An import answers for its own inputs, independently of background work.
    // A loop that ended changes its status no more: look at the workers
    // each second the status has not changed.
    loop {
        let idle = runtime.daemon.status.wait_for(
            std::time::Instant::now() + Duration::from_secs(1),
            |status| status.passes > 0 && !status.busy,
        );
        if let Some(status) = idle {
            if !status.last_pass.errors.is_empty() {
                return Err(polylogue_model::Failure::joined(&status.last_pass.errors)
                    .map(|failure| failure.message)
                    .unwrap_or_default());
            }
            break;
        }
        if let Some(failure) = runtime.worker_failure() {
            return Err(failure.into());
        }
    }
    let invoke = |command: &str, arguments: Value| -> Result<Value, String> {
        let declaration =
            commands::command(command).ok_or_else(|| format!("no command {command}"))?;
        let mut caller = crate::operation::Caller::local(
            command_surface::Via::Cli,
            runtime.daemon.archive.stop().child(),
        );
        caller.cwd = Some(directory.clone());
        let request = crate::client::request(
            declaration.name,
            arguments,
            polylogue_store::user_writes::new_request_id(),
        );
        let (_, answer) =
            crate::daemon::handler::answer_collected(&runtime.daemon, &request, &caller);
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
            let text = if command == "rows" {
                let answer = crate::protocol::line::LineAnswer::deserialize(&result)
                    .map_err(|error| format!("{command}: {error}"))?;
                line_text(&answer)?
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
        "rows",
        json!({"selection": "messages where written_by:human chrono"}),
    )?;
    step(
        "Read the parser session",
        "Read the session around the decision, including the assistant's account of its work.",
        "rows",
        json!({"selection": format!("session:{PARSER}"), "steps": [{"verb": "read", "format": "markdown"}]}),
    )?;
    step(
        "Check recorded tool failures",
        "Structured outcomes record two failures: cargo test exited 101 and cargo clippy exited 1. An assistant's claim is prose; the outcome comes from the tool result.",
        "rows",
        json!({"selection": "tool_calls where outcome:error"}),
    )?;
    step(
        "Follow parent and child work",
        "The parser session has a subagent and a later continuation. Include every session kind when following its children.",
        "rows",
        json!({"selection": format!("sessions where kind:* parent:{PARSER}")}),
    )?;
    step(
        "Read the continuation with its parents",
        "Inherited messages appear with the continuation when requested.",
        "rows",
        json!({"selection": format!("session:{RESUME}"), "steps": [{"verb": "read", "show": ["inherited"], "format": "markdown"}]}),
    )?;
    step(
        "Keep unknown costs visible",
        "Four top-level sessions have unknown total cost, including both Codex sessions. Recorded token observations alone do not establish complete costs.",
        "rows",
        json!({"selection": "sessions where cost:unknown"}),
    )?;
    step(
        "Inspect usage",
        "Usage and cost come from the current catalog. The historical example price table is a separate illustration; unknown figures stay unknown.",
        "rows",
        json!({"selection": "sessions where kind:* | measure usage"}),
    )?;
    step(
        "Add a decision note",
        "Write a note only in this tour's archive.",
        "rows",
        json!({"selection": format!("session:{PARSER}"), "steps": [{"verb": "note", "as": "decision", "text": "Use chrono for date parsing."}]}),
    )?;
    step(
        "Read the note",
        "Notes can be retrieved independently of the conversation.",
        "rows",
        json!({"selection":format!("annotations where about:\"session:{PARSER}\"")}),
    )?;
    step(
        "Prepare takeover context",
        "Takeover carries the session's prompts and last answer for the next reader. It names the annotation without quoting it, because another author wrote it and the operator has not accepted it; the previous step still reads it.",
        "rows",
        json!({"selection": format!("session:{PARSER}"), "steps": [{"verb": "brief"}]}),
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

fn line_text(answer: &crate::protocol::line::LineAnswer) -> Result<String, String> {
    use crate::protocol::line::{LineAnswer, StepAnswer, StepOutcome};
    let reports = match answer {
        LineAnswer::Listing(listing) => {
            return Ok(crate::cli::print::query(
                listing,
                command_surface::TextStyle::Plain,
            ));
        }
        LineAnswer::Steps(reports) => &reports.steps,
        LineAnswer::Explained(_) => return Err("the tour expected completed steps".into()),
    };
    let mut text = String::new();
    let mut undo = Vec::new();
    for report in reports {
        let StepOutcome::Done { reply, undo: lines } = &report.outcome else {
            return Err(match &report.outcome {
                StepOutcome::Failed { error } => {
                    format!("{} failed: {}", report.verb, error.message)
                }
                _ => format!("{} did not run", report.verb),
            });
        };
        match reply {
            StepAnswer::Read(page) => {
                let mut bytes = Vec::new();
                polylogue_render::encode_document(
                    &page.document,
                    polylogue_render::Format::Markdown,
                    polylogue_render::EncodeOptions {
                        commands: Some(commands::declarations()),
                        via: command_surface::Via::Cli,
                        model_facing: false,
                        ..Default::default()
                    },
                    &mut bytes,
                )
                .map_err(|error| error.to_string())?;
                text.push_str(&String::from_utf8(bytes).map_err(|error| error.to_string())?);
            }
            StepAnswer::Brief(prepared) => text.push_str(&crate::cli::print::context(
                prepared,
                command_surface::TextStyle::Plain,
            )),
            StepAnswer::Resume(found) => text.push_str(&crate::cli::print::resume(
                found,
                command_surface::TextStyle::Plain,
            )),
            other => text.push_str(&crate::cli::print::json(
                other,
                command_surface::TextStyle::Plain,
            )),
        }
        undo.extend(lines);
    }
    for line in undo {
        use std::fmt::Write;
        writeln!(text, "undo: {line}").map_err(|error| error.to_string())?;
    }
    Ok(text)
}

fn command_line(command: &Command, arguments: &Value) -> String {
    let mut words = vec!["polylogue".to_owned()];
    let declarations = commands::declarations();
    if command.name == "rows" {
        if let Some(selection) = arguments["selection"].as_str() {
            words.push(polylogue_render::shell_quote(selection));
        }
        if let Some(steps) = arguments["steps"].as_array() {
            let declared = declarations
                .steps("rows", "steps")
                .expect("rows declares steps");
            for step in steps {
                let verb = step["verb"].as_str().expect("a tour step has a verb");
                words.push(verb.to_owned());
                let spec = declared
                    .verbs()
                    .iter()
                    .find(|spec| spec.verb() == verb)
                    .expect("a tour verb is declared");
                append_arguments(&mut words, step, spec.arguments());
            }
        }
    } else {
        words.push(command.name.to_owned());
        let declared = declarations
            .spec_named(command.name)
            .expect("a tour command is declared");
        append_arguments(&mut words, arguments, &declared.arguments);
    }
    words.join(" ")
}

/// An argument value as the command line writes it.
fn scalar_text(value: &Value) -> String {
    value
        .as_str()
        .map_or_else(|| value.to_string(), str::to_owned)
}

fn append_arguments(
    words: &mut Vec<String>,
    arguments: &Value,
    declared: &[command_surface::ArgumentSpec],
) {
    let quote = polylogue_render::shell_quote;
    for arg in declared.iter().filter(|arg| !arg.calling_process) {
        let Some(value) = arguments.get(&arg.name) else {
            continue;
        };
        if value.is_null() || value == false {
            continue;
        }
        if !matches!(
            arg.form,
            command_surface::ArgumentForm::Positional
                | command_surface::ArgumentForm::Words
                | command_surface::ArgumentForm::Rest
        ) {
            words.push(format!("--{}", arg.name.replace('_', "-")));
        }
        if value == true {
            continue;
        }
        if let Some(values) = value.as_array() {
            words.extend(values.iter().map(|v| quote(&scalar_text(v))));
        } else {
            words.push(quote(&scalar_text(value)));
        }
    }
}
