//! Local artifact comparison and explicit CLI output; inference runs in the daemon.
use crate::commands::Command;
use crate::daemon::schema::Inference;
use polylogue_sources::schema;
use std::{
    fs::File,
    io::{self, BufReader, Write},
    process::ExitCode,
};

pub fn diff(
    _: &Command,
    values: &command_surface::ParsedCommand,
    output: crate::config::OutputConfig,
    json: bool,
) -> ExitCode {
    run(values, json, output, || {
        let read = |name| -> Result<Inference, String> {
            let file = File::open(
                values
                    .request
                    .arguments
                    .get(name)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_default(),
            )
            .map_err(|_| format!("cannot read {name} schema"))?;
            serde_json::from_reader(BufReader::new(file))
                .map_err(|_| format!("invalid {name} schema artifact"))
        };
        let before = read("before")?;
        let after = read("after")?;
        let mut origins = serde_json::Map::new();
        let mut reasons = std::collections::BTreeSet::new();
        if before.selection != after.selection {
            reasons.insert("different_selection_coverage");
        }
        if before.selection.selected_inputs == 0 || after.selection.selected_inputs == 0 {
            reasons.insert("empty_selection");
        }
        let names: std::collections::BTreeSet<_> =
            before.origins.keys().chain(after.origins.keys()).collect();
        for name in names {
            for schema in [before.origins.get(name), after.origins.get(name)]
                .into_iter()
                .flatten()
            {
                // Refuse a malformed artifact on each side, including an origin the other side lacks.
                schema::validate(schema)?;
            }
            let report = match (before.origins.get(name), after.origins.get(name)) {
                (Some(left), Some(right)) => {
                    let compared = schema::compare(left, right)?;
                    if !compared.comparable {
                        reasons.insert("incomparable_origin_samples");
                    }
                    serde_json::to_value(compared).map_err(|e| e.to_string())?
                }
                (left, right) => {
                    reasons.insert("different_origins");
                    serde_json::json!({"kind": if left.is_some() { "removed_origin" } else { "added_origin" }, "before": left, "after": right})
                }
            };
            origins.insert(name.clone(), report);
        }
        let report = serde_json::json!({"origins": origins, "before": before.selection, "after": after.selection, "comparable": reasons.is_empty(), "reasons": reasons});
        serde_json::to_value(report).map_err(|e| e.to_string())
    })
}

pub fn output(
    values: &command_surface::ParsedCommand,
    answer: &command_surface::Answer<serde_json::Value>,
    client: &crate::config::Client,
    json: bool,
) -> ExitCode {
    let value = answer
        .value()
        .expect("output handlers receive answers with values");
    let destination = match destination(values) {
        Ok(destination) => destination,
        Err(message) => {
            return super::fail(
                client.output,
                json,
                polylogue_model::Code::BadRequest,
                &message,
            );
        }
    };
    if json && destination.is_none() {
        println!(
            "{}",
            polylogue_render::terminal::json(
                &serde_json::to_value(answer).expect("answer serializes"),
                false
            )
        );
        return ExitCode::from(answer.exit_code());
    }
    run(values, json, client.output, || Ok(value.clone()))
}

/// Where `--to` sends the result; standard output without it.
fn destination(values: &command_surface::ParsedCommand) -> Result<Option<String>, String> {
    use crate::cli::calling_process;
    calling_process::options::<calling_process::SchemaOutput>(values).map(|options| options.to)
}

fn run(
    values: &command_surface::ParsedCommand,
    json: bool,
    output: crate::config::OutputConfig,
    action: impl FnOnce() -> Result<serde_json::Value, String>,
) -> ExitCode {
    let result = action().and_then(|value| {
        let mut bytes = Vec::new();
        let format = if json {
            polylogue_render::Format::Json
        } else {
            polylogue_render::Format::Terminal
        };
        polylogue_render::schema::render(&value, format, &mut bytes).map_err(|e| e.to_string())?;
        match destination(values)? {
            Some(path) => std::fs::write(path, bytes).map_err(|e| e.to_string()),
            None => io::stdout()
                .lock()
                .write_all(&bytes)
                .map_err(|e| e.to_string()),
        }
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => super::fail(output, json, polylogue_model::Code::Schema, &message),
    }
}

#[cfg(test)]
mod tests;
