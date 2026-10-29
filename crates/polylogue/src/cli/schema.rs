//! Local artifact comparison and explicit CLI output; inference runs in the daemon.
use crate::commands::{Command, Values};
use crate::daemon::schema::Inference;
use polylogue_sources::schema;
use std::{
    fs::File,
    io::{self, BufReader, Write},
    process::ExitCode,
};

pub fn diff(_: &Command, values: &Values, json: bool) -> ExitCode {
    run(values, json, || {
        let read = |name| -> Result<Inference, String> {
            let file = File::open(values.text(name).unwrap_or_default())
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
                schema::compare(schema, schema)?;
            }
            let report = match (before.origins.get(name), after.origins.get(name)) {
                (Some(left), Some(right)) => serde_json::to_value(schema::compare(left, right)?)
                    .map_err(|e| e.to_string())?,
                (left, right) => {
                    serde_json::json!({"kind": if left.is_some() { "removed_origin" } else { "added_origin" }, "before": left, "after": right})
                }
            };
            if report.get("kind").is_some() {
                reasons.insert("different_origins");
            } else if report["comparable"] != true {
                reasons.insert("incomparable_origin_samples");
            }
            origins.insert(name.clone(), report);
        }
        let report = serde_json::json!({"origins": origins, "before": before.selection, "after": after.selection, "comparable": reasons.is_empty(), "reasons": reasons});
        serde_json::to_value(report).map_err(|e| e.to_string())
    })
}

pub fn output(values: &Values, value: &serde_json::Value, json: bool) -> ExitCode {
    if json && values.text("output").is_none() {
        let answer = crate::protocol::answer(value.clone());
        println!(
            "{}",
            polylogue_render::terminal::json(
                &serde_json::to_value(&answer).expect("answer serializes"),
                false
            )
        );
        return ExitCode::from(answer.exit_code());
    }
    run(values, json, || Ok(value.clone()))
}

fn run(
    values: &Values,
    json: bool,
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
        match values.text("output") {
            Some(path) => std::fs::write(path, bytes).map_err(|e| e.to_string()),
            None => io::stdout()
                .lock()
                .write_all(&bytes)
                .map_err(|e| e.to_string()),
        }
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => super::fail(json, "schema", &message, 1),
    }
}

#[cfg(test)]
mod tests;
