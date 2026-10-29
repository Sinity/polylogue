//! One SSE batch of change references, prepared before delivery.
use super::{Failure, present, validate, values};
use polylogue_model::ChangePosition;
use polylogue_query::Options;
use polylogue_store::{Reader, StoreError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

/// Positions on event lines stay at the previous completed batch. Only
/// `Complete` gives a new position, after every preceding line.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Line {
    Begin {
        position: Option<String>,
        generation: String,
        replacement: bool,
        generation_changed: bool,
    },
    Change {
        position: Option<String>,
        component: Component,
        change: Value,
    },
    Complete {
        position: String,
        count: u64,
        replacement: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Component {
    Archive,
    Index,
}

fn invalid(message: &str) -> Failure {
    Failure {
        kind: "bad_request".into(),
        message: message.into(),
        detail: None,
    }
}

fn io(error: impl ToString) -> Failure {
    Failure {
        kind: "io".into(),
        message: error.to_string(),
        detail: None,
    }
}

fn write(file: &mut File, line: Line) -> Result<(), Failure> {
    let name = match &line {
        Line::Begin { .. } => "begin",
        Line::Change { .. } => "record",
        Line::Complete { .. } => "complete",
    };
    crate::protocol::stream::event(file, name, &line).map_err(io)
}

fn check_change(
    change: &Value,
    component: Component,
    target: &ChangePosition,
) -> Result<(), Failure> {
    let kind = change["kind"].as_str();
    let allowed = match component {
        Component::Archive => matches!(kind, Some("note")),
        Component::Index => matches!(kind, Some("session" | "removed")),
    };
    let bound = match component {
        Component::Archive => target.archive,
        Component::Index => target.index,
    };
    if !allowed
        || !change["ref"].as_str().is_some_and(|s| !s.is_empty())
        || !change["seq"].as_i64().is_some_and(|n| n >= 0 && n <= bound)
    {
        return Err(Failure {
            kind: "store".into(),
            message: "a change reference is outside its declared component or view".into(),
            detail: None,
        });
    }
    Ok(())
}

/// Prepare every row before delivery, releasing the reader before network writes.
pub(crate) fn prepare_with(
    reader: &Reader<'_>,
    since: Option<&str>,
    clock: &query_language::time::Clock,
    mut file: File,
) -> Result<File, Failure> {
    let previous = since
        .map(|s| ChangePosition::parse(s).ok_or_else(|| invalid("since takes GEN:COUNTER:ARCHIVE")))
        .transpose()?;
    let target = reader.change_position()?;
    let generation_changed = previous
        .as_ref()
        .is_some_and(|p| p.generation != target.generation);
    let replacement = previous.is_none() || generation_changed;
    if !replacement
        && previous
            .as_ref()
            .is_some_and(|p| p.index > target.index || p.archive > target.archive)
    {
        return Err(invalid("the change position is ahead of the archive"));
    }
    let position = previous.as_ref().map(ToString::to_string);
    write(
        &mut file,
        Line::Begin {
            position: position.clone(),
            generation: target.generation.clone(),
            replacement,
            generation_changed,
        },
    )?;
    // Archive local order is meaningful only within archive.db. Index-side
    // references follow as a separate component, never a combined chronology.
    let components = [
        (Component::Archive, "kind:note"),
        (
            Component::Index,
            if replacement {
                "kind:session"
            } else {
                "(kind:session OR kind:removed)"
            },
        ),
    ];
    let mut count = 0_u64;
    for (component, condition) in components {
        let after = if replacement {
            String::new()
        } else {
            format!(
                " AND after:{}",
                position.as_deref().expect("delta has a prior position")
            )
        };
        let text = format!("changes where {condition}{after} | sort by seq, kind, ref");
        // This fixed internal query has neither saved sources nor kept
        // results, so compilation need not load either archive-wide catalog.
        let opts = Options {
            clock: clock.clone(),
            origins: Vec::new(),
            command_words: Vec::new(),
            page_size: None,
            random_seed: Some(0),
            embeddings: false,
            self_session: None,
            saved: Default::default(),
            fields: Vec::new(),
            kept: Default::default(),
            chat_pages: Vec::new(),
        };
        let program = polylogue_store::query::compile(&text, &opts)?;
        validate(reader, &program)?;
        reader.query_each(
            &program.sql,
            &values(&program.params),
            |mut change| -> Result<(), Failure> {
                present(&program, &mut change)?;
                check_change(&change, component, &target)?;
                count = count
                    .checked_add(1)
                    .ok_or_else(|| io("change count exhausted"))?;
                write(
                    &mut file,
                    Line::Change {
                        position: position.clone(),
                        component,
                        change,
                    },
                )
            },
        )?;
    }
    write(
        &mut file,
        Line::Complete {
            position: target.to_string(),
            count,
            replacement,
        },
    )?;
    file.flush().map_err(io)?;
    file.seek(SeekFrom::Start(0)).map_err(io)?;
    Ok(file)
}

#[cfg(test)]
mod tests;
