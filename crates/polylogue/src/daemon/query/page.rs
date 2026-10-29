//! The shared authenticated continuation binds one question to its read context.
use super::{Failure, Options, Program};
use polylogue_model::ChangePosition;
use query_language::cursor::{
    Continuation, CursorContext, CursorSeal, OrderPosition, SealedCursor,
};
use query_language::identity::QuestionId;
use query_language::{Error, ErrorKind};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

fn failure(error: Error) -> Failure {
    Failure {
        kind: error.kind.code().into(),
        message: error.to_string(),
        detail: serde_json::to_value(error).ok(),
    }
}
fn invalid(message: &str) -> Failure {
    failure(Error::new(ErrorKind::InvalidQuery, message, 0))
}
fn stale(message: &str) -> Failure {
    failure(
        Error::new(ErrorKind::StaleCursor, message, 0)
            .hint("run the query again without its cursor"),
    )
}

/// The caller supplies the secret from its current reader; it never enters a
/// request, answer, public Debug value, question identity or execution log.
pub(super) struct Paging<'a> {
    pub seal: &'a CursorSeal,
    pub context: CursorContext,
}

#[cfg(test)]
pub(super) fn identity(text: &str, options: &Options) -> Result<QuestionId, Failure> {
    Ok(polylogue_query::compile(text, options)?.question_id())
}

pub(super) fn open(
    text: &str,
    seal: &CursorSeal,
    question: QuestionId,
    position: &ChangePosition,
    index_file: (u64, u64),
    options: &Options,
) -> Result<Continuation, Failure> {
    seal.open(
        &SealedCursor::from_wire(text),
        question,
        |name| {
            if name == options.clock.day.zone_name() {
                Ok(options.clock.day.zone().clone())
            } else {
                Err(Error::new(
                    ErrorKind::StaleCursor,
                    "the configured query zone changed",
                    0,
                ))
            }
        },
        |context| {
            context.clock.day == options.clock.day
                && context.read_positions.get("generation") == Some(&position.generation)
                && context.read_positions.get("index_file")
                    == Some(&format!("{}:{}", index_file.0, index_file.1))
                && context
                    .read_positions
                    .get("index")
                    .and_then(|n| n.parse::<i64>().ok())
                    .is_some_and(|n| n <= position.index)
                && context
                    .read_positions
                    .get("archive")
                    .and_then(|n| n.parse::<i64>().ok())
                    .is_some_and(|n| n <= position.archive)
                && context.versions.get("ranking").is_none_or(|previous| {
                    previous == &polylogue_store::vectors::writes().to_string()
                })
        },
    )
    .map_err(failure)
}

fn context_digest(domain: &str, value: &impl Serialize) -> Result<String, Failure> {
    let mut digest = blake3::Hasher::new();
    digest.update(b"polylogue.query.cursor-context\0");
    digest.update(domain.as_bytes());
    digest.update(b"\0");
    serde_json::to_writer(&mut digest, value)
        .map_err(|_| invalid("the query context cannot be encoded"))?;
    Ok(digest.finalize().to_hex().to_string())
}

fn binding(options: &Options, why: bool) -> Result<String, Failure> {
    context_digest(
        "options",
        &(
            &options.self_session,
            &options.fields,
            options.embeddings,
            why,
            options.clock.day.wire(),
        ),
    )
}
fn definitions(program: &Program) -> Result<String, Failure> {
    let mut definitions: Vec<_> = program.definitions.iter().collect();
    definitions.sort_by(|a, b| (&a.name, &a.revision).cmp(&(&b.name, &b.revision)));
    definitions.dedup();
    context_digest("definitions", &definitions)
}

pub(super) fn context(
    question: QuestionId,
    position: &ChangePosition,
    index_file: (u64, u64),
    options: &Options,
    program: &Program,
    seed: u64,
    why: bool,
) -> Result<CursorContext, Failure> {
    let mut versions = BTreeMap::from([
        ("compiler".into(), "shared-query-v1".into()),
        ("binding".into(), binding(options, why)?),
        ("definitions".into(), definitions(program)?),
    ]);
    if program.near.is_some() {
        versions.insert(
            "ranking".into(),
            polylogue_store::vectors::writes().to_string(),
        );
    }
    Ok(CursorContext {
        question,
        clock: options.clock.clone(),
        resolved_times: program.resolved_times.clone(),
        random_seed: seed,
        read_positions: BTreeMap::from([
            ("generation".into(), position.generation.clone()),
            (
                "index_file".into(),
                format!("{}:{}", index_file.0, index_file.1),
            ),
            ("index".into(), position.index.to_string()),
            ("archive".into(), position.archive.to_string()),
        ]),
        versions,
    })
}

pub(super) fn check(
    cursor: &Continuation,
    program: &Program,
    options: &Options,
    why: bool,
) -> Result<Vec<Value>, Failure> {
    if cursor.context.versions.get("binding") != Some(&binding(options, why)?)
        || cursor.after.values.len() != program.order.len()
    {
        return Err(invalid(
            "the cursor belongs to different query options or ordering",
        ));
    }
    if cursor.context.versions.get("compiler").map(String::as_str) != Some("shared-query-v1")
        || cursor.context.versions.get("definitions") != Some(&definitions(program)?)
    {
        return Err(stale("the query compiler or saved definitions changed"));
    }
    if cursor.context.resolved_times != program.resolved_times {
        return Err(stale("the resolved query times changed"));
    }
    Ok(cursor.after.values.clone())
}

pub(super) fn after_row(
    program: &Program,
    row: &Value,
    paging: &Paging<'_>,
) -> Result<String, Failure> {
    let values: Vec<Value> = program
        .order
        .iter()
        .map(|key| row[&key.column].clone())
        .collect();
    // The complete product key is already total, with its stable identity last.
    let row_identity =
        serde_json::to_string(&values).map_err(|_| invalid("the row key cannot be encoded"))?;
    let continuation = Continuation {
        context: paging.context.clone(),
        after: OrderPosition {
            values,
            row_identity,
        },
    };
    paging
        .seal
        .seal(&continuation)
        .map(|cursor| cursor.as_str().to_owned())
        .map_err(failure)
}

#[cfg(test)]
mod tests;
