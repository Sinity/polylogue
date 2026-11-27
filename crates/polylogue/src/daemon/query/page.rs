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

const COMPILER_VERSION: &str = "shared-query-v5";

fn stale(message: &str) -> Failure {
    Failure::stale_cursor(format!("{message}; run the query again without its cursor"))
}

fn invalid(message: &str) -> Failure {
    Failure::from(Error::new(ErrorKind::InvalidQuery, message, 0))
}

/// The read positions a cursor records: the index instance, the index file and
/// the counters the read saw.
pub(crate) fn read_positions(
    position: &ChangePosition,
    index_file: (u64, u64),
) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("index_instance".into(), position.index_instance.clone()),
        (
            "index_file".into(),
            format!("{}:{}", index_file.0, index_file.1),
        ),
        ("index".into(), position.index.to_string()),
        ("archive".into(), position.archive.to_string()),
    ])
}

/// Whether the recorded read positions are still reachable from the current
/// ones: same index instance and index file, counters not ahead.
pub(crate) fn read_positions_unchanged(
    context: &CursorContext,
    position: &ChangePosition,
    index_file: (u64, u64),
) -> bool {
    let at_most = |key: &str, bound: i64| {
        context
            .read_positions
            .get(key)
            .and_then(|value| value.parse::<i64>().ok())
            .is_some_and(|value| value <= bound)
    };
    context.read_positions.get("index_instance") == Some(&position.index_instance)
        && context.read_positions.get("index_file")
            == Some(&format!("{}:{}", index_file.0, index_file.1))
        && at_most("index", position.index)
        && at_most("archive", position.archive)
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
    polylogue_store::read::open_cursor(
        seal,
        &position.index_instance,
        &SealedCursor::from_wire(text),
        question,
        |name| {
            if name == options.clock.day.zone_name() {
                Ok(options.clock.day.zone().clone())
            } else {
                Err(Error::new(
                    ErrorKind::StaleCursor,
                    "the configured query zone changed; run the query again without its cursor",
                    0,
                ))
            }
        },
        |context| {
            context.clock.day == options.clock.day
                && read_positions_unchanged(context, position, index_file)
                && context.versions.get("ranking").is_none_or(|previous| {
                    previous == &polylogue_store::vectors::writes().to_string()
                })
        },
    )
    .map_err(Failure::from)
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
        ("compiler".into(), COMPILER_VERSION.into()),
        ("binding".into(), binding(options, why)?),
        ("definitions".into(), definitions(program)?),
    ]);
    if program.near.is_some() {
        versions.insert(
            "ranking".into(),
            polylogue_store::vectors::writes().to_string(),
        );
    }
    let mut context = CursorContext {
        question,
        clock: options.clock.clone(),
        resolved_times: program.resolved_times.clone(),
        random_seed: seed,
        read_positions: read_positions(position, index_file),
        versions,
    };
    polylogue_store::query::continuation::compact_resolved_times(&mut context)?;
    Ok(context)
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
    if cursor.context.versions.get("compiler").map(String::as_str) != Some(COMPILER_VERSION)
        || cursor.context.versions.get("definitions") != Some(&definitions(program)?)
    {
        return Err(stale("the query compiler or saved definitions changed"));
    }
    polylogue_store::query::continuation::check_resolved_times(
        &cursor.context,
        &program.resolved_times,
    )?;
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
        .map_err(Failure::from)
}

#[cfg(test)]
mod tests;
