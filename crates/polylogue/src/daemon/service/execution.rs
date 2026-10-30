//! Durable Begin and immutable Outcome for one accepted execution.
use super::{Response, error, store_error};
use polylogue_model::{
    BlobHash, Timestamp,
    query_objects::{QueryRun, QueryRunOutcome, RunOutcome},
    read::ReadyCall,
};
use polylogue_store::WriterSender;

/// The observed result of this invocation, without changing its frozen inputs.
pub(super) struct Completed<T> {
    pub value: T,
    pub count: Option<u64>,
    pub exact: bool,
    pub next: Option<ReadyCall>,
}

pub(super) fn record<T>(
    writer: &WriterSender,
    parent: Option<std::sync::Arc<polylogue_store::writer::ParentLifetime>>,
    run: QueryRun,
    binding_hash: BlobHash,
    failure_kind: &str,
    completed_at: impl FnOnce() -> Timestamp,
    execute: impl FnOnce() -> Result<Completed<T>, Response>,
) -> Result<T, Response> {
    match parent {
        Some(parent) => writer.begin_query_run_with_parent(run.clone(), parent),
        None => writer.begin_query_run(run.clone()),
    }
    .map_err(|e| store_error(&e))?;
    let result = execute();
    let outcome = QueryRunOutcome {
        run: run.id.clone(),
        binding_hash,
        run_hash: run.record_hash().map_err(|e| error("internal", e))?,
        at: completed_at(),
        outcome: match &result {
            Ok(_) => RunOutcome::Ok,
            Err(Response::Error { kind, .. }) => RunOutcome::Error { kind: kind.clone() },
            Err(_) => RunOutcome::Error {
                kind: failure_kind.into(),
            },
        },
        count: result.as_ref().ok().and_then(|completed| completed.count),
        exact: result.as_ref().is_ok_and(|completed| completed.exact),
        next: result
            .as_ref()
            .ok()
            .and_then(|completed| completed.next.clone()),
    };
    let completion = writer
        .finish_query_run(outcome)
        .map_err(|e| store_error(&e));
    match result {
        Ok(completed) => {
            drop(completion?);
            Ok(completed.value)
        }
        Err(error) => {
            drop(completion);
            Err(error)
        }
    }
}
