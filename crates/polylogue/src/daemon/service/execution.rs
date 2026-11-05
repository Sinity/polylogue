//! Durable Begin and immutable Outcome for one accepted execution, both
//! committed before the execution's answer returns.
use super::Failure;
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
    completed_at: impl FnOnce() -> Timestamp,
    execute: impl FnOnce() -> Result<Completed<T>, Failure>,
) -> Result<T, Failure> {
    match parent {
        Some(parent) => writer.begin_query_run_with_parent(run.clone(), parent),
        None => writer.begin_query_run(run.clone()),
    }?;
    let result = execute();
    let outcome = QueryRunOutcome {
        run: run.id.clone(),
        binding_hash,
        run_hash: run.record_hash().map_err(|e| Failure::new("internal", e))?,
        at: completed_at(),
        outcome: match &result {
            Ok(_) => RunOutcome::Ok,
            Err(failure) => RunOutcome::Error {
                kind: failure.code().into(),
            },
        },
        count: result.as_ref().ok().and_then(|completed| completed.count),
        exact: result.as_ref().is_ok_and(|completed| completed.exact),
        next: result
            .as_ref()
            .ok()
            .and_then(|completed| completed.next.clone()),
    };
    // The answer cites `run:<id>`, so its outcome is committed before the
    // answer returns: a caller that reads the run next sees how it ended.
    let committed = writer.finish_query_run(outcome).and_then(|receipt| {
        receipt
            .recv()
            .map_err(|_| polylogue_store::StoreError::WriterGone)?
    });
    match result {
        Ok(completed) => {
            committed?;
            Ok(completed.value)
        }
        Err(error) => Err(error),
    }
}
