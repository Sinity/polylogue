//! The begin record and the fixed outcome of one accepted execution, both
//! committed before the execution's answer returns.
use crate::Error;
use polylogue_model::{
    BlobHash, Timestamp,
    query_objects::{QueryRun, QueryRunOutcome, RunOutcome},
    read::ReadyCall,
};
use polylogue_store::WriterSender;

/// The observed result of this invocation, without changing its recorded inputs.
pub struct Completed<T> {
    pub value: T,
    pub count: Option<u64>,
    pub exact: bool,
    pub next: Option<ReadyCall>,
}

pub fn record<T>(
    writer: &WriterSender,
    parent: Option<std::sync::Arc<polylogue_store::writer::ParentLifetime>>,
    run: QueryRun,
    binding_hash: BlobHash,
    completed_at: impl FnOnce() -> Timestamp,
    execute: impl FnOnce() -> Result<Completed<T>, Error>,
) -> Result<T, Error> {
    writer.archive().begin_query_run(run.clone(), parent)?;
    let result = execute();
    let outcome = QueryRunOutcome {
        run: run.id.clone(),
        binding_hash,
        run_hash: run
            .record_hash()
            .map_err(|error| Error::Internal(error.to_string()))?,
        at: completed_at(),
        outcome: match &result {
            Ok(_) => RunOutcome::Ok,
            Err(failure) => RunOutcome::Error {
                kind: failure.code().as_str().into(),
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
    let committed = writer
        .archive()
        .finish_query_run(outcome)
        .and_then(|committed| {
            committed
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
