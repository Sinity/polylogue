//! Saved questions are written before their asynchronous execution history.
use super::*;
use polylogue_model::query_objects::{QueryObject, QueryRun, QueryRunOutcome};

/// Content and scope validation of a saved query.s objects.
pub(super) struct PreparedObjects {
    scope: crate::ReadScopeToken,
    queries: crate::selection::ProvisionalQueries,
}
impl PreparedObjects {
    pub(super) fn validate(
        authored: &Connection,
        scope: crate::ReadScopeToken,
        queries: crate::selection::ProvisionalQueries,
    ) -> Result<Self> {
        if authored.is_autocommit() {
            return Err(StoreError::Invalid(
                "query batch validation requires its pinned authored view".into(),
            ));
        }
        scope.check()?;
        let source = crate::source::Source::reading(authored)?;
        for object in queries.objects() {
            crate::query_objects::validate_object(object)?;
            crate::query_objects::validate_result_bindings(authored, source, object)?;
            scope.check()?;
        }
        Ok(Self { scope, queries })
    }
}

/// Space reserved before a read, consumed by its completed execution record.
/// Dropping it returns capacity without creating a run.
pub struct HistorySlot {
    slot: super::user_queue::QueueSlot,
    writer: WriterQueue,
}

pub(super) struct History {
    pub(super) object: QueryObject,
    pub(super) run: QueryRun,
    pub(super) outcome: QueryRunOutcome,
}

impl HistorySlot {
    /// Hand one whole record to the writer, without waiting for a commit.
    pub fn record(
        self,
        object: QueryObject,
        run: QueryRun,
        outcome: QueryRunOutcome,
    ) -> Result<()> {
        self.slot.push(UserMsg::History(Box::new(History {
            object,
            run,
            outcome,
        })))?;
        self.writer.wake();
        Ok(())
    }
}

impl WriterSender {
    /// Reserve bounded writer capacity before preparing a read. The request's
    /// existing deadline and cancellation cover this reservation.
    pub fn reserve_history(&self) -> Result<HistorySlot> {
        crate::read::limit::ReadLimit::set_stage("query history reservation");
        let limit = crate::read::limit::ReadLimit::current();
        let slot = self.0.users.reserve(limit.as_ref())?;
        Ok(HistorySlot {
            slot,
            writer: self.0.clone(),
        })
    }

    /// Stores the question before a response exposes its reference.
    pub fn query_object(
        &self,
        object: QueryObject,
        parent: Option<Arc<ParentLifetime>>,
    ) -> Result<()> {
        let (reply, wait) = sync_channel(1);
        self.0
            .send_user(UserMsg::QueryObject(Box::new(object), parent, reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    /// Acknowledges immutable inputs before a continuation exposes its run reference.
    pub fn begin_query_run(
        &self,
        run: QueryRun,
        parent: Option<Arc<ParentLifetime>>,
    ) -> Result<()> {
        let (reply, wait) = sync_channel(1);
        self.0
            .send_user(UserMsg::QueryRun(Box::new(run), parent, reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    /// Dropping the completion answer never cancels accepted history work.
    pub fn finish_query_run(&self, outcome: QueryRunOutcome) -> Result<Receiver<Result<()>>> {
        let (reply, wait) = sync_channel(1);
        self.0
            .send_user(UserMsg::QueryRunOutcome(Box::new(outcome), reply))
            .map_err(|_| StoreError::WriterGone)?;
        Ok(wait)
    }
}

impl Inner {
    pub(super) fn write_prepared_objects(
        &mut self,
        prepared: PreparedObjects,
    ) -> Result<crate::ReadScopeToken> {
        for object in prepared.queries.objects() {
            scoped::require_current(&self.location, Some(&prepared.scope))?;
            self.persist_query_object(object, Some(&prepared.scope))?;
            scoped::require_current(&self.location, Some(&prepared.scope))?;
        }
        Ok(prepared.scope)
    }

    /// The reusable question has its own name in the producer spool;
    /// one completed run and outcome form one immutable event group.
    pub(super) fn write_history(&mut self, history: &History) -> Result<()> {
        let History {
            object,
            run,
            outcome,
        } = history;
        crate::query_objects::validate_object(object)?;
        crate::query_objects::validate_run(run)?;
        crate::query_objects::validate_run_outcome(outcome)?;
        if run.question != Some(object.id)
            || outcome.run != run.id
            || outcome.binding_hash != run.binding_hash().map_err(crate::StoreError::invalid)?
            || outcome.run_hash != run.record_hash().map_err(crate::StoreError::invalid)?
        {
            return Err(StoreError::Invalid(
                "query history has inconsistent execution identities".into(),
            ));
        }
        crate::query_objects::validate_result_bindings(
            &self.archive,
            crate::source::Source::of(&self.archive)?,
            object,
        )?;
        let tx = self.archive.transaction()?;
        let commit_seq = bump_counter(&tx)?;
        let before = tx.total_changes();
        crate::query_objects::apply_object(&tx, object, commit_seq)?;
        crate::query_objects::apply_run(&tx, run, commit_seq, 0)?;
        crate::query_objects::apply_run_outcome(&tx, outcome)?;
        if tx.total_changes() != before {
            crate::write_lock::commit(tx)?;
            self.commits
                .committed(crate::completion::Committed::QUERY_RECORDS);
            self.report.transactions += 1;
        } else {
            tx.rollback()?;
        }
        Ok(())
    }

    pub(super) fn write_query_object(&mut self, object: &QueryObject) -> Result<()> {
        crate::query_objects::validate_object(object)?;
        crate::query_objects::validate_result_bindings(
            &self.archive,
            crate::source::Source::of(&self.archive)?,
            object,
        )?;
        self.persist_query_object(object, None)
    }

    /// Both callers have validated bodies and result schemas: the object path
    /// against stored rows, or the sealed batch against its original and
    /// authored view.
    fn persist_query_object(
        &mut self,
        object: &QueryObject,
        scope: Option<&crate::ReadScopeToken>,
    ) -> Result<()> {
        scoped::require_current(&self.location, scope)?;
        let tx = self.archive.transaction()?;
        let commit_seq = bump_counter(&tx)?;
        let before = tx.total_changes();
        crate::query_objects::apply_object(&tx, object, commit_seq)?;
        if tx.total_changes() != before {
            scoped::require_current(&self.location, scope)?;
            crate::write_lock::commit(tx)?;
            self.commits
                .committed(crate::completion::Committed::QUERY_RECORDS);
            self.report.transactions += 1;
        } else {
            tx.rollback()?;
        }
        Ok(())
    }

    pub(super) fn write_query_run(&mut self, run: &QueryRun) -> Result<()> {
        crate::query_objects::validate_run(run)?;
        let tx = self.archive.transaction()?;
        let commit_seq = bump_counter(&tx)?;
        let before = tx.total_changes();
        crate::query_objects::apply_run(&tx, run, commit_seq, 0)?;
        if tx.total_changes() != before {
            crate::write_lock::commit(tx)?;
            self.commits
                .committed(crate::completion::Committed::QUERY_RECORDS);
            self.report.transactions += 1;
        } else {
            tx.rollback()?;
        }
        Ok(())
    }
    pub(super) fn write_query_run_outcome(&mut self, outcome: &QueryRunOutcome) -> Result<()> {
        crate::query_objects::validate_run_outcome(outcome)?;
        let tx = self.archive.transaction()?;
        let before = tx.total_changes();
        crate::query_objects::apply_run_outcome(&tx, outcome)?;
        if tx.total_changes() != before {
            bump_counter(&tx)?;
            crate::write_lock::commit(tx)?;
            self.commits
                .committed(crate::completion::Committed::QUERY_RECORDS);
            self.report.transactions += 1;
        } else {
            tx.rollback()?;
        }
        Ok(())
    }
}
