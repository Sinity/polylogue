//! Question custody is acknowledged separately from asynchronous execution history.
use super::*;
use polylogue_model::query_objects::{QueryCustody, QueryObject, QueryRun, QueryRunOutcome};

/// Semantic/scope validation only. This private value is neither a custody
/// receipt nor evidence of native execution eligibility.
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

fn object_via(
    queue: &WriterQueue,
    object: QueryObject,
    parent: Option<Arc<ParentLifetime>>,
) -> Result<QueryCustody> {
    let (reply, wait) = sync_channel(1);
    queue
        .send_user(UserMsg::QueryObject(Box::new(object), parent, reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

fn begin_via(
    queue: &WriterQueue,
    run: QueryRun,
    parent: Option<Arc<ParentLifetime>>,
) -> Result<QueryCustody> {
    let (reply, wait) = sync_channel(1);
    queue
        .send_user(UserMsg::QueryRun(Box::new(run), parent, reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

fn finish_via(
    queue: &WriterQueue,
    outcome: QueryRunOutcome,
) -> Result<Receiver<Result<QueryCustody>>> {
    let (reply, wait) = sync_channel(1);
    queue
        .send_user(UserMsg::QueryRunOutcome(Box::new(outcome), reply))
        .map_err(|_| StoreError::WriterGone)?;
    Ok(wait)
}

/// Space admitted before a read, consumed by its completed execution record.
/// Dropping it returns capacity without creating a run.
pub struct HistorySlot {
    slot: super::user_queue::Reservation,
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

impl Writer {
    /// Establishes durable custody before a response exposes the question reference.
    pub fn query_object(&self, object: QueryObject) -> Result<QueryCustody> {
        object_via(
            self.tx.as_ref().ok_or(StoreError::WriterGone)?,
            object,
            None,
        )
    }

    /// Acknowledges immutable inputs before a continuation exposes its run reference.
    pub fn begin_query_run(&self, run: QueryRun) -> Result<QueryCustody> {
        begin_via(self.tx.as_ref().ok_or(StoreError::WriterGone)?, run, None)
    }

    /// Dropping the completion receipt never cancels accepted history work.
    pub fn finish_query_run(
        &self,
        outcome: QueryRunOutcome,
    ) -> Result<Receiver<Result<QueryCustody>>> {
        finish_via(self.tx.as_ref().ok_or(StoreError::WriterGone)?, outcome)
    }
}

impl WriterSender {
    /// Reserve bounded writer capacity before preparing an ordinary read.
    /// The request's existing deadline and cancellation cover this admission.
    pub fn reserve_history(&self) -> Result<HistorySlot> {
        crate::read::limit::ReadLimit::set_stage("query history admission");
        let limit = crate::read::limit::ReadLimit::current();
        let slot = self.0.users.reserve(limit.as_ref())?;
        Ok(HistorySlot {
            slot,
            writer: self.0.clone(),
        })
    }

    pub fn query_object_with_parent(
        &self,
        object: QueryObject,
        parent: Arc<ParentLifetime>,
    ) -> Result<QueryCustody> {
        object_via(&self.0, object, Some(parent))
    }
    pub fn begin_query_run_with_parent(
        &self,
        run: QueryRun,
        parent: Arc<ParentLifetime>,
    ) -> Result<QueryCustody> {
        begin_via(&self.0, run, Some(parent))
    }
    pub fn query_object(&self, object: QueryObject) -> Result<QueryCustody> {
        object_via(&self.0, object, None)
    }

    pub fn begin_query_run(&self, run: QueryRun) -> Result<QueryCustody> {
        begin_via(&self.0, run, None)
    }
    pub fn finish_query_run(
        &self,
        outcome: QueryRunOutcome,
    ) -> Result<Receiver<Result<QueryCustody>>> {
        finish_via(&self.0, outcome)
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

    /// The reusable question has its own natural name in primary custody;
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
            || outcome.binding_hash
                != run
                    .binding_hash()
                    .map_err(|error| StoreError::Invalid(error.to_string()))?
            || outcome.run_hash
                != run
                    .record_hash()
                    .map_err(|error| StoreError::Invalid(error.to_string()))?
        {
            return Err(StoreError::Invalid(
                "query history has inconsistent execution identities".into(),
            ));
        }
        if self.sinex.is_some() {
            self.primary_cache_ready()?;
            // Refuse a conflicting canonical Begin or Outcome before adding
            // another immutable intent to producer custody.
            let run_known = crate::query_objects::known_run(&self.index, run)?;
            let outcome_known = crate::query_objects::known_run_outcome(&self.index, outcome)?;
            // Questions are shared by runs and remain directly addressable
            // through their existing query:<hash> custody name.
            self.write_query_object(object)?;
            if run_known && outcome_known {
                return Ok(());
            }
            self.sinex
                .as_ref()
                .expect("primary writer")
                .prepare_named_products(
                    &format!("query-run:{}", run.id),
                    &[
                        crate::primary::ProductWrite {
                            product: crate::primary::Product::QueryRun {
                                run: Box::new(run.clone()),
                            },
                            location: None,
                            parents: Vec::new(),
                            machine: Some(run.machine.clone()),
                            output_index: 0,
                            publish: true,
                        },
                        crate::primary::ProductWrite {
                            product: crate::primary::Product::QueryRunOutcome {
                                outcome: Box::new(outcome.clone()),
                            },
                            location: None,
                            parents: Vec::new(),
                            machine: None,
                            output_index: 1,
                            publish: true,
                        },
                    ],
                )?;
            return Ok(());
        }
        crate::query_objects::validate_result_bindings(
            &self.archive,
            crate::source::Source::of(&self.archive)?,
            object,
        )?;
        let tx = self.archive.transaction()?;
        let counter = bump_counter(&tx)?;
        let before = tx.total_changes();
        crate::query_objects::apply_object(&tx, object, counter)?;
        crate::query_objects::apply_run(&tx, run, counter, 0)?;
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

    pub(super) fn write_query_object(&mut self, object: &QueryObject) -> Result<QueryCustody> {
        crate::query_objects::validate_object(object)?;
        let conn = if self.sinex.is_some() {
            self.primary_cache_ready()?;
            &self.index
        } else {
            &self.archive
        };
        crate::query_objects::validate_result_bindings(
            conn,
            crate::source::Source::of(conn)?,
            object,
        )?;
        self.persist_query_object(object, None)
    }

    /// Both callers have validated semantic bodies and result schemas: the
    /// ordinary object path against canonical rows, or the sealed batch against
    /// its actual original plus pending authored view.
    fn persist_query_object(
        &mut self,
        object: &QueryObject,
        scope: Option<&crate::ReadScopeToken>,
    ) -> Result<QueryCustody> {
        scoped::require_current(&self.location, scope)?;
        if let Some(sinex) = &self.sinex {
            self.primary_cache_ready()?;
            scoped::require_current(&self.location, scope)?;
            if crate::query_objects::known_object(&self.index, object)? {
                return Ok(QueryCustody::Canonical);
            }
            let mut queued = false;
            sinex.visit_named_products(
                &crate::primary::NamedProductSelector::Exact(format!("query:{}", object.id)),
                &mut |product| {
                    if let crate::primary::Product::Query { object: existing } = product
                        && existing.id == object.id
                    {
                        if !crate::query_objects::equivalent(existing, object) {
                            return Err(StoreError::Conflict(format!("query:{}", object.id)));
                        }
                        queued = true;
                    }
                    Ok(())
                },
            )?;
            if queued {
                return Ok(QueryCustody::Queued);
            }
            scoped::require_current(&self.location, scope)?;
            sinex.prepare_named_products(
                &format!("query:{}", object.id),
                &[crate::primary::ProductWrite {
                    product: crate::primary::Product::Query {
                        object: Box::new(object.clone()),
                    },
                    location: None,
                    parents: Vec::new(),
                    machine: None,
                    output_index: 0,
                    publish: true,
                }],
            )?;
            scoped::require_current(&self.location, scope)?;
            return Ok(QueryCustody::Queued);
        }
        let tx = self.archive.transaction()?;
        let counter = bump_counter(&tx)?;
        let before = tx.total_changes();
        crate::query_objects::apply_object(&tx, object, counter)?;
        if tx.total_changes() != before {
            scoped::require_current(&self.location, scope)?;
            crate::write_lock::commit(tx)?;
            self.commits
                .committed(crate::completion::Committed::QUERY_RECORDS);
            self.report.transactions += 1;
        } else {
            tx.rollback()?;
        }
        Ok(QueryCustody::Canonical)
    }

    pub(super) fn write_query_run(&mut self, run: &QueryRun) -> Result<QueryCustody> {
        crate::query_objects::validate_run(run)?;
        if let Some(sinex) = &self.sinex {
            match self.primary_cache_ready() {
                Ok(()) => {
                    if crate::query_objects::known_run(&self.index, run)? {
                        return Ok(QueryCustody::Canonical);
                    }
                }
                Err(StoreError::ScopeChanged) => {}
                Err(error) => return Err(error),
            }

            sinex.prepare_named_products(
                &format!("query-run:{}", run.id),
                &[crate::primary::ProductWrite {
                    product: crate::primary::Product::QueryRun {
                        run: Box::new(run.clone()),
                    },
                    location: None,
                    parents: Vec::new(),
                    machine: Some(run.machine.clone()),
                    output_index: 0,
                    publish: true,
                }],
            )?;
            let mut visible = false;
            sinex.visit_named_products(
                &crate::primary::NamedProductSelector::Exact(format!("query-run:{}", run.id)),
                &mut |product| {
                    if let crate::primary::Product::QueryRun { run: stored } = product
                        && stored.id == run.id
                    {
                        if **stored != *run {
                            return Err(StoreError::Conflict(format!("run:{}", run.id)));
                        }
                        visible = true;
                    }
                    Ok(())
                },
            )?;
            if !visible {
                return Err(StoreError::Sinex(
                    "acknowledged query run is unavailable from named custody".into(),
                ));
            }
            return Ok(QueryCustody::Queued);
        }
        let tx = self.archive.transaction()?;
        let sequence = bump_counter(&tx)?;
        let before = tx.total_changes();
        crate::query_objects::apply_run(&tx, run, sequence, 0)?;
        if tx.total_changes() != before {
            crate::write_lock::commit(tx)?;
            self.commits
                .committed(crate::completion::Committed::QUERY_RECORDS);
            self.report.transactions += 1;
        } else {
            tx.rollback()?;
        }
        Ok(QueryCustody::Canonical)
    }
    pub(super) fn write_query_run_outcome(
        &mut self,
        outcome: &QueryRunOutcome,
    ) -> Result<QueryCustody> {
        crate::query_objects::validate_run_outcome(outcome)?;
        if let Some(sinex) = &self.sinex {
            match self.primary_cache_ready() {
                Ok(()) => {
                    if crate::query_objects::known_run_outcome(&self.index, outcome)? {
                        return Ok(QueryCustody::Canonical);
                    }
                }
                Err(StoreError::ScopeChanged) => {}
                Err(error) => return Err(error),
            }

            sinex.prepare_named_products(
                &format!("query-run-outcome:{}", outcome.run),
                &[crate::primary::ProductWrite {
                    product: crate::primary::Product::QueryRunOutcome {
                        outcome: Box::new(outcome.clone()),
                    },
                    location: None,
                    parents: Vec::new(),
                    machine: None,
                    output_index: 0,
                    publish: true,
                }],
            )?;
            return Ok(QueryCustody::Queued);
        }
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
        Ok(QueryCustody::Canonical)
    }
}
