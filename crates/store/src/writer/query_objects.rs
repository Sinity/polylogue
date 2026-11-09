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
    /// Hand a read's question, its run and the run's outcome to the writer
    /// without waiting for them: the writer commits them on its next turn,
    /// after the answer has gone. The read never waits on custody.
    pub fn record_after_answer(
        &self,
        object: QueryObject,
        run: QueryRun,
        outcome: QueryRunOutcome,
    ) -> Result<()> {
        let (object_reply, _) = sync_channel(1);
        self.0
            .send_user(UserMsg::QueryObject(Box::new(object), None, object_reply))
            .map_err(|_| StoreError::WriterGone)?;
        let (run_reply, _) = sync_channel(1);
        self.0
            .send_user(UserMsg::QueryRun(Box::new(run), None, run_reply))
            .map_err(|_| StoreError::WriterGone)?;
        let (outcome_reply, _) = sync_channel(1);
        self.0
            .send_user(UserMsg::QueryRunOutcome(Box::new(outcome), outcome_reply))
            .map_err(|_| StoreError::WriterGone)?;
        Ok(())
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
            self.commits.committed();
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
            self.commits.committed();
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
            self.commits.committed();
            self.report.transactions += 1;
        } else {
            tx.rollback()?;
        }
        Ok(QueryCustody::Canonical)
    }
}
