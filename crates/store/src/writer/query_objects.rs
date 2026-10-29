//! Question custody is acknowledged separately from asynchronous execution history.
use super::*;
use polylogue_model::query_objects::{QueryCustody, QueryObject, QueryRun, QueryRunOutcome};

fn object_via(queue: &WriterQueue, object: QueryObject) -> Result<QueryCustody> {
    let (reply, wait) = sync_channel(1);
    queue
        .send_user(UserMsg::QueryObject(Box::new(object), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

fn begin_via(queue: &WriterQueue, run: QueryRun) -> Result<QueryCustody> {
    let (reply, wait) = sync_channel(1);
    queue
        .send_user(UserMsg::QueryRun(Box::new(run), reply))
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
        object_via(self.tx.as_ref().ok_or(StoreError::WriterGone)?, object)
    }

    /// Acknowledges immutable inputs before a continuation exposes its run reference.
    pub fn begin_query_run(&self, run: QueryRun) -> Result<QueryCustody> {
        begin_via(self.tx.as_ref().ok_or(StoreError::WriterGone)?, run)
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
    pub fn query_object(&self, object: QueryObject) -> Result<QueryCustody> {
        object_via(&self.0, object)
    }

    pub fn begin_query_run(&self, run: QueryRun) -> Result<QueryCustody> {
        begin_via(&self.0, run)
    }
    pub fn finish_query_run(
        &self,
        outcome: QueryRunOutcome,
    ) -> Result<Receiver<Result<QueryCustody>>> {
        finish_via(&self.0, outcome)
    }
}

impl Inner {
    pub(super) fn write_query_object(&mut self, object: &QueryObject) -> Result<QueryCustody> {
        crate::query_objects::validate_object(object)?;
        if let Some(sinex) = &self.sinex {
            self.primary_cache_ready()?;
            crate::query_objects::validate_result_bindings(
                &self.index,
                crate::source::Source::of(&self.index)?,
                object,
            )?;
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
            return Ok(QueryCustody::Queued);
        }
        crate::query_objects::validate_result_bindings(
            &self.archive,
            crate::source::Source::of(&self.archive)?,
            object,
        )?;
        let tx = self.archive.transaction()?;
        let before = tx.total_changes();
        crate::query_objects::apply_object(&tx, object)?;
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

    pub(super) fn write_query_run(&mut self, run: &QueryRun) -> Result<QueryCustody> {
        crate::query_objects::validate_run(run)?;
        if let Some(sinex) = &self.sinex {
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
