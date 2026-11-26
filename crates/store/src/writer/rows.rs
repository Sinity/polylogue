//! What one parse step writes in `index.db`: its rows into its chain's
//! reading ([`super::readings`]), the parse position they cover and the
//! step's counts and wire shapes, all in the transaction that commits them.
//! The shown rows of what the transaction's readings state are computed
//! when it ends ([`finish`]).

use super::readings::{self, Touched};
use super::*;
use crate::change_summary::ChangeSummary;

mod checkpoint;

/// How a step's rows are written.
pub(super) struct StepWrite<'a> {
    pub counter: i64,
    /// The chain the step's bytes were stored in.
    pub chain: i64,
    /// The record commit that holds the step's bytes.
    pub source_commit: &'a crate::wire_shapes::ReadingCommit,
    /// The first step of a complete replacement replaces the chain's wire
    /// shapes rather than adding to them.
    pub replacement: bool,
    pub changes: &'a ChangeSummary,
}

/// Writes a parse step's rows into its chain's reading, recording what they
/// state in `touched`.
pub(super) fn write_step(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    w: StepWrite<'_>,
    touched: &mut Touched,
) -> Result<()> {
    p.counts.check()?;
    p.prepared.validate(&p.output)?;
    super::hook_fields::validate(&p.output, &p.prepared.messages)?;
    let chain = w.chain;
    validate_range(b, p)?;
    if !crate::wire_shapes::admit_export(tx, &b.input.id, chain, b.length, p)? {
        return Ok(());
    }
    tx.prepare_cached(
        "INSERT INTO input_ref(id,format,route) VALUES (?1,?2,?3)
         ON CONFLICT(id) DO UPDATE SET format=excluded.format,route=excluded.route
         WHERE format IS NOT excluded.format OR route IS NOT excluded.route",
    )?
    .execute(params![b.input.id, b.input.format, b.input.route.as_str()])?;
    let input_pk: i64 = tx
        .prepare_cached("SELECT pk FROM input_ref WHERE id=?1")?
        .query_row([&b.input.id], |row| row.get(0))?;
    let previous = checkpoint::read(tx, input_pk, chain)?;
    if p.delivery.is_none() && !accept_step(previous.as_ref(), p)? {
        return Ok(());
    }
    if let Some(update) = &p.comparison {
        update.write(tx, input_pk)?;
    }
    let source_commit = w.source_commit;
    let checkpoint = checkpoint::Update::new(b, p, previous.as_ref(), w.counter)?;
    if p.reset {
        readings::delete(tx, w.changes, input_pk, chain, touched)?;
    }
    let committed = Some(source_commit.sequence).filter(|n| *n > 0);
    readings::write(
        tx,
        &readings::Chain {
            input_pk,
            input: &b.input.id,
            route: b.input.route,
            chain,
            committed,
            changes: w.changes,
        },
        b,
        p,
        touched,
    )?;
    let apply_wire = if w.replacement {
        crate::wire_shapes::replace
    } else {
        crate::wire_shapes::apply
    };
    apply_wire(
        tx,
        crate::wire_shapes::Source {
            input_pk,
            input: &b.input.id,
            format: &b.input.format,
            chain,
            length: b.length,
            commit_sequence: source_commit.sequence,
            commit_id: source_commit.id.as_deref(),
            observation: source_commit.observation.as_deref(),
            observed_at_ms: source_commit.observed_at_ms,
        },
        p,
    )?;
    checkpoint.write(tx, input_pk, chain)?;
    let a = &p.counts;
    let groups = [
        ("skipped:", &a.skipped, 1),
        ("unreadable:", &a.unreadable, 0),
        ("field:", &a.unread_fields, 0),
    ];
    for (prefix, map, counted_only) in groups {
        for (what, count) in map {
            tx.prepare_cached(
                "INSERT INTO unused(input_pk, chain, format, what, count, counted_only)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(input_pk, chain, what) DO UPDATE SET count = count + excluded.count",
            )?
            .execute(params![
                input_pk,
                chain,
                b.input.format,
                format!("{prefix}{what}"),
                *count as i64,
                counted_only,
            ])?;
        }
    }
    Ok(())
}

/// Ends a transaction that wrote readings: computes the shown rows of what
/// they state and marks the sessions it touched due for their summaries.
/// Returns the messages written.
pub(crate) fn finish(
    tx: &Transaction<'_>,
    touched: Touched,
    changes: &ChangeSummary,
    counter: i64,
) -> Result<u64> {
    let written = super::shown::compose(
        &super::shown::Context {
            tx,
            counter,
            changes,
        },
        touched,
    )?;
    finish_summaries(tx, changes, counter)?;
    Ok(written)
}

/// Marks the sessions a transaction's shown rows changed due for their
/// summaries. Nothing of a summary is computed here.
pub(super) fn finish_summaries(
    tx: &Transaction<'_>,
    changes: &ChangeSummary,
    counter: i64,
) -> Result<()> {
    changes.mark_due(tx, counter)?;
    changes.finish_links(tx)
}

fn validate_range(batch: &Batch, step: &ParseStep) -> Result<()> {
    if step.start > step.parsed_upto
        || step.parsed_upto > batch.length
        || i64::try_from(batch.length).is_err()
    {
        return Err(StoreError::Invalid(
            "reading range is outside its stored input".into(),
        ));
    }
    Ok(())
}

/// A continuing reading can only extend its committed prefix. Explicit resets
/// replace a complete reading, including row dumps whose accounting starts later.
fn accept_step(previous: Option<&checkpoint::Stored>, step: &ParseStep) -> Result<bool> {
    if step.reset {
        return Ok(true);
    }
    let cursor = previous.map_or(0, checkpoint::Stored::cursor);
    if step.start < cursor
        && step.parsed_upto <= cursor
        && previous.is_some_and(|prior| prior.version() == step.parse_version)
    {
        return Ok(false);
    }
    if step.start != cursor {
        return Err(StoreError::Invalid(format!(
            "reading starts at {} after committed position {cursor}",
            step.start
        )));
    }
    Ok(true)
}
