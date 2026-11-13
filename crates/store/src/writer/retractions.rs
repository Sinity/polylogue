//! Native lifecycle retracts source contributions through the ordinary writer.
use super::*;
use std::collections::BTreeSet;

pub(crate) fn prepare_source_retractions(
    resolver: &Arc<resolve::Resolver>,
    batches: &mut [Batch],
) -> Result<()> {
    let mut distinct = BTreeSet::new();
    for batch in batches.iter() {
        let Change::Reparse { chain } = batch.change else {
            return Err(StoreError::Invalid(
                "source retraction has no exact chain".into(),
            ));
        };
        if chain < 1 || !distinct.insert((&batch.input.id, chain)) {
            return Err(StoreError::Invalid(
                "a position must name each valid input chain once".into(),
            ));
        }
        let [step] = batch.parse.as_slice() else {
            return Err(StoreError::Invalid(
                "source retraction needs one ordinary reset".into(),
            ));
        };
        if !step.reset
            || step.start != 0
            || step.parsed_upto != 0
            || !step.source_prefix_missing
            || step.delivery.is_some()
            || step.checkpoint.is_some()
            || step.output != Output::default()
        {
            return Err(StoreError::Invalid(
                "native lifecycle cannot reconstruct parser values".into(),
            ));
        }
    }
    resolver.batches(&mut batches.iter_mut().collect::<Vec<_>>())
}

pub(crate) fn apply_source_retractions(
    tx: &Transaction<'_>,
    batches: &[Batch],
    counter: i64,
    changes: &crate::change_summary::ChangeSummary,
) -> Result<()> {
    for batch in batches {
        let Change::Reparse { chain } = batch.change else {
            return Err(StoreError::Invalid(
                "source retraction has no exact chain".into(),
            ));
        };
        for step in &batch.parse {
            let resolution = step
                .resolution
                .as_ref()
                .filter(|r| r.chain == chain)
                .ok_or_else(|| {
                    StoreError::Invalid("source retraction was not resolved for its chain".into())
                })?;
            rows::retract_source(tx, batch, step, resolution, counter, changes)?;
        }
        rows::prune_sessions(tx, &batch.input.id, chain, counter)?;
    }
    Ok(())
}
