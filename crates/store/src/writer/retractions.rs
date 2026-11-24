//! Sinex lifecycle changes retract source contributions through the writer:
//! each names one chain whose reading is removed.
use super::*;
use std::collections::BTreeSet;

/// A retraction is one plain reset of each chain it names, with no rows.
pub(crate) fn check_source_retractions(batches: &[Batch]) -> Result<()> {
    let mut distinct = BTreeSet::new();
    for batch in batches {
        let Change::Reparse { chain } = batch.change else {
            return Err(StoreError::Invalid(
                "source retraction names no chain".into(),
            ));
        };
        if chain < 1 || !distinct.insert((&batch.input.id, chain)) {
            return Err(StoreError::Invalid(
                "a retraction must name each valid input chain once".into(),
            ));
        }
        let [step] = batch.parse.as_slice() else {
            return Err(StoreError::Invalid(
                "source retraction needs one plain reset".into(),
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
                "a Sinex lifecycle change cannot rebuild parser values".into(),
            ));
        }
    }
    Ok(())
}

/// Removes the readings of the chains the retractions name; their shown
/// rows are computed again when the transaction ends.
pub(crate) fn apply_source_retractions(
    tx: &Transaction<'_>,
    batches: &[Batch],
    counter: i64,
    changes: &crate::change_summary::ChangeSummary,
    touched: &mut Touched,
) -> Result<()> {
    check_source_retractions(batches)?;
    for batch in batches {
        let Change::Reparse { chain } = batch.change else {
            unreachable!("checked above");
        };
        for step in &batch.parse {
            rows::write_step(
                tx,
                batch,
                step,
                rows::StepWrite {
                    counter,
                    chain,
                    source_commit: None,
                    replacement: false,
                    retraction: true,
                    primary: false,
                    changes,
                },
                touched,
            )?;
        }
    }
    Ok(())
}
