//! Native lifecycle changes retract source contributions; only the parser supplies values.
use super::*;

pub(crate) fn apply_confirmed_readings(
    tx: &Transaction<'_>,
    readings: &mut [(InputKey, i64, u64, ChainReading)],
    counter: i64,
    changes: &crate::change_summary::ChangeSummary,
) -> Result<()> {
    for (input, chain, length, reading) in readings {
        let batch = Batch {
            required_disposition: None,
            input: input.clone(),
            change: Change::Reparse { chain: *chain },
            observation: None,
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length: *length,
            parse: Vec::new(),
            acquisition: None,
            basis: None,
            release: Default::default(),
        };
        reading.visit_source(|step| {
            if step.delivery.is_some() || step.complete_output()? != Output::default() {
                return Err(StoreError::Invalid(
                    "native lifecycle cannot reconstruct parser values".into(),
                ));
            }
            Ok(())
        })?;
        reading.visit(*chain, |step, resolution| {
            rows::write_confirmed_rows(tx, &batch, step, resolution, counter, None, None, changes)?;
            Ok(())
        })?;
        rows::prune_sessions(tx, &input.id, *chain, counter)?;
    }
    Ok(())
}
