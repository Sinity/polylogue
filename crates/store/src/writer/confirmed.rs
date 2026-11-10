//! Applying prepared confirmed readings uses the ordinary row codecs.
use super::*;
use std::collections::BTreeMap;

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
        // Confirmed rematerialization replays the full retained fact set. Verify
        // both its existing head and its complete callback sequence before
        // rebuilding derived controls; ordinary producer retries stay read-only.
        let mut exports = BTreeMap::new();
        reading.visit_source(|step| {
            if step.delivery.is_some() {
                if !exports.contains_key(&step.reading_version) {
                    exports.insert(
                        step.reading_version.clone(),
                        (
                            crate::wire_shapes::load_prefix(
                                tx,
                                &input.id,
                                *chain,
                                &step.reading_version,
                            )?,
                            None,
                        ),
                    );
                }
                let (retained, rebuilt) = exports
                    .get_mut(&step.reading_version)
                    .expect("inserted reading");
                crate::wire_shapes::advance_prefix(retained, *length, step)?;
                crate::wire_shapes::advance_prefix(rebuilt, *length, step)?;
            }
            Ok(())
        })?;
        for version in exports.keys() {
            // The tables are derived from the retained callback sequence.
            // Remove its dependent types, shapes and receipts before rebuilding the
            // reading parent.
            for table in ["wire_type", "wire_shape", "wire_receipt"] {
                tx.execute(
                    &format!("DELETE FROM {table} WHERE input_pk=(SELECT pk FROM input_ref WHERE id=?1) AND chain=?2 AND reading_version=?3"),
                    params![input.id,*chain,version],
                )?;
            }
            tx.execute(
                "DELETE FROM wire_reading WHERE input_pk=(SELECT pk FROM input_ref WHERE id=?1) AND chain=?2 AND reading_version=?3",
                params![input.id,*chain,version],
            )?;
        }
        reading.visit(*chain, |step, resolution| {
            rows::write_confirmed_rows(tx, &batch, step, resolution, counter, None, None, changes)?;
            Ok(())
        })?;
        rows::prune_sessions(tx, &input.id, *chain, counter)?;
    }
    Ok(())
}
