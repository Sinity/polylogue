//! Applying prepared confirmed readings uses the ordinary row codecs.
use super::*;

pub(crate) fn apply_confirmed_readings(
    tx: &Transaction<'_>,
    readings: &mut [(InputKey, i64, u64, ChainReading)],
    counter: i64,
) -> Result<()> {
    for (input, chain, length, reading) in readings {
        let batch = Batch {
            input: input.clone(),
            change: Change::Reparse { chain: *chain },
            observation: None,
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length: *length,
            parse: Vec::new(),
            release: Default::default(),
        };
        reading.visit(|step| {
            rows::write_confirmed_rows(tx, &batch, step, *chain, counter, None)?;
            Ok(())
        })?;
        rows::prune_sessions(tx, &input.id, *chain)?;
    }
    Ok(())
}
