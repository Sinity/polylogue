//! Historical canonical authored records are restored without executing intent.
use polylogue_model::bundle::{AuthoredRecord, Record};

pub(super) fn replay(
    reader: impl std::io::Read,
    mut apply: impl FnMut(AuthoredRecord) -> Result<(), String>,
) -> Result<(), String> {
    let mut reader = std::io::BufReader::new(reader);
    polylogue_sources::polylogue_bundle::visit(&mut reader, |_, record, _| {
        if let Record::Authored(record) = record {
            apply(*record)?;
        }
        Ok(())
    })
    .map(|_| ())
    .map_err(|error| error.to_string())
}
