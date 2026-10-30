//! Historical canonical authored records are restored without executing intent.
use polylogue_model::bundle::{AuthoredRecord, Record};

pub(super) enum Historical {
    Request(Box<AuthoredRecord>),
    History(Box<polylogue_model::bundle::HistoricalRecord>),
    Assertion(Box<polylogue_model::assertion::AssertionRecord>),
}
impl Historical {
    pub(super) fn restore(
        self,
        writer: &polylogue_store::Writer,
        current: polylogue_model::authored::ImportAttribution,
    ) -> polylogue_store::Result<polylogue_model::user::UserOutcome> {
        match self {
            Self::History(record) => {
                writer.restore_history(record.evidence, record.imports, current)
            }
            Self::Request(record) => {
                writer.restore_authored(record.prepared, record.imports, current)
            }
            Self::Assertion(record) => {
                writer.restore_assertion(record.assertion, record.imports, current)
            }
        }
    }
}

pub(super) fn replay(
    reader: impl std::io::Read,
    mut apply: impl FnMut(Historical) -> Result<(), String>,
) -> Result<(), String> {
    let mut reader = std::io::BufReader::new(reader);
    polylogue_sources::polylogue_bundle::visit(&mut reader, |_, record, _| {
        match record {
            Record::Authored(record) => apply(Historical::Request(record))?,
            Record::Historical(record) => apply(Historical::History(record))?,
            Record::Assertion(record) => apply(Historical::Assertion(record))?,
            _ => {}
        }
        Ok(())
    })
    .map(|_| ())
    .map_err(|error| error.to_string())
}
