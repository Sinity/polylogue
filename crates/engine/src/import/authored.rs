//! Authored records a bundle carried are restored unchanged; their intent is
//! never resolved or run again.
use polylogue_model::bundle::{AuthoredRecord, Record};

pub(super) enum Carried {
    Request(Box<AuthoredRecord>),
    Assertion(Box<polylogue_model::assertion::AssertionRecord>),
}
impl Carried {
    pub(super) fn restore(
        self,
        writer: &polylogue_store::Writer,
    ) -> polylogue_store::Result<polylogue_model::user::UserOutcome> {
        match self {
            Self::Request(record) => writer.restore_authored(record.prepared),
            Self::Assertion(record) => writer.restore_assertion(record.assertion),
        }
    }
}

pub(super) fn replay(
    reader: impl std::io::Read,
    mut apply: impl FnMut(Carried) -> Result<(), String>,
) -> Result<(), String> {
    let mut reader = std::io::BufReader::new(reader);
    polylogue_sources::polylogue_bundle::visit(&mut reader, |_, record, _| {
        match record {
            Record::Authored(record) => apply(Carried::Request(record))?,
            Record::Assertion(record) => apply(Carried::Assertion(record))?,
            _ => {}
        }
        Ok(())
    })
    .map(|_| ())
    .map_err(|error| error.to_string())
}
