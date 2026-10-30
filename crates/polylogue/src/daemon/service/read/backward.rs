//! Adjacent canonical record coordinates. The caller hydrates exactly the
//! returned record through records::reference and stops after its first record.
use super::super::{Response, error, store_error};
use polylogue_model::{
    Reference,
    read::{ReadRecordKind, ReadRecordLocator},
};
use polylogue_store::Reader;

fn stale() -> Response {
    error("stale_cursor", "the previous record coordinate changed")
}

pub(super) fn last_record(
    reader: &Reader<'_>,
    reference: &Reference,
    inherited: bool,
) -> Result<Option<ReadRecordLocator>, Response> {
    adjacent(reader, reference, inherited, None)
}
pub(super) fn previous_record(
    reader: &Reader<'_>,
    reference: &Reference,
    inherited: bool,
    current: &ReadRecordLocator,
) -> Result<Option<ReadRecordLocator>, Response> {
    adjacent(reader, reference, inherited, Some(current))
}
fn adjacent(
    reader: &Reader<'_>,
    reference: &Reference,
    inherited: bool,
    current: Option<&ReadRecordLocator>,
) -> Result<Option<ReadRecordLocator>, Response> {
    use ReadRecordKind::{Document, Event, Link, LinkIssue, Message, Session};
    reader.check().map_err(|error| store_error(&error))?;
    let mut families = [
        (Session, 0),
        (LinkIssue, 0),
        (Message, 0),
        (Event, 0),
        (Link, 0),
    ];
    let length = match reference {
        Reference::Session(id) => {
            let Some(physical) = reader
                .physical_record_counts(id)
                .map_err(|error| store_error(&error))?
            else {
                return if current.is_some() {
                    Err(stale())
                } else {
                    Ok(None)
                };
            };
            let selected = reader
                .select_messages(id, inherited)
                .map_err(|error| store_error(&error))?;
            let issues = if inherited {
                u64::try_from(selected.parents().issues.len()).map_err(|_| stale())?
            } else {
                0
            };
            let messages = selected.len().map_err(|error| store_error(&error))?;
            let messages = u64::try_from(messages).map_err(|_| stale())?;
            families = [
                (Session, 1),
                (LinkIssue, issues),
                (Message, messages),
                (Event, physical.events),
                (Link, physical.links),
            ];
            5
        }
        Reference::Message(_) | Reference::Block { .. } => {
            families[0] = (Message, 1);
            1
        }
        Reference::Document(_) => {
            families[0] = (Document, 1);
            1
        }
        Reference::Event(_) => {
            families[0] = (Event, 1);
            1
        }
        _ => {
            return Err(error(
                "not_built",
                format!("the selected reference {reference} has no canonical read adapter"),
            ));
        }
    };
    let families = &families[..length];
    let before = if let Some(current) = current {
        let family = families
            .iter()
            .position(|(kind, _)| *kind == current.kind)
            .ok_or_else(stale)?;
        if current.ordinal >= families[family].1 {
            return Err(stale());
        }
        if current.ordinal != 0 {
            return Ok(Some(ReadRecordLocator {
                kind: current.kind,
                ordinal: current.ordinal - 1,
            }));
        }
        family
    } else {
        families.len()
    };
    Ok(families[..before]
        .iter()
        .rev()
        .find(|(_, count)| *count != 0)
        .map(|(kind, count)| ReadRecordLocator {
            kind: *kind,
            ordinal: count - 1,
        }))
}

#[cfg(test)]
#[path = "backward/tests.rs"]
mod tests;
