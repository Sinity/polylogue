//! Adjacent canonical record coordinates. The caller hydrates exactly the
//! returned record through records::reference and stops after its first record.
use super::super::Failure;
use polylogue_model::read::{ReadRecordKind, ReadRecordLocator, ReadTarget};
use polylogue_store::Reader;

fn stale() -> Failure {
    Failure::new("stale_cursor", "the previous record coordinate changed")
}

pub(super) fn last_record(
    reader: &Reader<'_>,
    target: &ReadTarget,
    inherited: bool,
) -> Result<Option<ReadRecordLocator>, Failure> {
    adjacent(reader, target, inherited, None)
}
pub(super) fn previous_record(
    reader: &Reader<'_>,
    target: &ReadTarget,
    inherited: bool,
    current: &ReadRecordLocator,
) -> Result<Option<ReadRecordLocator>, Failure> {
    adjacent(reader, target, inherited, Some(current))
}
fn adjacent(
    reader: &Reader<'_>,
    target: &ReadTarget,
    inherited: bool,
    current: Option<&ReadRecordLocator>,
) -> Result<Option<ReadRecordLocator>, Failure> {
    use ReadRecordKind::{Authored, Document, Event, Link, LinkIssue, Message, Session};
    reader.check()?;
    let mut families = [
        (Session, 0),
        (LinkIssue, 0),
        (Message, 0),
        (Event, 0),
        (Link, 0),
    ];
    let length = match target {
        ReadTarget::Session(id) => {
            let Some(physical) = reader.physical_record_counts(id)? else {
                return if current.is_some() {
                    Err(stale())
                } else {
                    Ok(None)
                };
            };
            let selected = reader.select_messages(id, inherited)?;
            let issues = if inherited {
                u64::try_from(selected.parents().issues.len()).map_err(|_| stale())?
            } else {
                0
            };
            let messages = selected.len()?;
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
        ReadTarget::Message(_) | ReadTarget::Block { .. } => {
            families[0] = (Message, 1);
            1
        }
        ReadTarget::MessageRange { first, last } => {
            let mut count = 0_u64;
            reader.visit_message_range(first, last, 0, |_| {
                count += 1;
                Ok(std::ops::ControlFlow::Continue(()))
            })?;
            families[0] = (Message, count);
            1
        }
        ReadTarget::Document(_) => {
            families[0] = (Document, 1);
            1
        }
        ReadTarget::Event(_) => {
            families[0] = (Event, 1);
            1
        }
        ReadTarget::Assertion { .. } | ReadTarget::Delivery(_) => {
            families[0] = (Authored, 1);
            1
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
