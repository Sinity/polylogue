//! A reader's displayed position, addressed by target and computed author.
use crate::{Reader, Result, StoreError, source::Source};
use polylogue_model::{ChangePosition, Reference, annotation::values, user::SeenMark};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::BTreeMap;

fn marks(
    conn: &Connection,
    source: Source,
    author: &str,
    targets: &[String],
) -> Result<Vec<SeenMark>> {
    if targets.is_empty() {
        return Ok(Vec::new());
    }
    let mut statement = conn.prepare_cached(&source.sql(
        "SELECT a.key,a.value FROM @source.annotation a JOIN @source.revision_state r ON r.id=a.id
         WHERE a.kind='seen' AND a.key IN (SELECT value FROM json_each(?1)) AND a.author=?2
           AND r.current=1 AND r.withdrawn=0 ORDER BY a.key,a.id",
    ))?;
    let mut rows = statement.query(params![serde_json::to_string(targets)?, author])?;
    let mut result = Vec::new();
    while let Some(row) = rows.next()? {
        let target: String = row.get(0)?;
        let raw: String = row.get(1)?;
        let value: values::Structured<values::Seen> = serde_json::from_str(&raw)?;
        let at = ChangePosition::parse(&value.fields.position.at)
            .ok_or_else(|| StoreError::stored(1, "invalid seen change position"))?;
        result.push(SeenMark {
            target,
            item: value.fields.position.item,
            at,
        });
    }
    Ok(result)
}

impl Reader<'_> {
    /// Current marks of this author only. Conflicting heads remain visible.
    pub fn seen_marks(&self, author: &str, targets: &[Reference]) -> Result<Vec<SeenMark>> {
        let conn = self.conn()?;
        let result = marks(
            conn,
            Source::reading(conn)?,
            author,
            &targets.iter().map(ToString::to_string).collect::<Vec<_>>(),
        )?;
        self.check()?;
        Ok(result)
    }
}

fn message_item(reference: &Reference) -> Option<(&polylogue_model::MessageId, u64)> {
    match reference {
        Reference::Message(id) => Some((id, 0)),
        Reference::Block { message, n } => Some((message, n.saturating_add(1))),
        _ => None,
    }
}

/// Positions come from the owning relation, never the spelling of an opaque ID.
fn later_item(
    conn: &Connection,
    target: &str,
    previous: Option<&str>,
    next: Option<&str>,
) -> Result<bool> {
    let Some(next) = next else { return Ok(false) };
    let next =
        Reference::parse(next).map_err(|error| StoreError::stored_data(error.to_string()))?;
    let target =
        Reference::parse(target).map_err(|error| StoreError::stored_data(error.to_string()))?;
    if let (Reference::Session(session), Some((next_message, next_block))) =
        (&target, message_item(&next))
    {
        let Some(previous) = previous else {
            crate::read::compare_session_messages(conn, session, next_message, next_message)?;
            return Ok(true);
        };
        let previous = Reference::parse(previous)
            .map_err(|error| StoreError::stored_data(error.to_string()))?;
        let Some((previous_message, previous_block)) = message_item(&previous) else {
            return Ok(false);
        };
        let order =
            crate::read::compare_session_messages(conn, session, previous_message, next_message)?;
        return Ok(order.is_lt() || order.is_eq() && next_block > previous_block);
    }
    if matches!(target, Reference::Session(_)) {
        return Err(polylogue_model::Reason::SeenMarkNeedsItem {}.into());
    }
    let Some(previous) = previous else {
        return Ok(true);
    };
    let previous =
        Reference::parse(previous).map_err(|error| StoreError::stored_data(error.to_string()))?;
    if let (Some((previous, previous_block)), Some((next, next_block))) =
        (message_item(&previous), message_item(&next))
    {
        let coordinate = |id: &polylogue_model::MessageId| -> Result<(i64, Vec<u8>, String)> {
            let resolved = crate::chain_names::resolved_message(conn, id)?;
            conn.query_row(
                "SELECT session_pk,session_order,id FROM message WHERE id=?1 AND duplicate_of IS NULL",
                [resolved.to_string()],
                |row| <(_, _, _)>::try_from(row),
            )
            .optional()?
            .ok_or_else(|| StoreError::NotFound(Reference::Message(id.clone()).to_string()))
        };
        let (previous, next) = (coordinate(previous)?, coordinate(next)?);
        return Ok(previous.0 == next.0
            && (next.1, next.2, next_block) > (previous.1, previous.2, previous_block));
    }
    Ok(match (&previous, &next) {
        (
            Reference::Span {
                container: a,
                unit: au,
                end: ae,
                ..
            },
            Reference::Span {
                container: b,
                unit: bu,
                end: be,
                ..
            },
        ) => a == b && au == bu && be > ae,
        (
            Reference::Result {
                id: a,
                row: Some(ar),
            },
            Reference::Result {
                id: b,
                row: Some(br),
            },
        ) => a == b && br > ar,
        _ => false,
    })
}

fn advances(conn: &Connection, previous: &SeenMark, next: &SeenMark) -> Result<bool> {
    if previous.at.generation != next.at.generation {
        // The caller validated the incoming generation against the active index.
        later_item(conn, &next.target, None, next.item.as_deref())?;
        return Ok(true);
    }
    let before = (previous.at.archive, previous.at.index);
    let after = (next.at.archive, next.at.index);
    if after > before {
        later_item(conn, &next.target, None, next.item.as_deref())?;
        return Ok(true);
    }
    Ok(after == before
        && later_item(
            conn,
            &next.target,
            previous.item.as_deref(),
            next.item.as_deref(),
        )?)
}

/// Accepted marks are monotonic against both stored and earlier marks in this request.
pub(crate) fn advancing(
    authored: &Connection,
    index: &Connection,
    author: &str,
    proposed: &[SeenMark],
) -> Result<Vec<SeenMark>> {
    let generation: String = index.query_row(
        "SELECT value FROM main.meta WHERE key='index_instance'",
        [],
        |row| row.get(0),
    )?;
    let proposed = proposed
        .iter()
        .map(|mark| {
            let target = Reference::parse(&mark.target)
                .map_err(|error| polylogue_model::Reason::BadReference {
                    text: mark.target.clone(),
                    problem: error.to_string(),
                })?
                .to_string();
            let item = mark
                .item
                .as_deref()
                .map(|item| {
                    Reference::parse(item)
                        .map(|item| item.to_string())
                        .map_err(|error| {
                            StoreError::from(polylogue_model::Reason::BadReference {
                                text: item.into(),
                                problem: error.to_string(),
                            })
                        })
                })
                .transpose()?;
            Ok(SeenMark {
                target,
                item,
                at: mark.at.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let targets = proposed
        .iter()
        .map(|mark| mark.target.clone())
        .collect::<Vec<_>>();
    let mut current: BTreeMap<String, Vec<SeenMark>> = BTreeMap::new();
    for mark in marks(authored, Source::of(authored)?, author, &targets)? {
        current.entry(mark.target.clone()).or_default().push(mark);
    }
    let mut accepted = BTreeMap::new();
    for mark in &proposed {
        if mark.at.generation != generation {
            return Err(polylogue_model::Reason::SeenFromAnotherGeneration {}.into());
        }
        if ChangePosition::parse(&mark.at.to_string()).is_none() {
            return Err(StoreError::stored_data("invalid seen change position"));
        }
        if !current.contains_key(&mark.target) {
            later_item(index, &mark.target, None, mark.item.as_deref())?;
        }
        let mut advancing = true;
        for previous in current.get(&mark.target).into_iter().flatten() {
            if !advances(index, previous, mark)? {
                advancing = false;
                break;
            }
        }
        if advancing {
            current.insert(mark.target.clone(), vec![mark.clone()]);
            accepted.insert(mark.target.clone(), mark.clone());
        }
    }
    Ok(accepted.into_values().collect())
}

#[cfg(test)]
mod tests;
