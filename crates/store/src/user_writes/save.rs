//! One accepted Save: immutable rows and any changed question and watch.
use super::*;
use crate::assertions::{SavedInput, SavedResolver, WriteContext};
use polylogue_model::user::{KeptPart, SaveReport, SavedPart, WatchPart};
use polylogue_model::{
    Reference, RevisionReference,
    assertion::{Assertion, AssertionKind, LinkRole, NewAssertion, NewAssertionLink, values},
};
use rusqlite::Connection;

fn current(conn: &Connection, kind: AssertionKind, name: &str) -> Result<Option<Assertion>> {
    let rows = crate::assertions::named_current(conn, kind, name)?;
    if rows.len() > 1 {
        return Err(StoreError::Ambiguous {
            reference: format!("{}:{name}", kind.as_str()),
            candidates: rows
                .iter()
                .map(|row| {
                    Reference::Assertion(RevisionReference {
                        root: row.fact.root.clone(),
                        revision: Some(row.fact.id.clone()),
                    })
                    .to_string()
                })
                .collect(),
        });
    }
    Ok(rows.into_iter().next())
}

fn statement(
    kind: AssertionKind,
    name: &str,
    value: serde_json::Value,
    previous: Option<&Assertion>,
) -> NewAssertion {
    NewAssertion {
        kind,
        key: Some(name.into()),
        value: Some(value),
        valid_until: None,
        links: previous
            .into_iter()
            .map(|row| NewAssertionLink {
                role: LinkRole::Revises,
                reference: Reference::Assertion(RevisionReference {
                    root: row.fact.root.clone(),
                    revision: Some(row.fact.id.clone()),
                }),
            })
            .collect(),
    }
}

/// Withdrawing a current named question also withdraws its current watch.
pub(super) fn withdrawal_targets(
    conn: &Connection,
    targets: &[Reference],
) -> Result<Vec<Reference>> {
    let mut all = targets.to_vec();
    let mut known = targets
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    for target in targets {
        let Reference::Assertion(wanted) = target else {
            continue;
        };
        for head in crate::assertions::heads(
            conn,
            &Reference::Assertion(RevisionReference {
                root: wanted.root.clone(),
                revision: None,
            }),
        )? {
            let Reference::Assertion(exact) = head else {
                continue;
            };
            if wanted.revision.is_some() && wanted.revision != exact.revision {
                continue;
            }
            let Some(id) = &exact.revision else {
                continue;
            };
            let Some(saved) = crate::assertions::get(conn, id)? else {
                continue;
            };
            if saved.fact.kind != AssertionKind::SavedQuery {
                continue;
            }
            let Some(name) = saved.fact.key else {
                continue;
            };
            for watch in crate::assertions::named_current(conn, AssertionKind::Signal, &name)? {
                let reference = Reference::Assertion(RevisionReference {
                    root: watch.fact.root,
                    revision: wanted.revision.as_ref().map(|_| watch.fact.id),
                });
                if known.insert(reference.clone()) {
                    all.push(reference);
                }
            }
        }
    }
    Ok(all)
}

pub(super) fn apply(
    conn: &Connection,
    save: &polylogue_model::user::Save,
    context: WriteContext<'_>,
    resolve: &mut SavedResolver<'_>,
) -> Result<SaveReport> {
    let request = context.request;
    if save.name.is_none() && save.watch.is_some() {
        return Err(StoreError::invalid("a watch needs a saved query name"));
    }
    if save.name.is_none() && save.denominator.is_some() {
        return Err(StoreError::invalid("a figure is a named question"));
    }
    if let Some(watch) = &save.watch
        && let Some(hold) = &watch.hold
        && query_language::time::duration_ms(hold).is_none_or(|duration| duration < 0)
    {
        return Err(StoreError::invalid("a watch needs a duration such as 20m"));
    }
    crate::kept::validate_body(&request.id, &save.rows_json)?;
    crate::selection::check_kept_body(conn, &request.id, &save.rows_json)?;
    let body: serde_json::Value = serde_json::from_str(&save.rows_json)?;
    let rows = body["rows"]
        .as_array()
        .ok_or_else(|| StoreError::invalid("kept result has no rows"))?
        .len() as u64;
    let mut statements = Vec::new();
    let mut saved_part = None;
    let mut watch_part = None;
    let mut saved_position = None;
    let mut watch_position = None;
    if let Some(name) = &save.name {
        if !polylogue_model::assertion::policy::sayable_query_name(name) {
            return Err(StoreError::invalid(format!(
                "{name:?} is not a saved query name"
            )));
        }
        let previous = current(conn, AssertionKind::SavedQuery, name)?;
        let previous_value = previous
            .as_ref()
            .map(|row| {
                serde_json::from_value::<values::SavedQuery>(
                    row.fact
                        .value
                        .clone()
                        .ok_or_else(|| StoreError::invalid("saved question has no value"))?,
                )
                .map_err(StoreError::from)
            })
            .transpose()?;
        // An own-name Save keeps the accepted original text. Reading its stored
        // object validates the old hash without parsing it as a new question.
        let value = match &previous_value {
            Some(previous) if previous.text == save.query && save.denominator.is_none() => {
                resolve(SavedInput::Stored {
                    value: previous,
                    changed: false,
                })?
            }
            _ => resolve(SavedInput::Text {
                text: &save.query,
                denominator: save.denominator.as_deref(),
            })?,
        };
        let changed = previous_value
            .as_ref()
            .is_none_or(|old| old.hash != value.hash || old.denominator != value.denominator);
        let old_revision = previous
            .as_ref()
            .map(|row| {
                crate::assertions::saved_revision_number(
                    conn,
                    &RevisionReference {
                        root: row.fact.root.clone(),
                        revision: Some(row.fact.id.clone()),
                    },
                )
            })
            .transpose()?
            .unwrap_or(0);
        let revision = if changed {
            old_revision
                .checked_add(1)
                .ok_or_else(|| StoreError::invalid("saved revision number is out of range"))?
        } else {
            old_revision
        };
        if changed {
            saved_position = Some(statements.len());
            statements.push(statement(
                AssertionKind::SavedQuery,
                name,
                serde_json::to_value(value)?,
                previous.as_ref(),
            ));
        }
        saved_part = Some(SavedPart {
            id: previous
                .as_ref()
                .map(|row| row.fact.id.clone())
                .unwrap_or_default(),
            name: name.clone(),
            revision,
            new_revision: false,
        });
        let previous_watch = current(conn, AssertionKind::Signal, name)?;
        if save.watch.is_some() || (changed && previous_watch.is_some()) {
            let mut rule = previous_watch
                .as_ref()
                .and_then(|row| row.fact.value.clone())
                .unwrap_or_else(|| {
                    serde_json::json!({
                        "name":name,"fires":"on_new","deliver":"notice","interrupt":false,
                    })
                });
            let object = rule
                .as_object_mut()
                .ok_or_else(|| StoreError::invalid("saved watch rule is not an object"))?;
            object.insert("query".into(), format!("query:{name}@{revision}").into());
            if let Some(watch) = &save.watch {
                if let Some(hold) = &watch.hold {
                    object.insert("for".into(), hold.clone().into());
                }
                if watch.notify {
                    object.insert("interrupt".into(), true.into());
                }
            }
            if previous_watch
                .as_ref()
                .and_then(|row| row.fact.value.as_ref())
                != Some(&rule)
            {
                watch_position = Some(statements.len());
                statements.push(statement(
                    AssertionKind::Signal,
                    name,
                    rule,
                    previous_watch.as_ref(),
                ));
            }
        }
        if let Some(watch) = &previous_watch {
            watch_part = Some(WatchPart {
                id: watch.fact.id.clone(),
                name: name.clone(),
                new_rule: false,
            });
        } else if save.watch.is_some() {
            watch_part = Some(WatchPart {
                id: String::new(),
                name: name.clone(),
                new_rule: false,
            });
        }
    }
    if !statements.is_empty() {
        let parts = crate::assertions::write_parts_preparing(conn, &statements, context, resolve)?;
        if parts.len() != statements.len() {
            return Err(StoreError::invalid(
                "Save did not accept every requested part",
            ));
        }
        if let Some(index) = saved_position {
            let part = saved_part.as_mut().expect("named Save");
            part.id = parts[index].id.clone();
            part.new_revision = !parts[index].repeat;
        }
        if let Some(index) = watch_position {
            let part = watch_part
                .as_mut()
                .expect("watch requested or already present");
            part.id = parts[index].id.clone();
            part.new_rule = !parts[index].repeat;
        }
    }
    crate::primary::authored::insert_kept(
        conn,
        &polylogue_model::authored::KeptResult {
            id: request.id.clone(),
            name: save.name.clone(),
            query: save.query.clone(),
            ran_at: request.at,
            read_at: save.read_at.clone(),
            rows_json: save.rows_json.clone(),
        },
    )?;
    Ok(SaveReport {
        request: request.id.clone(),
        saved: saved_part,
        kept: KeptPart {
            result: format!("result:{}", request.id),
            rows,
            at: request.at,
        },
        watch: watch_part,
        repeat: false,
        pending: false,
    })
}
