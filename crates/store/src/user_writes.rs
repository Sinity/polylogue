//! Writes people and agents make to `archive.db`: notes and their
//! withdrawals, context deliveries and settings. They go through the one
//! writer thread like everything else, each in its own small transaction,
//! so a note never waits behind a first build for longer than one group.
//!
//! Every write carries a client request ID. The same ID with the same
//! intent writes nothing and returns what the first write created; the
//! same ID with other intent is refused as a conflict.

use crate::{Result, StoreError};
use rusqlite::{Transaction, params};

use polylogue_model::user::{UserOutcome, UserRequest, UserWrite};

/// Product catalog captured by trusted acceptance, outside the serialized
/// request and its retry identity. Saved/kept bindings come from the snapshot.
/// Authored-origin and execution eligibility remain separate from this catalog.
#[derive(Debug, Clone)]
pub struct PreparationContext {
    bindings: polylogue_query::Bindings,
}
impl PreparationContext {
    pub fn new(origins: Vec<String>) -> Self {
        Self {
            bindings: polylogue_query::Bindings {
                origins,
                ..Default::default()
            },
        }
    }
    pub(crate) fn bindings(&self) -> &polylogue_query::Bindings {
        &self.bindings
    }
}

pub(crate) struct Preparation<'a> {
    pub context: Option<&'a PreparationContext>,
    pub pending: Option<&'a dyn crate::primary::Sinex>,
    pub provisional: Option<&'a mut crate::selection::ProvisionalQueries>,
    pub owner: Option<polylogue_model::settings::Owner>,
}
impl<'a> Preparation<'a> {
    pub(crate) fn native(
        context: Option<&'a PreparationContext>,
        pending: Option<&'a dyn crate::primary::Sinex>,
    ) -> Self {
        Self {
            context,
            pending,
            provisional: None,
            owner: None,
        }
    }
    pub(crate) fn preflight(
        context: Option<&'a PreparationContext>,
        pending: Option<&'a dyn crate::primary::Sinex>,
        owner: polylogue_model::settings::Owner,
        provisional: &'a mut crate::selection::ProvisionalQueries,
    ) -> Self {
        Self {
            context,
            pending,
            provisional: Some(provisional),
            owner: Some(owner),
        }
    }
}

/// A new UUIDv7 for a request.
pub fn new_request_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Applies one request in `tx`, which the writer commits. `seq` is the
/// archive's write counter for this transaction.
#[cfg(test)]
pub(crate) fn apply(tx: &Transaction<'_>, r: &UserRequest, seq: i64) -> Result<UserOutcome> {
    apply_resolving(tx, tx, r, seq, Preparation::native(None, None))
}

pub(crate) fn apply_resolving(
    tx: &Transaction<'_>,
    original: &rusqlite::Connection,
    r: &UserRequest,
    seq: i64,
    preparation: Preparation<'_>,
) -> Result<UserOutcome> {
    let source = crate::source::Source::of(tx)?;
    apply_using(
        tx,
        original,
        r,
        seq,
        &|attachment| {
            Ok(tx
                .prepare_cached(
                    &source.sql("SELECT 1 FROM @source.blob WHERE hash=?1 AND size=?2"),
                )?
                .exists(params![
                    &attachment.blob.0[..],
                    i64::try_from(attachment.size).map_err(|_| StoreError::Invalid(
                        "attachment size is out of range".into()
                    ))?
                ])?)
        },
        preparation,
    )
}

/// These deliveries already carry their complete body. They need no authored
/// snapshot, target selection or post-write readback to freeze their effect.
pub(crate) fn prepare_delivery(
    r: &UserRequest,
) -> Option<polylogue_model::authored::PreparedRequest> {
    use polylogue_model::authored::{
        AuthoredEffect, DeliveryChange, DeliveryEvent, DeliveryRecord, PreparedRequest,
    };
    let delivery = match &r.write {
        UserWrite::Deliver(delivery) | UserWrite::DeliveredAdvice(delivery) => delivery,
        _ => return None,
    };
    // Delivery rows and their receipts have always stored Unix milliseconds;
    // the original request retains its full timestamp and retry identity.
    let at = polylogue_model::Timestamp::from_unix_ms(r.at.unix_ms())
        .expect("a stored millisecond timestamp remains representable");
    let events = if matches!(r.write, UserWrite::DeliveredAdvice(_)) {
        vec![DeliveryEvent {
            request: r.id.clone(),
            what: polylogue_model::DeliveryOutcome::Delivered,
            at,
            detail: Some("written to the harness hook output".into()),
        }]
    } else {
        Vec::new()
    };
    Some(PreparedRequest {
        request: r.clone(),
        outcome: UserOutcome {
            pending: false,
            id: r.id.clone(),
            repeat: false,
        },
        effect: AuthoredEffect {
            deliveries: vec![DeliveryChange {
                delivery: DeliveryRecord {
                    signal: None,
                    id: r.id.clone(),
                    prepared: delivery.clone(),
                    at,
                    events: events.clone(),
                },
                added_events: events,
            }],
            ..AuthoredEffect::default()
        },
    })
}

pub(crate) fn apply_using(
    tx: &Transaction<'_>,
    original: &rusqlite::Connection,
    r: &UserRequest,
    seq: i64,
    available: &dyn Fn(&polylogue_model::user::ChatAttachment) -> Result<bool>,
    preparation: Preparation<'_>,
) -> Result<UserOutcome> {
    let source = crate::source::Source::of(tx)?;
    if let Some(known) = crate::primary::authored::known_request(tx, r)? {
        let mut outcome = known.outcome;
        outcome.repeat = true;
        return Ok(outcome);
    }
    if let Some(prepared) = prepare_delivery(r) {
        crate::primary::authored::require_new_delivery(tx, &r.id)?;
        crate::primary::authored::apply(tx, &prepared, seq)?;
        return Ok(prepared.outcome);
    }
    let hash = crate::primary::authored::request_hash(r)?;
    tx.execute(
        &source.sql(
            "INSERT INTO @source.client_request(id, content_hash, created_at, request, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        ),
        params![
            r.id,
            &hash[..],
            r.at.unix_ms(),
            crate::encoding::pack_text(&serde_json::to_string(r)?)?,
            seq
        ],
    )?;
    let at = r.at.unix_ms();
    let owner = preparation.owner.unwrap_or({
        if matches!(source, crate::source::Source::Cache) {
            polylogue_model::settings::Owner::Primary
        } else {
            polylogue_model::settings::Owner::Standalone
        }
    });
    let hash = |reference: &polylogue_model::Reference| match reference {
        polylogue_model::Reference::Session(_)
        | polylogue_model::Reference::Message(_)
        | polylogue_model::Reference::Block { .. }
        | polylogue_model::Reference::Document(_)
        | polylogue_model::Reference::MessageRange { .. }
        | polylogue_model::Reference::Span { .. }
        | polylogue_model::Reference::Assertion(_)
        | polylogue_model::Reference::Variant { .. } => Ok(
            match crate::reference::link_hash(original, tx, reference)? {
                crate::reference::LinkHash::Text(hash) => Some(hash),
                crate::reference::LinkHash::Missing | crate::reference::LinkHash::NotApplicable => {
                    None
                }
            },
        ),
        _ => Ok(None),
    };
    let quote = |reference: &polylogue_model::Reference| {
        crate::reference::link_quote(original, tx, reference)
    };
    let mut saved = crate::assertions::SavedPreparation::new(original, tx, r, preparation);
    let mut resolve_saved = |input: crate::assertions::SavedInput<'_>| saved.resolve(input);
    let context = crate::assertions::WriteContext {
        request: r,
        counter: seq,
        hash: &hash,
        quote: &quote,
        owner,
    };
    let outcome = match &r.write {
        UserWrite::Assert { assertions } => {
            crate::assertions::write_preparing(tx, assertions, context, &mut resolve_saved)
        }
        UserWrite::AssertQuery {
            assertion,
            resolved,
            ..
        } => {
            use polylogue_model::assertion::{LinkRole, NewAssertionLink};
            if resolved.is_empty()
                || assertion
                    .links
                    .iter()
                    .any(|link| link.role == LinkRole::About)
            {
                return Err(StoreError::Invalid("a query assertion needs resolved targets and cannot supply its own about links".into()));
            }
            // One deliberate write of a single-valued mark revises it; a bulk
            // write over several rows never does. Systematic labelling goes
            // through a rubric field, whose slot includes its author.
            if assertion.kind.spec().single_valued_mark() && resolved.len() > 1 {
                return Err(StoreError::Invalid(format!(
                    "single_valued_bulk: `{}` holds one value per target and is not written over {} rows; label rows with a rubric field (annotation) instead",
                    assertion.kind.as_str(),
                    resolved.len()
                )));
            }
            let mut targets = std::collections::BTreeSet::new();
            let mut statements = Vec::with_capacity(resolved.len());
            for reference in resolved {
                if !targets.insert(reference) {
                    return Err(StoreError::Invalid(
                        "a query assertion contains duplicate resolved targets".into(),
                    ));
                }
                let mut statement = assertion.clone();
                statement.links.push(NewAssertionLink {
                    role: LinkRole::About,
                    reference: reference.clone(),
                });
                statements.push(statement);
            }
            crate::assertions::write_preparing(tx, &statements, context, &mut resolve_saved)
        }
        UserWrite::ReviseAssertions { bases, patch } => {
            crate::assertions::revise_preparing(tx, bases, patch, context, &mut resolve_saved)
        }
        UserWrite::WithdrawAssertions { targets, reason } => {
            use polylogue_model::assertion::{
                AssertionKind, LinkRole, NewAssertion, NewAssertionLink,
            };
            let statement = NewAssertion {
                kind: AssertionKind::Withdrawal,
                key: None,
                value: reason
                    .as_ref()
                    .map(|reason| serde_json::Value::String(reason.clone())),
                valid_until: None,
                links: targets
                    .iter()
                    .map(|reference| NewAssertionLink {
                        role: LinkRole::Withdraws,
                        reference: reference.clone(),
                    })
                    .collect(),
            };
            crate::assertions::write_preparing(tx, &[statement], context, &mut resolve_saved)
        }
        UserWrite::SignalCommit(commit) => crate::signals::write(tx, r, commit, seq),
        UserWrite::Variant { variant } => crate::variants::write(tx, r, variant, seq),
        UserWrite::SaveQuery {
            name,
            text,
            denominator,
        } => {
            if !polylogue_model::assertion::policy::sayable_query_name(name) {
                return Err(StoreError::Invalid(format!(
                    "{name:?} cannot follow `query:`; use letters, digits, `_`, `-` and `.`"
                )));
            }
            let value = resolve_saved(crate::assertions::SavedInput::Native {
                text,
                denominator: denominator.as_deref(),
            })?;
            let statement = polylogue_model::assertion::NewAssertion {
                kind: polylogue_model::assertion::AssertionKind::SavedQuery,
                key: Some(name.clone()),
                value: Some(serde_json::to_value(value)?),
                valid_until: None,
                links: Vec::new(),
            };
            crate::assertions::write_preparing(tx, &[statement], context, &mut resolve_saved)
        }
        UserWrite::Deliver(_) | UserWrite::DeliveredAdvice(_) => {
            unreachable!("literal deliveries are applied before resolving other writes")
        }
        UserWrite::DeliveryEvent {
            delivery,
            what,
            detail,
        } => {
            let known: bool = tx
                .prepare_cached(&source.sql("SELECT 1 FROM @source.delivery WHERE id = ?1"))?
                .exists([delivery])?;
            if !known {
                return Err(StoreError::NotFound(format!("delivery {delivery}")));
            }
            tx.execute(
                &source.sql("INSERT OR IGNORE INTO @source.delivery_event(delivery, what, at, detail, request, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)"),
                params![delivery, what.as_str(), at, detail, r.id, seq],
            )?;
            Ok(UserOutcome {
                pending: false,
                id: delivery.clone(),
                repeat: false,
            })
        }
        UserWrite::KeepResult {
            name,
            query,
            rows_json,
            read_at,
        } => {
            crate::primary::authored::insert_kept(
                tx,
                &polylogue_model::authored::KeptResult {
                    id: r.id.clone(),
                    name: name.clone(),
                    query: query.clone(),
                    ran_at: r.at,
                    read_at: read_at.clone(),
                    rows_json: rows_json.clone(),
                },
            )?;
            Ok(UserOutcome {
                pending: false,
                id: r.id.clone(),
                repeat: false,
            })
        }
        UserWrite::ChatEnqueue(new) => {
            let id = crate::chat_actions::enqueue(tx, new, at, seq, &r.id, available)?;
            Ok(UserOutcome {
                pending: false,
                repeat: id != new.id,
                id,
            })
        }
        UserWrite::ChatClaim { by } => Ok(UserOutcome {
            pending: false,
            id: crate::chat_actions::claim(tx, by, at, seq, &r.id)?,
            repeat: false,
        }),
        UserWrite::ChatReport {
            action,
            by,
            outcome,
            phase,
            detail,
            receipt,
        } => {
            crate::chat_actions::report(
                tx,
                action,
                by,
                outcome,
                phase,
                detail.as_deref(),
                receipt.as_ref(),
                at,
                seq,
                &r.id,
            )?;
            Ok(UserOutcome {
                pending: false,
                id: action.clone(),
                repeat: false,
            })
        }
        UserWrite::ChatDecide {
            action,
            approve,
            by,
            detail,
        } => {
            crate::chat_actions::decide(tx, action, *approve, by, detail.as_deref(), seq, r)?;
            Ok(UserOutcome {
                pending: false,
                id: action.clone(),
                repeat: false,
            })
        }
        UserWrite::BackfillStart(new) => {
            let id = crate::backfill_jobs::start(tx, new, at, seq, &r.id)?;
            Ok(UserOutcome {
                pending: false,
                repeat: id != new.id,
                id,
            })
        }
        UserWrite::BackfillClaim { by } => Ok(UserOutcome {
            pending: false,
            id: crate::backfill_jobs::claim(tx, by, at, seq, &r.id)?,
            repeat: false,
        }),
        UserWrite::BackfillReport { job, by, report } => {
            crate::backfill_jobs::report(tx, job, by, report, at, seq, &r.id)?;
            Ok(UserOutcome {
                pending: false,
                id: job.clone(),
                repeat: false,
            })
        }
        UserWrite::BackfillCancel { job } => {
            crate::backfill_jobs::cancel(tx, job, at, seq, &r.id)?;
            Ok(UserOutcome {
                pending: false,
                id: job.clone(),
                repeat: false,
            })
        }
        UserWrite::SetSetting {
            key,
            value,
            machine,
        } => {
            let value = polylogue_model::settings::parse(key, value, owner)
                .map_err(|error| StoreError::Invalid(error.to_string()))?;
            let statement = polylogue_model::assertion::NewAssertion {
                kind: polylogue_model::assertion::AssertionKind::Setting,
                key: Some(key.clone()),
                value: Some(serde_json::to_value(
                    polylogue_model::assertion::values::Setting {
                        machine: machine.clone(),
                        value,
                    },
                )?),
                valid_until: None,
                links: Vec::new(),
            };
            crate::assertions::write_preparing(tx, &[statement], context, &mut resolve_saved)
        }
    }?;
    let effect = crate::primary::authored::capture(tx, r, seq)?;
    tx.execute(
        &source.sql("UPDATE @source.client_request SET outcome=?2,effect=?3 WHERE id=?1"),
        params![
            r.id,
            crate::primary::authored::pack(&outcome)?,
            crate::primary::authored::pack(&effect)?
        ],
    )?;
    Ok(outcome)
}

#[cfg(test)]
mod tests;
