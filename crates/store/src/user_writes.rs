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

/// Runtime completion, distinct from the immutable stored outcome schema.
#[derive(Debug)]
pub struct UserReceipt {
    pub outcome: UserOutcome,
    pub save: Option<polylogue_model::user::SaveReport>,
}
impl UserReceipt {
    pub(crate) fn recorded(
        prepared: &polylogue_model::authored::PreparedRequest,
        outcome: UserOutcome,
    ) -> Result<Self> {
        let save =
            match &prepared.request.write {
                UserWrite::SaveSelection(save) => Some(save.outcomes.clone().ok_or_else(|| {
                    StoreError::invalid("accepted Save has no resolved outcomes")
                })?),
                _ => None,
            };
        Ok(Self { outcome, save })
    }
}

/// Product catalog captured by trusted acceptance, outside the serialized
/// request and its retry identity. Saved/kept bindings come from the snapshot.
/// Who authored a request and whether its actions may run on this host stay
/// separate from this catalog.
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
    pub mode: Option<polylogue_model::settings::SettingMode>,
}
impl<'a> Preparation<'a> {
    pub(crate) fn plain(
        context: Option<&'a PreparationContext>,
        pending: Option<&'a dyn crate::primary::Sinex>,
    ) -> Self {
        Self {
            context,
            pending,
            provisional: None,
            mode: None,
        }
    }
    pub(crate) fn preflight(
        context: Option<&'a PreparationContext>,
        pending: Option<&'a dyn crate::primary::Sinex>,
        mode: polylogue_model::settings::SettingMode,
        provisional: &'a mut crate::selection::ProvisionalQueries,
    ) -> Self {
        Self {
            context,
            pending,
            provisional: Some(provisional),
            mode: Some(mode),
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
    apply_resolving(tx, tx, r, seq, Preparation::plain(None, None)).map(|receipt| receipt.outcome)
}

pub(crate) fn apply_resolving(
    tx: &Transaction<'_>,
    original: &rusqlite::Connection,
    r: &UserRequest,
    seq: i64,
    preparation: Preparation<'_>,
) -> Result<UserReceipt> {
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
                    crate::to_i64(attachment.size, "attachment size is out of range")?
                ])?)
        },
        preparation,
    )
}

/// These deliveries already carry their complete body. They need no authored
/// snapshot, target selection or post-write readback to fix their effect.
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
    // the original request keeps its full timestamp and retry identity.
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
) -> Result<UserReceipt> {
    let source = crate::source::Source::of(tx)?;
    if let Some(known) = crate::primary::authored::known_request(tx, r)? {
        let mut outcome = known.outcome.clone();
        outcome.repeat = true;
        return UserReceipt::recorded(&known, outcome);
    }
    if let Some(prepared) = prepare_delivery(r) {
        crate::primary::authored::require_new_delivery(tx, &r.id)?;
        crate::primary::authored::apply(tx, &prepared, seq)?;
        return Ok(UserReceipt {
            outcome: prepared.outcome,
            save: None,
        });
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
    let mode = preparation.mode.unwrap_or({
        if matches!(source, crate::source::Source::Cache) {
            polylogue_model::settings::SettingMode::Primary
        } else {
            polylogue_model::settings::SettingMode::Standalone
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
        mode,
    };
    let mut save_outcomes = None;
    let outcome = match &r.write {
        UserWrite::SaveSelection(save) => {
            use polylogue_model::assertion::{AssertionKind, NewAssertion};
            if save.watch.is_some() && save.name.is_none() {
                return Err(StoreError::invalid("a watch needs a saved query name"));
            }
            if let Some(watch) = &save.watch
                && let Some(hold) = &watch.hold
                && query_language::time::duration_ms(hold).is_none_or(|value| value < 0)
            {
                return Err(StoreError::invalid("a watch needs a duration such as 20m"));
            }
            let identities = save.identities(&r.id);
            crate::kept::validate_body(&identities.kept, &save.rows_json)?;
            crate::selection::check_kept_body(tx, &identities.kept, &save.rows_json)?;
            let saved = if let Some(name) = &save.name {
                if !polylogue_model::assertion::policy::sayable_query_name(name) {
                    return Err(StoreError::invalid(format!(
                        "{name:?} cannot follow `query:`"
                    )));
                }
                let value = resolve_saved(crate::assertions::SavedInput::Text {
                    text: &save.query,
                    denominator: save.denominator.as_deref(),
                })?;
                let saved_request = UserRequest {
                    id: identities.saved.clone().expect("named Save identity"),
                    machine: r.machine.clone(),
                    authorship: r.authorship.clone(),
                    at: r.at,
                    write: UserWrite::SaveQuery {
                        name: name.clone(),
                        text: save.query.clone(),
                        denominator: save.denominator.clone(),
                    },
                };
                let statement = NewAssertion {
                    kind: AssertionKind::SavedQuery,
                    key: Some(name.clone()),
                    value: Some(serde_json::to_value(value)?),
                    valid_until: None,
                    links: Vec::new(),
                };
                Some(crate::assertions::write_preparing(
                    tx,
                    &[statement],
                    crate::assertions::WriteContext {
                        request: &saved_request,
                        counter: seq,
                        hash: &hash,
                        quote: &quote,
                        mode,
                    },
                    &mut resolve_saved,
                )?)
            } else {
                None
            };
            crate::primary::authored::insert_kept(
                tx,
                &polylogue_model::authored::KeptResult {
                    id: identities.kept.clone(),
                    name: save.name.clone(),
                    query: save.query.clone(),
                    ran_at: r.at,
                    read_at: save.read_at.clone(),
                    rows_json: save.rows_json.clone(),
                },
            )?;
            let watch = if let Some(watch) = &save.watch {
                let mut rule = serde_json::json!({"name":save.name,"query":save.query,"fires":"on_new","deliver":"notice","interrupt":watch.notify});
                if let Some(hold) = &watch.hold {
                    rule["for"] = serde_json::json!(hold);
                }
                let assertion = NewAssertion {
                    kind: AssertionKind::Signal,
                    key: save.name.clone(),
                    value: Some(rule),
                    valid_until: None,
                    links: Vec::new(),
                };
                let watched_request = UserRequest {
                    id: identities.watch.expect("watch identity"),
                    machine: r.machine.clone(),
                    authorship: r.authorship.clone(),
                    at: r.at,
                    write: UserWrite::Assert {
                        assertions: vec![assertion.clone()],
                    },
                };
                Some(crate::assertions::write_preparing(
                    tx,
                    &[assertion],
                    crate::assertions::WriteContext {
                        request: &watched_request,
                        counter: seq,
                        hash: &hash,
                        quote: &quote,
                        mode,
                    },
                    &mut resolve_saved,
                )?)
            } else {
                None
            };
            save_outcomes = Some(polylogue_model::user::SaveReport {
                saved,
                kept: UserOutcome {
                    id: identities.kept.clone(),
                    repeat: false,
                    pending: false,
                },
                watch,
            });
            Ok(UserOutcome {
                id: r.id.clone(),
                repeat: false,
                pending: false,
            })
        }
        UserWrite::ResolveLine { .. } => Ok(UserOutcome {
            id: r.id.clone(),
            repeat: false,
            pending: false,
        }),
        UserWrite::Assert { assertions } => {
            crate::assertions::write_preparing(tx, assertions, context, &mut resolve_saved)
        }
        UserWrite::AssertQuery {
            assertion,
            resolved,
            ..
        } => {
            use polylogue_model::assertion::{LinkRole, NewAssertionLink};
            if assertion
                .links
                .iter()
                .any(|link| link.role == LinkRole::About)
            {
                return Err(StoreError::Invalid(
                    "a query assertion cannot supply its own about links".into(),
                ));
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
            if statements.is_empty() {
                Ok(UserOutcome {
                    id: r.id.clone(),
                    repeat: false,
                    pending: false,
                })
            } else {
                crate::assertions::write_preparing(tx, &statements, context, &mut resolve_saved)
            }
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
            let value = resolve_saved(crate::assertions::SavedInput::Text {
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
                crate::chat_actions::ChatEvent {
                    id: action,
                    by: Some(by),
                    kind: outcome,
                    phase,
                    detail: detail.as_deref(),
                    receipt: receipt.as_ref(),
                    at,
                    seq,
                    request: &r.id,
                },
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
            let value = polylogue_model::settings::parse(key, value, mode)
                .map_err(crate::StoreError::invalid)?;
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
    let effect = if matches!(r.write, UserWrite::ResolveLine { .. })
        || matches!(&r.write,UserWrite::AssertQuery {resolved,..} if resolved.is_empty())
    {
        polylogue_model::authored::AuthoredEffect::default()
    } else {
        crate::primary::authored::capture(tx, r, seq)?
    };
    let accepted = save_outcomes.as_ref().map(|outcomes| {
        let mut accepted = r.clone();
        if let UserWrite::SaveSelection(save) = &mut accepted.write {
            save.outcomes = Some(outcomes.clone());
        }
        accepted
    });
    let accepted = accepted.as_ref().unwrap_or(r);
    tx.execute(
        &source
            .sql("UPDATE @source.client_request SET outcome=?2,effect=?3,request=?4 WHERE id=?1"),
        params![
            r.id,
            crate::primary::authored::pack(&outcome)?,
            crate::primary::authored::pack(&effect)?,
            crate::primary::authored::pack(accepted)?,
        ],
    )?;
    Ok(UserReceipt {
        outcome,
        save: save_outcomes,
    })
}

#[cfg(test)]
mod tests;
