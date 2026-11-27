//! Writes people and agents make to `archive.db`: annotations (tags, notes,
//! judgments and the rest) and their withdrawals, seen marks, saved
//! queries, signals, alternates, chat actions, import jobs, context
//! deliveries and their events, and settings. They go through the one writer
//! thread like everything else, each in its own small transaction, so a write
//! never waits behind a first build for longer than one group.
//!
//! Every write carries a client request ID. The same ID with the same
//! intent writes nothing and returns what the first write created; the
//! same ID with other intent is refused as a conflict.

mod save;
use crate::{Result, StoreError};
use rusqlite::{Transaction, params};

use polylogue_model::user::{UserOutcome, UserRequest, UserWrite};

/// Runtime completion, distinct from the immutable stored outcome schema.
#[derive(Debug)]
pub struct WriteResult {
    pub outcome: UserOutcome,
    pub save: Option<polylogue_model::user::SaveReport>,
    /// Exact annotation revisions created by the accepted write, including on retry.
    pub written: Vec<polylogue_model::Reference>,
    /// Resolved tag targets absent from the original accepted write's effects.
    pub already: Vec<polylogue_model::Reference>,
}
impl WriteResult {
    pub(crate) fn recorded(
        prepared: &polylogue_model::authored::PreparedRequest,
        outcome: UserOutcome,
    ) -> Result<Self> {
        let save =
            match &prepared.request.write {
                UserWrite::Save(save) => Some(save.outcomes.clone().ok_or_else(|| {
                    StoreError::internal("accepted Save has no resolved outcomes")
                })?),
                _ => None,
            };
        Ok(Self {
            outcome,
            save,
            already: Self::already(&prepared.request, &prepared.effect),
            written: if matches!(prepared.request.write, UserWrite::MarkSeen { .. }) {
                Vec::new()
            } else {
                Self::references(&prepared.effect)
            },
        })
    }
    fn already(
        request: &UserRequest,
        effect: &polylogue_model::authored::AuthoredEffect,
    ) -> Vec<polylogue_model::Reference> {
        let UserWrite::AssertQuery {
            annotation,
            resolved,
            ..
        } = &request.write
        else {
            return Vec::new();
        };
        if annotation.kind != polylogue_model::annotation::AnnotationKind::Tag {
            return Vec::new();
        }
        let written: std::collections::BTreeSet<_> = effect
            .annotations
            .iter()
            .flat_map(|fact| fact.about())
            .collect();
        resolved
            .iter()
            .filter(|target| !written.contains(target))
            .cloned()
            .collect()
    }
    fn references(
        effect: &polylogue_model::authored::AuthoredEffect,
    ) -> Vec<polylogue_model::Reference> {
        effect
            .annotations
            .iter()
            .map(|fact| {
                polylogue_model::Reference::Annotation(polylogue_model::RevisionReference::exact(
                    fact.root.clone(),
                    fact.id.clone(),
                ))
            })
            .collect()
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
    pub provisional: Option<&'a mut crate::selection::ProvisionalQueries>,
}
impl<'a> Preparation<'a> {
    pub(crate) fn plain(context: Option<&'a PreparationContext>) -> Self {
        Self {
            context,
            provisional: None,
        }
    }
    pub(crate) fn preflight(
        context: Option<&'a PreparationContext>,
        provisional: &'a mut crate::selection::ProvisionalQueries,
    ) -> Self {
        Self {
            context,
            provisional: Some(provisional),
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
    apply_resolving(tx, tx, r, seq, Preparation::plain(None)).map(|receipt| receipt.outcome)
}

pub(crate) fn apply_resolving(
    tx: &Transaction<'_>,
    original: &rusqlite::Connection,
    r: &UserRequest,
    seq: i64,
    preparation: Preparation<'_>,
) -> Result<WriteResult> {
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
    // Delivery rows and their receipts store Unix milliseconds;
    // the original request keeps its full timestamp and retry identity.
    let at = polylogue_model::Timestamp::from_unix_ms(r.at.unix_ms())
        .expect("a stored millisecond timestamp remains representable");
    let events = if matches!(r.write, UserWrite::DeliveredAdvice(_)) {
        vec![DeliveryEvent {
            request: r.id.clone(),
            outcome: polylogue_model::DeliveryOutcome::Delivered,
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
) -> Result<WriteResult> {
    let source = crate::source::Source::of(tx)?;
    if let Some(known) = crate::authored_records::known_request(tx, r)? {
        let mut outcome = known.outcome.clone();
        outcome.repeat = true;
        return WriteResult::recorded(&known, outcome);
    }
    let advancing_seen = if let UserWrite::MarkSeen { marks } = &r.write {
        let marks = crate::annotations::seen::advancing(
            tx,
            original,
            &r.authorship.author.to_string(),
            marks,
        )?;
        if marks.is_empty() {
            return Ok(WriteResult {
                outcome: UserOutcome {
                    id: r.id.clone(),
                    repeat: true,
                    pending: false,
                },
                save: None,
                written: Vec::new(),
                already: Vec::new(),
            });
        }
        Some(marks)
    } else {
        None
    };
    if let Some(prepared) = prepare_delivery(r) {
        crate::authored_records::require_new_delivery(tx, &r.id)?;
        crate::authored_records::apply(tx, &prepared, seq)?;
        return Ok(WriteResult {
            outcome: prepared.outcome,
            save: None,
            written: WriteResult::references(&prepared.effect),
            already: Vec::new(),
        });
    }
    let hash = crate::authored_records::request_hash(r)?;
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
    let mode = polylogue_model::settings::SettingMode::Standalone;
    let hash = |reference: &polylogue_model::Reference| match reference {
        polylogue_model::Reference::Session(_)
        | polylogue_model::Reference::Message(_)
        | polylogue_model::Reference::Block { .. }
        | polylogue_model::Reference::Document(_)
        | polylogue_model::Reference::MessageRange { .. }
        | polylogue_model::Reference::Span { .. }
        | polylogue_model::Reference::Annotation(_)
        | polylogue_model::Reference::Alternate { .. } => Ok(
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
    let mut saved = crate::annotations::SavedPreparation::new(original, tx, r, preparation);
    let mut resolve_saved = |input: crate::annotations::SavedInput<'_>| saved.resolve(input);
    let context = crate::annotations::WriteContext {
        request: r,
        counter: seq,
        hash: &hash,
        quote: &quote,
        mode,
    };
    let mut save_outcomes = None;
    let outcome = match &r.write {
        UserWrite::MarkSeen { .. } => {
            let statements = advancing_seen
                .as_ref()
                .expect("seen marks prepared")
                .iter()
                .map(|mark| {
                    Ok(polylogue_model::annotation::NewAnnotation {
                        kind: polylogue_model::annotation::AnnotationKind::Seen,
                        key: Some(mark.target.clone()),
                        value: Some(serde_json::to_value(
                            polylogue_model::annotation::values::Seen {
                                position: polylogue_model::annotation::values::SeenPosition {
                                    item: mark.item.clone(),
                                    at: mark.at.to_string(),
                                },
                            },
                        )?),
                        valid_until: None,
                        links: Vec::new(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            crate::annotations::write_preparing(tx, &statements, context, &mut resolve_saved)
        }

        UserWrite::Save(selected) => {
            save_outcomes = Some(save::apply(tx, selected, context, &mut resolve_saved)?);
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
        UserWrite::Assert { annotations } => {
            crate::annotations::write_preparing(tx, annotations, context, &mut resolve_saved)
        }
        UserWrite::AssertQuery {
            annotation,
            resolved,
            ..
        } => {
            use polylogue_model::annotation::{LinkRole, NewAnnotationLink};
            if annotation
                .links
                .iter()
                .any(|link| link.role == LinkRole::About)
            {
                return Err(polylogue_model::Reason::QueryAnnotationAboutLinks {}.into());
            }
            // One deliberate write of a single-valued mark revises it; a bulk
            // write over several rows never does. Systematic labelling goes
            // through a rubric field, whose slot includes its author.
            if annotation.kind.spec().single_valued_mark() && resolved.len() > 1 {
                return Err(polylogue_model::Reason::SingleValuedBulk {
                    kind: annotation.kind.as_str().into(),
                    rows: resolved.len(),
                }
                .into());
            }
            let mut targets = std::collections::BTreeSet::new();
            let mut statements = Vec::with_capacity(resolved.len());
            for reference in resolved {
                if !targets.insert(reference) {
                    return Err(StoreError::internal(
                        "a query annotation contains duplicate resolved targets",
                    ));
                }
                let mut statement = annotation.clone();
                statement.links.push(NewAnnotationLink {
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
                crate::annotations::write_preparing(tx, &statements, context, &mut resolve_saved)
            }
        }
        UserWrite::ReviseAnnotations { bases, patch } => {
            crate::annotations::revise_preparing(tx, bases, patch, context, &mut resolve_saved)
        }
        UserWrite::WithdrawAnnotations { targets, reason } => {
            use polylogue_model::annotation::{
                AnnotationKind, LinkRole, NewAnnotation, NewAnnotationLink,
            };
            let targets = save::withdrawal_targets(tx, targets)?;
            let statement = NewAnnotation {
                kind: AnnotationKind::Withdrawal,
                key: None,
                value: reason
                    .as_ref()
                    .map(|reason| serde_json::Value::String(reason.clone())),
                valid_until: None,
                links: targets
                    .iter()
                    .map(|reference| NewAnnotationLink {
                        role: LinkRole::Withdraws,
                        reference: reference.clone(),
                    })
                    .collect(),
            };
            crate::annotations::write_preparing(tx, &[statement], context, &mut resolve_saved)
        }
        UserWrite::SignalCommit(commit) => crate::signals::write(tx, r, commit, seq),
        UserWrite::Alternate { alternate } => crate::alternates::write(tx, r, alternate, seq),
        UserWrite::Deliver(_) | UserWrite::DeliveredAdvice(_) => {
            unreachable!("literal deliveries are applied before resolving other writes")
        }
        UserWrite::DeliveryEvent {
            delivery,
            outcome,
            detail,
        } => {
            let known: bool = tx
                .prepare_cached(&source.sql("SELECT 1 FROM @source.delivery WHERE id = ?1"))?
                .exists([delivery])?;
            if !known {
                return Err(StoreError::NotFound(format!("delivery {delivery}")));
            }
            tx.execute(
                &source.sql("INSERT OR IGNORE INTO @source.delivery_event(delivery, outcome, at, detail, request, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)"),
                params![delivery, outcome.as_str(), at, detail, r.id, seq],
            )?;
            Ok(UserOutcome {
                pending: false,
                id: delivery.clone(),
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
                crate::chat_actions::ChatEventReport {
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
        UserWrite::ImportJobStart(new) => {
            let id = crate::import_jobs::start(tx, new, at, seq, &r.id)?;
            Ok(UserOutcome {
                pending: false,
                repeat: id != new.id,
                id,
            })
        }
        UserWrite::ImportJobClaim { by } => Ok(UserOutcome {
            pending: false,
            id: crate::import_jobs::claim(tx, by, at, seq, &r.id)?,
            repeat: false,
        }),
        UserWrite::ImportJobReport { job, by, report } => {
            crate::import_jobs::report(tx, job, by, report, at, seq, &r.id)?;
            Ok(UserOutcome {
                pending: false,
                id: job.clone(),
                repeat: false,
            })
        }
        UserWrite::ImportJobCancel { job } => {
            crate::import_jobs::cancel(tx, job, at, seq, &r.id)?;
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
                .map_err(polylogue_model::Reason::from)?;
            let statement = polylogue_model::annotation::NewAnnotation {
                kind: polylogue_model::annotation::AnnotationKind::Setting,
                key: Some(key.clone()),
                value: Some(serde_json::to_value(
                    polylogue_model::annotation::values::Setting {
                        machine: machine.clone(),
                        value,
                    },
                )?),
                valid_until: None,
                links: Vec::new(),
            };
            crate::annotations::write_preparing(tx, &[statement], context, &mut resolve_saved)
        }
    }?;
    let effect = if matches!(r.write, UserWrite::ResolveLine { .. })
        || matches!(&r.write,UserWrite::AssertQuery {resolved,..} if resolved.is_empty())
    {
        polylogue_model::authored::AuthoredEffect::default()
    } else {
        crate::authored_records::capture(tx, r, seq)?
    };
    let accepted = save_outcomes.as_ref().map(|outcomes| {
        let mut accepted = r.clone();
        if let UserWrite::Save(save) = &mut accepted.write {
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
            crate::authored_records::pack(&outcome)?,
            crate::authored_records::pack(&effect)?,
            crate::authored_records::pack(accepted)?,
        ],
    )?;
    Ok(WriteResult {
        outcome,
        save: save_outcomes,
        already: WriteResult::already(accepted, &effect),
        written: if matches!(r.write, UserWrite::MarkSeen { .. }) {
            Vec::new()
        } else {
            WriteResult::references(&effect)
        },
    })
}

#[cfg(test)]
mod tests;
