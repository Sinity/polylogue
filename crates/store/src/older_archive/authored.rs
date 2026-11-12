//! What the older format's authored rows become here. Notes and settings
//! become assertion facts restored unchanged; deliveries, their events and
//! kept results become the authored requests that wrote them, with their
//! original identities, times and outcomes. A row that cannot be expressed
//! here is refused with the reason, never changed to fit.

use super::{Delivery, DeliveryEvent, KeptResult, Machine, Note, Request, Setting};
use polylogue_model::assertion::{
    AssertionFact, AssertionKind, AssertionLink, LinkRole, ValueShape,
};
use polylogue_model::authored::{
    AuthoredEffect, DeliveryChange, DeliveryEvent as RestoredEvent, DeliveryRecord,
    KeptResult as RestoredKept, PreparedRequest,
};
use polylogue_model::authorship::{Authorship, Channel};
use polylogue_model::user::{
    Delivery as RestoredDelivery, LeftOut, UserOutcome, UserRequest, UserWrite,
};
use polylogue_model::{Author, DeliveryOutcome, Reference, RevisionReference, Timestamp};
use std::collections::{BTreeMap, BTreeSet};

/// The result of mapping one row: its form here, or why it has none.
pub type Mapped<T> = std::result::Result<T, String>;

fn time(ms: i64) -> Mapped<Timestamp> {
    Timestamp::from_unix_ms(ms).ok_or_else(|| format!("time {ms} is out of range"))
}

/// How the older format named a note, `note:<id>`, as a reference to that
/// exact assertion revision; `roots` gives each note's first revision. A
/// withdrawal named the whole family by its first note, so it stays a
/// reference to the family. Other references are read as they were written.
fn reference(text: &str, role: LinkRole, roots: &BTreeMap<String, String>) -> Mapped<Reference> {
    if let Some(id) = text.strip_prefix("note:") {
        if id.is_empty() {
            return Err("a note reference names no note".into());
        }
        return Ok(Reference::Assertion(if role == LinkRole::Withdraws {
            RevisionReference {
                root: id.to_owned(),
                revision: None,
            }
        } else {
            RevisionReference::exact(
                roots.get(id).cloned().unwrap_or_else(|| id.to_owned()),
                id.to_owned(),
            )
        }));
    }
    Reference::parse(text).map_err(|error| error.to_string())
}

fn author(text: &str) -> Mapped<Author> {
    Author::parse(text).map_err(|error| error.to_string())
}

fn kind(text: &str) -> Mapped<AssertionKind> {
    serde_json::from_value(serde_json::Value::String(text.to_owned()))
        .map_err(|_| format!("note kind {text} has no counterpart here"))
}

fn role(text: &str) -> Mapped<LinkRole> {
    serde_json::from_value(serde_json::Value::String(text.to_owned()))
        .map_err(|_| format!("note relation {text} has no counterpart here"))
}

/// A note's value as the kind's value shape reads it. Text kinds keep the
/// text exactly; the older format kept a tag's value as text, which here is
/// the tag record's `value`; other kinds were stored as JSON text.
fn value(kind: AssertionKind, value: Option<&str>) -> Option<serde_json::Value> {
    let value = value?;
    match kind.spec().value {
        ValueShape::Text | ValueShape::OptionalText => Some(value.into()),
        ValueShape::Tag => Some(serde_json::json!({ "value": value })),
        _ => Some(serde_json::from_str(value).unwrap_or_else(|_| value.into())),
    }
}

/// Each note's first revision, by note ID.
pub fn roots(notes: &[Note]) -> BTreeMap<String, String> {
    notes
        .iter()
        .map(|note| (note.id.clone(), note.first_note.clone()))
        .collect()
}

/// A note as an assertion fact, checked against this archive's rules for
/// its kind. `roots` is [`roots`] of every note in the older archive.
pub fn note(note: &Note, roots: &BTreeMap<String, String>) -> Mapped<AssertionFact> {
    let kind = if note.kind == "tag" && note.key.as_deref() == Some("hidden") {
        AssertionKind::Hidden
    } else {
        kind(&note.kind)?
    };
    let mut links = Vec::new();
    for target in &note.targets {
        links.push(AssertionLink {
            role: LinkRole::About,
            reference: reference(target, LinkRole::About, roots)?,
            seen_hash: None,
            quote: None,
        });
    }
    for (relation, other) in &note.relations {
        let role = role(relation)?;
        let target = reference(other, role, roots)?;
        // A predecessor relation already states what the note is about.
        if role.is_relation()
            && !links
                .iter()
                .any(|link| link.role == LinkRole::About && link.reference == target)
        {
            links.push(AssertionLink {
                role: LinkRole::About,
                reference: target.clone(),
                seen_hash: None,
                quote: None,
            });
        }
        links.push(AssertionLink {
            role,
            reference: target,
            seen_hash: None,
            quote: None,
        });
    }
    let fact = AssertionFact {
        id: note.id.clone(),
        root: note.first_note.clone(),
        kind,
        key: (kind != AssertionKind::Hidden)
            .then(|| note.key.clone())
            .flatten(),
        value: value(kind, note.value.as_deref()),
        author: author(&note.author)?,
        via: Channel::Import,
        machine: note.machine.clone(),
        created_at: Some(time(note.created_at)?),
        valid_until: note.valid_until.map(time).transpose()?,
        links,
    };
    fact.canonicalized().map_err(|error| error.to_string())
}

/// Settings as assertion facts, each with its outcome. A setting row was a
/// new value for its key on its machine; here a later value revises the one
/// before it, so the value in force is the same one. An archive-wide
/// setting (`*`) records no machine that wrote it; when the older archive
/// knew exactly one machine, that is the one.
pub fn settings(rows: &[Setting], machines: &[Machine]) -> Vec<(String, Mapped<AssertionFact>)> {
    let only_machine = {
        let ids: BTreeSet<&str> = machines.iter().map(|m| m.id.as_str()).collect();
        (ids.len() == 1).then(|| ids.into_iter().next().expect("one machine").to_owned())
    };
    // The previous mapped revision of each (machine, key), and the root it shares.
    let mut previous: BTreeMap<(String, String), (String, String)> = BTreeMap::new();
    let mut out = Vec::new();
    for row in rows {
        let slot = (row.machine.clone(), row.key.clone());
        let mapped = (|| {
            let explicit = (row.machine != "*").then_some(row.machine.as_str());
            let written_on = match explicit {
                Some(machine) => machine.to_owned(),
                None => only_machine.clone().ok_or_else(|| {
                    format!(
                        "archive-wide setting {} records no machine and the archive knew {} machines",
                        row.key,
                        machines.len()
                    )
                })?,
            };
            let value = polylogue_model::settings::parse(
                &row.key,
                &row.value,
                polylogue_model::settings::Owner::Standalone,
            )
            .map_err(|error| error.to_string())?;
            let target = polylogue_model::settings::target_machine(&row.key, explicit, &written_on)
                .map_err(|error| error.to_string())?;
            let (root, links) = match previous.get(&slot) {
                Some((root, before)) => (
                    root.clone(),
                    vec![AssertionLink {
                        role: LinkRole::Revises,
                        reference: Reference::Assertion(RevisionReference::exact(
                            root.clone(),
                            before.clone(),
                        )),
                        seen_hash: None,
                        quote: None,
                    }],
                ),
                None => (row.id.clone(), Vec::new()),
            };
            let fact = AssertionFact {
                id: row.id.clone(),
                root,
                kind: AssertionKind::Setting,
                key: Some(row.key.clone()),
                value: Some(
                    serde_json::to_value(polylogue_model::assertion::values::Setting {
                        machine: target,
                        value,
                    })
                    .map_err(|error| error.to_string())?,
                ),
                author: author(&row.author)?,
                via: Channel::Import,
                machine: written_on,
                created_at: Some(time(row.set_at)?),
                valid_until: None,
                links,
            };
            fact.canonicalized().map_err(|error| error.to_string())
        })();
        if let Ok(fact) = &mapped {
            let root = previous
                .get(&slot)
                .map_or_else(|| fact.id.clone(), |(root, _)| root.clone());
            previous.insert(slot, (root, fact.id.clone()));
        }
        out.push((row.id.clone(), mapped));
    }
    out
}

/// Which kind of authored row a request wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestClass {
    /// Restored as an authored request with its delivery.
    Delivery,
    /// Restored as an authored request with the delivery it reported on.
    DeliveryEvent,
    /// Restored as an authored request with its kept result.
    KeptResult,
    /// Its effect is a note or setting, restored as an assertion fact.
    Note,
    /// A chat action or backfill job request: refused with those tables.
    Queue,
}

/// One request's outcome.
#[derive(Debug, Clone)]
pub struct MappedRequest {
    pub id: String,
    pub class: Option<RequestClass>,
    pub request: Mapped<Option<PreparedRequest>>,
}

/// Rows the request mapping could not carry, beyond the requests themselves.
#[derive(Debug, Default, Clone)]
pub struct RequestRows {
    pub requests: Vec<MappedRequest>,
    /// Deliveries no request carries, with the reason.
    pub deliveries: Vec<(String, String)>,
    /// Kept results no request carries, with the reason.
    pub kept_results: Vec<(String, String)>,
}

struct Body {
    machine: String,
    at: Timestamp,
    tag: String,
    payload: serde_json::Map<String, serde_json::Value>,
}

fn body(request: &Request) -> Mapped<Body> {
    let Some(packed) = &request.body else {
        return Err("the older archive kept no body for this request (before version 3)".into());
    };
    let text = crate::encoding::unpack_text(packed).map_err(|error| error.to_string())?;
    let mut value: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&text)
        .map_err(|error| format!("its body is not a request: {error}"))?;
    let machine = value
        .remove("machine")
        .and_then(|m| m.as_str().map(str::to_owned))
        .ok_or("its body names no machine")?;
    let at = value
        .remove("at")
        .and_then(|at| serde_json::from_value::<Timestamp>(at).ok())
        .map_or_else(|| time(request.created_at), Ok)?;
    let Some(serde_json::Value::Object(mut payload)) = value.remove("write") else {
        return Err("its body has no write".into());
    };
    let tag = payload
        .remove("write")
        .and_then(|tag| tag.as_str().map(str::to_owned))
        .ok_or("its write has no name")?;
    Ok(Body {
        machine,
        at,
        tag,
        payload,
    })
}

fn class(tag: &str) -> Option<RequestClass> {
    Some(match tag {
        "deliver" | "delivered_advice" => RequestClass::Delivery,
        "delivery_event" => RequestClass::DeliveryEvent,
        "keep_result" => RequestClass::KeptResult,
        "add_note" | "save_query" | "annotate_query" | "withdraw" | "set_setting" => {
            RequestClass::Note
        }
        "chat_enqueue" | "chat_claim" | "chat_report" | "chat_decide" | "backfill_start"
        | "backfill_claim" | "backfill_report" | "backfill_cancel" => RequestClass::Queue,
        _ => return None,
    })
}

fn delivery_text(row: &Delivery) -> Mapped<RestoredDelivery> {
    let text = crate::encoding::unpack_text(&row.body).map_err(|error| error.to_string())?;
    if blake3::hash(text.as_bytes()).as_bytes()[..] != row.content_hash[..]
        || text.len() as i64 != row.bytes
    {
        return Err("its text does not match its recorded hash and length".into());
    }
    let refs: Vec<String> = serde_json::from_str(&row.refs_json).map_err(|e| e.to_string())?;
    let left_out: Vec<LeftOut> =
        serde_json::from_str(&row.left_out_json).map_err(|e| e.to_string())?;
    Ok(RestoredDelivery {
        recipient_kind: row.recipient_kind.clone(),
        recipient_id: row.recipient_id.clone(),
        sender: row.sender.clone(),
        event: row.event.clone(),
        cwd: row.cwd.clone(),
        text,
        refs,
        left_out,
    })
}

fn outcome(id: &str) -> UserOutcome {
    UserOutcome {
        id: id.to_owned(),
        repeat: false,
        pending: false,
    }
}

fn request(id: &str, body: &Body, write: UserWrite) -> UserRequest {
    UserRequest {
        id: id.to_owned(),
        machine: body.machine.clone(),
        // The older format recorded no author for these writes.
        authorship: Authorship {
            author: Author::Unattributed,
            via: Channel::Import,
        },
        at: body.at,
        write,
    }
}

fn what(text: &str) -> Mapped<DeliveryOutcome> {
    serde_json::from_value(serde_json::Value::String(text.to_owned()))
        .map_err(|_| format!("delivery event {text} has no counterpart here"))
}

/// Deliveries, their events and kept results as the requests that wrote
/// them. Every request is accounted for: restored, carried by its note or
/// setting, or refused with the reason; and so is every delivery, delivery
/// event and kept result.
pub fn requests(
    requests: &[Request],
    deliveries: &[Delivery],
    events: &[DeliveryEvent],
    kept: &[KeptResult],
) -> RequestRows {
    let mut rows = RequestRows::default();
    let bodies: Vec<(&Request, Mapped<Body>)> = requests
        .iter()
        .map(|request| (request, body(request)))
        .collect();

    // Which request wrote each delivery event: the advice request that wrote
    // its delivery, or the first event request with the same delivery,
    // outcome and time.
    let mut event_request: BTreeMap<(String, String, i64), String> = BTreeMap::new();
    for (request, body) in &bodies {
        let Ok(body) = body else { continue };
        let key = match body.tag.as_str() {
            "delivered_advice" => (
                request.id.clone(),
                "delivered".to_owned(),
                body.at.unix_ms(),
            ),
            "delivery_event" => {
                let delivery = body.payload.get("delivery").and_then(|d| d.as_str());
                let what = body.payload.get("what").and_then(|w| w.as_str());
                let (Some(delivery), Some(what)) = (delivery, what) else {
                    continue;
                };
                (delivery.to_owned(), what.to_owned(), body.at.unix_ms())
            }
            _ => continue,
        };
        event_request
            .entry(key)
            .or_insert_with(|| request.id.clone());
    }

    // Each delivery's whole record, or why it has none.
    let mut records: BTreeMap<String, Mapped<DeliveryRecord>> = BTreeMap::new();
    for row in deliveries {
        let record = (|| {
            let prepared = delivery_text(row)?;
            let mut restored = Vec::new();
            for event in events.iter().filter(|e| e.delivery == row.id) {
                let request = event_request
                    .get(&(event.delivery.clone(), event.what.clone(), event.at))
                    .ok_or_else(|| {
                        format!(
                            "its {} event at {} has no request that wrote it",
                            event.what, event.at
                        )
                    })?;
                restored.push(RestoredEvent {
                    request: request.clone(),
                    what: what(&event.what)?,
                    at: time(event.at)?,
                    detail: event.detail.clone(),
                });
            }
            Ok(DeliveryRecord {
                signal: None,
                id: row.id.clone(),
                prepared,
                at: time(row.at)?,
                events: restored,
            })
        })();
        records.insert(row.id.clone(), record);
    }

    let mut carried_deliveries = BTreeSet::new();
    let mut carried_kept = BTreeSet::new();
    // Why the request that wrote a delivery or kept result could not carry it.
    let mut failed: BTreeMap<String, String> = BTreeMap::new();
    for (request, body) in bodies {
        let mapped = match body {
            Err(reason) => MappedRequest {
                id: request.id.clone(),
                class: None,
                request: Err(reason),
            },
            Ok(body) => {
                let class = class(&body.tag);
                let restored = match class {
                    None => Err(format!("request {} has no counterpart here", body.tag)),
                    Some(RequestClass::Note) => Ok(None),
                    Some(RequestClass::Queue) => Err(
                        "chat actions and backfill jobs are operational queues with no counterpart here"
                            .into(),
                    ),
                    Some(RequestClass::Delivery) => (|| {
                        let record = records
                            .get(&request.id)
                            .ok_or("the delivery it wrote is not in the older archive")?
                            .clone()?;
                        let payload: RestoredDelivery =
                            serde_json::from_value(serde_json::Value::Object(body.payload.clone()))
                                .map_err(|error| error.to_string())?;
                        if payload != record.prepared {
                            return Err("the request and the delivery it wrote disagree".into());
                        }
                        let write = if body.tag == "deliver" {
                            UserWrite::Deliver(payload)
                        } else {
                            UserWrite::DeliveredAdvice(payload)
                        };
                        carried_deliveries.insert(request.id.clone());
                        let added = record
                            .events
                            .iter()
                            .filter(|e| e.request == request.id)
                            .cloned()
                            .collect();
                        Ok(Some(PreparedRequest {
                            request: self::request(&request.id, &body, write),

                            outcome: outcome(&request.id),
                            effect: AuthoredEffect {
                                deliveries: vec![DeliveryChange {
                                    delivery: record,
                                    added_events: added,
                                }],
                                ..Default::default()
                            },
                        }))
                    })(),
                    Some(RequestClass::DeliveryEvent) => (|| {
                        let delivery = body
                            .payload
                            .get("delivery")
                            .and_then(|d| d.as_str())
                            .ok_or("it names no delivery")?
                            .to_owned();
                        let outcome_of = what(
                            body.payload
                                .get("what")
                                .and_then(|w| w.as_str())
                                .ok_or("it names no outcome")?,
                        )?;
                        let detail = match body.payload.get("detail") {
                            None | Some(serde_json::Value::Null) => None,
                            Some(serde_json::Value::String(text)) => Some(text.clone()),
                            Some(_) => return Err("its detail is not text".into()),
                        };
                        let record = records
                            .get(&delivery)
                            .ok_or("the delivery it reported on is not in the older archive")?
                            .clone()?;
                        let added = record
                            .events
                            .iter()
                            .filter(|e| e.request == request.id)
                            .cloned()
                            .collect();
                        Ok(Some(PreparedRequest {
                            request: self::request(
                                &request.id,
                                &body,
                                UserWrite::DeliveryEvent {
                                    delivery: delivery.clone(),
                                    what: outcome_of,
                                    detail,
                                },
                            ),

                            outcome: outcome(&delivery),
                            effect: AuthoredEffect {
                                deliveries: vec![DeliveryChange {
                                    delivery: record,
                                    added_events: added,
                                }],
                                ..Default::default()
                            },
                        }))
                    })(),
                    Some(RequestClass::KeptResult) => (|| {
                        let row = kept
                            .iter()
                            .find(|k| k.id == request.id)
                            .ok_or("the result it kept is not in the older archive")?;
                        let name = match body.payload.get("name") {
                            None | Some(serde_json::Value::Null) => None,
                            Some(serde_json::Value::String(text)) => Some(text.clone()),
                            Some(_) => return Err("its name is not text".into()),
                        };
                        let query = body
                            .payload
                            .get("query")
                            .and_then(|q| q.as_str())
                            .ok_or("it names no query")?
                            .to_owned();
                        if name != row.name || query != row.query {
                            return Err("the request and the result it kept disagree".into());
                        }
                        carried_kept.insert(row.id.clone());
                        Ok(Some(PreparedRequest {
                            request: self::request(
                                &request.id,
                                &body,
                                UserWrite::KeepResult {
                                    name: name.clone(),
                                    query: query.clone(),
                                    rows_json: row.rows_json.clone(),
                                    read_at: row.read_at.clone(),
                                },
                            ),

                            outcome: outcome(&request.id),
                            effect: AuthoredEffect {
                                kept_results: vec![RestoredKept {
                                    id: row.id.clone(),
                                    name,
                                    query,
                                    ran_at: time(row.ran_at)?,
                                    read_at: row.read_at.clone(),
                                    rows_json: row.rows_json.clone(),
                                }],
                                ..Default::default()
                            },
                        }))
                    })(),
                };
                if let (Some(RequestClass::Delivery | RequestClass::KeptResult), Err(reason)) =
                    (class, &restored)
                {
                    failed.insert(request.id.clone(), reason.clone());
                }
                MappedRequest {
                    id: request.id.clone(),
                    class,
                    request: restored,
                }
            }
        };
        rows.requests.push(mapped);
    }

    for row in deliveries {
        if carried_deliveries.contains(&row.id) {
            continue;
        }
        let reason = match (failed.get(&row.id), records.get(&row.id)) {
            (Some(reason), _) | (None, Some(Err(reason))) => reason.clone(),
            _ => "no request with a kept body wrote it".to_owned(),
        };
        rows.deliveries.push((row.id.clone(), reason));
    }
    for row in kept {
        if !carried_kept.contains(&row.id) {
            let reason = failed
                .get(&row.id)
                .cloned()
                .unwrap_or_else(|| "no request with a kept body wrote it".to_owned());
            rows.kept_results.push((row.id.clone(), reason));
        }
    }
    rows
}

#[cfg(test)]
mod tests;
