//! Snapshot intents are part of the same transaction as the confirmed position.
//! Selection comes from that transaction's projection; payloads come from its
//! complete canonical facts. The publisher never scans the read index later.
use super::{Product, ProductWrite, Sinex};
use crate::{Result, StoreError};
use polylogue_model::work_products::{
    self, ReadingSource, SelectedBackground, SelectedMessage, WorkMessage, WorkSnapshot,
};
use polylogue_model::{Message, Session, SessionId, WrittenBy};
use rusqlite::{Connection, OptionalExtension, Transaction, params};

mod outbox;
pub use outbox::stage_products;
pub(crate) use outbox::{SCHEMA, publish};

fn pack(value: &impl serde::Serialize) -> Result<Vec<u8>> {
    crate::encoding::pack_bytes(&serde_json::to_vec(value)?)
}
fn product(bytes: &[u8]) -> Result<Product> {
    crate::encoding::unpack_json(bytes)
}
fn session(product: &Product) -> Option<&SessionId> {
    match product {
        Product::Session { facts, .. } => Some(&facts.id),
        Product::Message { facts, .. } => Some(&facts.message.session),
        Product::Event { facts, .. } => facts.session.as_ref(),
        Product::Background { facts, .. } => Some(&facts.session),
        _ => None,
    }
}

/// Called after rematerialization and before the position transaction commits.
pub(crate) fn stage(
    tx: &Transaction<'_>,
    position: i64,
    counter: i64,
    proof: Option<&super::RawInputProof>,
) -> Result<()> {
    tx.execute_batch("CREATE TEMP TABLE IF NOT EXISTS primary_candidate AS SELECT * FROM primary_fact WHERE 0; CREATE TEMP VIEW IF NOT EXISTS primary_next AS SELECT * FROM primary_candidate UNION ALL SELECT f.* FROM main.primary_fact f WHERE NOT EXISTS(SELECT 1 FROM primary_candidate c WHERE c.event=f.event);")?;
    outbox::confirmed(tx, position)?;
    tx.execute("INSERT OR IGNORE INTO primary_work_dirty SELECT session FROM primary_candidate WHERE kind='parsed' AND session IS NOT NULL UNION SELECT f.session FROM primary_fact f JOIN primary_candidate c ON c.event=f.event WHERE f.kind='parsed' AND f.session IS NOT NULL UNION SELECT f.session FROM primary_fact f JOIN primary_candidate c ON c.input=f.input WHERE c.kind='input' AND f.kind='parsed' AND f.session IS NOT NULL",[])?;
    tx.execute("INSERT OR IGNORE INTO primary_work_dirty SELECT id FROM session WHERE changed=?1 AND EXISTS(SELECT 1 FROM primary_candidate WHERE kind IN ('parsed','input','blob'))",[counter])?;
    let Some(proof) = proof else {
        return Ok(());
    };
    if proof.position != position || proof.scope.is_empty() || proof.stamp.is_empty() {
        return Err(StoreError::Invalid(
            "canonical publication proof differs from the applied transaction".into(),
        ));
    }
    graph::stage(tx, proof)
}

mod graph;

fn hash_message(message: &Message) -> Result<[u8; 32]> {
    Ok(message.content_hash()?.0)
}

/// Only metadata used to select a reading is read from the projection.
/// Every supplied message body is the complete immutable confirmed Product.
fn selected(
    index: &Connection,
    reading: &super::Reading,
    facts: &polylogue_model::MessageRow,
    projection: Option<&polylogue_model::bundle::MessageProjection>,
) -> Result<Option<(String, i64, i64)>> {
    let choice:Option<(String,i64,i64,Vec<u8>)>=index.query_row("SELECT m.order_input_id,m.seq,m.record_position,s.version_hash FROM message m JOIN message_source s ON s.message_pk=m.pk AND s.shown=1 JOIN input_ref i ON i.pk=s.input_pk WHERE m.id=?1 AND i.id=?2 AND s.chain=?3 AND s.source_start=?4 AND s.source_end=?5 AND m.duplicate_of IS NULL AND coalesce(m.on_active_path,1)=1",params![facts.message.id.to_string(),reading.input.id,reading.chain,facts.offset,facts.end],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?))).optional()?;
    let Some((input, seq, ordinal, hash)) = choice else {
        return Ok(None);
    };
    Ok((hash.as_slice()
        == projection
            .map(|projection| projection.source_version.0)
            .map_or_else(|| hash_message(&facts.message), Ok)?)
    .then_some((input, seq, ordinal)))
}

fn snapshot(
    index: &Connection,
    work: &work_products::membership::LogicalWork,
    membership: work_products::SnapshotMembership,
    position: i64,
) -> Result<WorkSnapshot> {
    let id = work
        .root
        .as_ref()
        .ok_or_else(|| StoreError::Invalid("a selected work has no established root".into()))?;
    index.execute_batch("DROP TABLE IF EXISTS temp.primary_work_physical; CREATE TEMP TABLE primary_work_physical(session TEXT PRIMARY KEY) STRICT;")?;
    for session in &work.physical {
        index.execute(
            "INSERT INTO primary_work_physical VALUES (?1)",
            [session.to_string()],
        )?;
    }
    index.execute_batch("DROP TABLE IF EXISTS temp.primary_work_message; CREATE TEMP TABLE primary_work_message(event TEXT PRIMARY KEY, input TEXT NOT NULL, seq INTEGER NOT NULL, ordinal INTEGER NOT NULL, id TEXT NOT NULL, product BLOB NOT NULL) STRICT; DROP TABLE IF EXISTS temp.primary_work_parent; CREATE TEMP TABLE primary_work_parent(event TEXT PRIMARY KEY) STRICT;")?;
    let mut session_facts: Option<Session> = None;
    let mut sources = Vec::new();
    let native = Vec::new();
    let mut native_unavailable = Vec::new();
    let mut backgrounds = Vec::new();
    let mut query=index.prepare("SELECT event,product,state FROM primary_next WHERE kind='parsed' AND session IN (SELECT session FROM primary_work_physical) AND product IS NOT NULL ORDER BY input,chain,step_start,admitted,source_order,event")?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        let bytes: Vec<u8> = row.get(1)?;
        let fact = product(&bytes)?;
        if session(&fact).is_none_or(|session| !work.physical.contains(session)) {
            continue;
        }
        if row.get::<_, String>(2)? != "live" {
            continue;
        }
        let event: String = row.get(0)?;
        let reading = fact.reading().expect("parsed fact has reading");
        let chosen = if let Product::Message {
            facts, projection, ..
        } = &fact
        {
            selected(index, reading, facts, projection.as_ref())?
        } else {
            None
        };
        let contributes = match &fact {
            Product::Session { facts, .. } => {
                if &facts.id == id {
                    if let Some(session) = &mut session_facts {
                        session.merge(facts);
                    } else {
                        session_facts = Some(facts.clone());
                    }
                }
                true
            }
            Product::Message { facts, .. } => {
                if let Some((input, seq, ordinal)) = &chosen {
                    index.execute(
                        "INSERT OR REPLACE INTO primary_work_message VALUES (?1,?2,?3,?4,?5,?6)",
                        params![
                            event,
                            input,
                            seq,
                            ordinal,
                            facts.message.id.to_string(),
                            bytes
                        ],
                    )?;
                }
                chosen.is_some()
            }
            Product::Background { facts, .. } => {
                backgrounds.push(SelectedBackground {
                    logical_session: id.clone(),
                    update: facts.clone(),
                    event: event.clone(),
                });
                true
            }
            Product::Event { facts, .. } => {
                // Only actual structured native keys contribute attribution.
                if [
                    "repository",
                    "objectives",
                    "git_identity",
                    "terminal_identity",
                    "boot_id",
                    "native_unavailable",
                ]
                .iter()
                .any(|key| facts.data.get(*key).is_some())
                {
                    // SourceRef authenticates the receiver, but does not attest
                    // native acquisition rather than imported hook claims.
                    native_unavailable.push("native_acquisition_authority_unavailable".to_string());
                }
                true
            }
            _ => false,
        };
        // Source copies remain visible provenance even when another version wins.
        sources.push(ReadingSource {
            event: event.clone(),
            input: reading.input.id.clone(),
            chain: reading.chain,
            start: reading.start,
            end: reading.parsed_upto,
            selected: contributes,
            projection: match &fact {
                Product::Message { projection, .. } => projection.clone(),
                _ => None,
            },
        });
        index.execute(
            "INSERT OR IGNORE INTO primary_work_parent(event) VALUES (?1)",
            [event],
        )?;
    }
    let retired = false;
    let session = session_facts;
    let mut messages = Vec::new();
    let mut query = index.prepare(
        "SELECT event,product FROM primary_work_message ORDER BY input,seq,ordinal,id,event",
    )?;
    let mut rows = query.query([])?;
    let mut error = None;
    let values = std::iter::from_fn(|| {
        let result = (|| -> Result<Option<SelectedMessage>> {
            let Some(row) = rows.next()? else {
                return Ok(None);
            };
            let event: String = row.get(0)?;
            let Product::Message {
                facts, projection, ..
            } = product(&row.get::<_, Vec<u8>>(1)?)?
            else {
                unreachable!()
            };
            let message = facts.message;
            messages.push(WorkMessage {
                message: message.id.clone(),
                physical_session: message.session.clone(),
                role: message.role,
                written_by: WrittenBy::of(message.role, message.prompt_kind),
                prompt_kind: message.prompt_kind,
                at: message.at,
                at_evidence: message.time_evidence(),
                source_event: event.clone(),
            });
            let cost_nusd = message
                .reported_cost_usd
                .as_ref()
                .and_then(|value| value.as_f64())
                .and_then(|usd| {
                    let value = usd * 1_000_000_000.;
                    (value.is_finite() && value >= i64::MIN as f64 && value < i64::MAX as f64)
                        .then_some(value.round() as i64)
                });
            Ok(Some(SelectedMessage {
                logical_session: id.clone(),
                projection,
                message,
                event,
                cost_nusd,
            }))
        })();
        match result {
            Ok(value) => value,
            Err(failure) => {
                error = Some(failure);
                None
            }
        }
    });
    let (calls, turns) = work_products::derive_with_background(values, backgrounds);
    if let Some(error) = error {
        return Err(error);
    }
    let mut unavailable = if sources
        .iter()
        .any(|source| source.selected && source.projection.is_some())
    {
        vec!["selected_message_projection".into()]
    } else {
        Vec::new()
    };
    unavailable.extend(native_unavailable);
    unavailable.sort();
    unavailable.dedup();
    let snapshot = WorkSnapshot {
        reducer_version: work_products::VERSION,
        membership,
        position,
        session_id: id.clone(),
        session,
        retired,
        unavailable,
        messages,
        calls: calls.into_iter().map(|call| call.facts).collect(),
        turns: turns.into_iter().map(|turn| turn.facts).collect(),
        sources,
        native,
    };
    Ok(snapshot)
}

#[cfg(test)]
mod tests;

/// Internal indexed keys for retained canonical products, independent of UI references.
pub(crate) fn reference(product: &Product) -> Result<Option<String>> {
    let natural = match product {
        Product::WorkSnapshot { snapshot } => serde_json::json!([
            "work_snapshot",
            snapshot.session_id,
            snapshot.reducer_version
        ]),
        Product::WorkComponentStatus { status } => serde_json::json!([
            "work_component_status",
            status.component_kind,
            status.sessions().collect::<Vec<_>>()
        ]),
        _ => return Ok(None),
    };
    Ok(Some(format!(
        "canonical:{}",
        blake3::hash(&serde_json::to_vec(&natural)?).to_hex()
    )))
}

/// Canonical records are retained as typed facts, never fed back into raw rows.
pub(crate) fn validate_confirmed(product: &Product) -> Result<()> {
    let result = match product {
        Product::WorkSnapshot { snapshot } => snapshot.validate(),
        Product::WorkComponentStatus { status } => status.validate(),
        _ => return Ok(()),
    };
    result.map_err(|reason| StoreError::Invalid(reason.into()))
}

pub(crate) fn component_status(
    conn: &Connection,
    sessions: &[SessionId],
) -> Result<Option<work_products::ComponentStatus>> {
    super::scope::require_readable(conn)?;
    let state = super::projection::replay_state(conn)?;
    if state.availability != super::ProjectionAvailability::Certified
        || state.erasure_fence.is_some()
    {
        return Err(StoreError::Invalid(
            "canonical component status has no current owner projection".into(),
        ));
    }
    let mut sessions = sessions.to_vec();
    sessions.sort_by_key(ToString::to_string);
    sessions.dedup();
    if sessions.is_empty() {
        return Err(StoreError::Invalid(
            "a component needs physical session IDs".into(),
        ));
    }
    let natural = serde_json::json!([
        "work_component_status",
        work_products::component::ComponentKind::LogicalWork,
        sessions
    ]);
    let reference = format!(
        "canonical:{}",
        blake3::hash(&serde_json::to_vec(&natural)?).to_hex()
    );
    let mut query = conn.prepare("SELECT product FROM primary_fact WHERE reference=?1 AND state='live' AND native_context IS NOT NULL ORDER BY admitted,event")?;
    let mut rows = query.query([reference])?;
    let mut found = None;
    while let Some(row) = rows.next()? {
        let Product::WorkComponentStatus { status } = product(&row.get::<_, Vec<u8>>(0)?)? else {
            return Err(StoreError::Invalid(
                "canonical component reference has another product".into(),
            ));
        };
        status
            .validate()
            .map_err(|reason| StoreError::Invalid(reason.into()))?;
        if found.is_some() {
            return Err(StoreError::Invalid(
                "canonical component has conflicting live readings".into(),
            ));
        }
        found = Some(*status);
    }
    Ok(found)
}
