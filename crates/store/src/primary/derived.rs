//! Snapshot intents are part of the same transaction as the confirmed position.
//! Selection comes from that transaction's projection; payloads come from its
//! complete canonical facts. The publisher never scans the read index later.
use super::{Product, ProductWrite, Sinex};
use crate::{Result, StoreError};
use polylogue_model::work_products::{
    self, NativeEvidence, ReadingSource, SelectedMessage, WorkMessage, WorkSnapshot,
};
use polylogue_model::{Message, Session, SessionId, WrittenBy};
use rusqlite::{Connection, OptionalExtension, Transaction, params};

const SELECTION: &str = "polylogue-confirmed-selection-1";
pub(crate) const SCHEMA: &str = "
CREATE TABLE primary_derived (
 identity TEXT PRIMARY KEY, session TEXT NOT NULL, source_position INTEGER NOT NULL,
 body BLOB, cancelled_at INTEGER
) STRICT;
CREATE TABLE primary_derived_parent (
 identity TEXT NOT NULL REFERENCES primary_derived(identity) ON DELETE CASCADE,
 event TEXT NOT NULL, PRIMARY KEY(identity,event)
) STRICT, WITHOUT ROWID;
CREATE INDEX primary_derived_source ON primary_derived(session,source_position);
CREATE TABLE primary_work_dirty (session TEXT PRIMARY KEY) STRICT, WITHOUT ROWID;
";

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
/// Own snapshot confirmations only retire intents; they never become reducer input.
pub(crate) fn stage(
    tx: &Transaction<'_>,
    position: i64,
    counter: i64,
    publisher: bool,
) -> Result<()> {
    tx.execute_batch("DROP TABLE IF EXISTS temp.primary_work_changed; CREATE TEMP TABLE primary_work_changed(session TEXT PRIMARY KEY) STRICT;")?;
    tx.execute("INSERT OR IGNORE INTO primary_work_dirty SELECT session FROM primary_candidate WHERE kind='parsed' AND session IS NOT NULL UNION SELECT f.session FROM primary_fact f JOIN primary_candidate c ON c.event=f.event WHERE f.kind='parsed' AND f.session IS NOT NULL UNION SELECT f.session FROM primary_fact f JOIN primary_candidate c ON c.input=f.input WHERE c.kind='input' AND f.kind='parsed' AND f.session IS NOT NULL",[])?;
    tx.execute("INSERT OR IGNORE INTO primary_work_dirty SELECT id FROM session WHERE changed=?1 AND EXISTS(SELECT 1 FROM primary_candidate WHERE kind IN ('parsed','input','blob'))",[counter])?;
    let mut query=tx.prepare("SELECT product FROM primary_candidate WHERE kind='derived' AND state='live' AND product IS NOT NULL")?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        if let Product::WorkSnapshot { snapshot } = product(&row.get::<_, Vec<u8>>(0)?)? {
            let body = pack(&snapshot)?;
            tx.execute(
                "DELETE FROM primary_derived WHERE session=?1 AND source_position=?2 AND body=?3",
                params![snapshot.session_id.to_string(), snapshot.position, body],
            )?;
        }
    }
    drop(rows);
    drop(query);
    // A later observation may invalidate an intent before it was admitted.
    // Retain its exact identity until filesystem cleanup succeeds after commit.
    tx.execute("UPDATE primary_derived SET body=NULL,cancelled_at=?1 WHERE cancelled_at IS NULL AND EXISTS(SELECT 1 FROM primary_derived_parent p LEFT JOIN primary_next f ON f.event=p.event WHERE p.identity=primary_derived.identity AND coalesce(f.state,'missing')<>'live')",[position])?;
    if !publisher {
        return Ok(());
    }
    tx.execute("INSERT INTO primary_work_changed SELECT session FROM primary_work_dirty", [])?;
    let mut changed = tx.prepare("SELECT session FROM primary_work_changed ORDER BY session")?;
    let mut rows = changed.query([])?;
    while let Some(row) = rows.next()? {
        let id = SessionId::parse(&row.get::<_, String>(0)?)
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        if let Some(pk) = tx
            .query_row(
                "SELECT pk FROM session WHERE id=?1",
                [id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
        {
            crate::derive::select_confirmed(tx, pk)?;
        }
        let snapshot = snapshot(tx, &id, position)?;
        snapshot
            .validate()
            .map_err(|error| StoreError::Invalid(error.into()))?;
        let body = pack(&snapshot)?;
        let mut causal = blake3::Hasher::new();
        let mut parent_query =
            tx.prepare("SELECT event FROM primary_work_parent ORDER BY event")?;
        let mut parent_rows = parent_query.query([])?;
        while let Some(row) = parent_rows.next()? {
            let event: String = row.get(0)?;
            causal.update(&(event.len() as u64).to_le_bytes());
            causal.update(event.as_bytes());
        }
        let identity = format!(
            "work-snapshot:{}:{}:{}:{}",
            work_products::VERSION,
            SELECTION,
            position,
            blake3::hash(&serde_json::to_vec(&(
                &snapshot.session_id,
                causal.finalize().to_hex().as_str(),
                &body
            ))?)
            .to_hex()
        );
        tx.execute("INSERT OR IGNORE INTO primary_derived(identity,session,source_position,body) VALUES (?1,?2,?3,?4)",params![identity,id.to_string(),position,body])?;
        tx.execute("INSERT OR IGNORE INTO primary_derived_parent(identity,event) SELECT ?1,event FROM primary_work_parent",[identity])?;
        tx.execute("DELETE FROM primary_work_dirty WHERE session=?1", [id.to_string()])?;
    }
    Ok(())
}

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

fn snapshot(index: &Connection, id: &SessionId, position: i64) -> Result<WorkSnapshot> {
    index.execute_batch("DROP TABLE IF EXISTS temp.primary_work_message; CREATE TEMP TABLE primary_work_message(event TEXT PRIMARY KEY, input TEXT NOT NULL, seq INTEGER NOT NULL, ordinal INTEGER NOT NULL, id TEXT NOT NULL, product BLOB NOT NULL) STRICT; DROP TABLE IF EXISTS temp.primary_work_parent; CREATE TEMP TABLE primary_work_parent(event TEXT PRIMARY KEY) STRICT;")?;
    let mut session_facts: Option<Session> = None;
    let mut alive = false;
    let mut sources = Vec::new();
    let mut native = Vec::new();
    let mut backgrounds = Vec::new();
    let mut query=index.prepare("SELECT event,product,state FROM primary_next WHERE kind='parsed' AND session=?1 AND product IS NOT NULL ORDER BY input,chain,step_start,admitted,source_order,event")?;
    let mut rows = query.query([id.to_string()])?;
    while let Some(row) = rows.next()? {
        let bytes: Vec<u8> = row.get(1)?;
        let fact = product(&bytes)?;
        if session(&fact) != Some(id) {
            continue;
        }
        if row.get::<_, String>(2)? != "live" {
            continue;
        }
        alive = true;
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
                if let Some(session) = &mut session_facts {
                    session.merge(facts);
                } else {
                    session_facts = Some(facts.clone());
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
                backgrounds.push((facts.clone(), event.clone()));
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
                    if let Some(range) = &facts.range {
                        let location:Option<(String,u64)>=index.query_row("SELECT id,?3-offset FROM source_material WHERE input=?1 AND chain=?2 AND offset<=?3 AND offset+length>=?3 ORDER BY offset DESC LIMIT 1",params![reading.input.id,reading.chain,range.0],|row|Ok((row.get(0)?,row.get(1)?))).optional()?;
                        if let Some((material, anchor)) = location {
                            native.push(NativeEvidence {
                                event: event.clone(),
                                material,
                                anchor,
                                data: facts.data.clone(),
                            });
                        }
                    }
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
    let retired = !alive;
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
            let Product::Message { facts, .. } = product(&row.get::<_, Vec<u8>>(1)?)? else {
                unreachable!()
            };
            let message = facts.message;
            messages.push(WorkMessage {
                message: message.id.clone(),
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
    let unavailable = if sources
        .iter()
        .any(|source| source.selected && source.projection.is_some())
    {
        vec!["selected_message_projection".into()]
    } else {
        Vec::new()
    };
    let snapshot = WorkSnapshot {
        reducer_version: work_products::VERSION,
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

/// After the position transaction commits, publish immutable intents in order.
/// Removing one requires its confirmed product, never its publication receipt.
pub(crate) fn publish(index: &Connection, sinex: &dyn Sinex) -> Result<()> {
    let applied = super::projection::position(index)?;
    let mut query=index.prepare("SELECT identity,body,cancelled_at FROM primary_derived ORDER BY source_position,session,identity")?;
    let mut rows = query.query([])?;
    let mut discarded = Vec::new();
    while let Some(row) = rows.next()? {
        let identity: String = row.get(0)?;
        if let Some(position) = row.get::<_, Option<i64>>(2)? {
            sinex.discard_confirmed(
                &[identity.clone(), format!("{identity}:observation")],
                position,
                applied,
            )?;
            discarded.push(identity);
            continue;
        }
        let bytes: Vec<u8> = row.get(1)?;
        let snapshot: WorkSnapshot = crate::encoding::unpack_json(&bytes)?;
        let mut parents = index
            .prepare("SELECT event FROM primary_derived_parent WHERE identity=?1 ORDER BY event")?;
        let parents = parents
            .query_map([&identity], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        sinex.prepare_named_products(
            &identity,
            &[ProductWrite {
                product: Product::WorkSnapshot { snapshot },
                location: None,
                parents,
                machine: None,
                output_index: 0,
            }],
        )?;
    }
    drop(rows);
    drop(query);
    for identity in discarded {
        index.execute("DELETE FROM primary_derived WHERE identity=?1", [identity])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
