//! The primary cache applies complete confirmed positions, never producer acknowledgments.
pub(crate) const SCHEMA: &str = concat!("
CREATE TABLE confirmed_event (
 event TEXT PRIMARY KEY, committed INTEGER NOT NULL, changed INTEGER NOT NULL, source_order INTEGER NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('live','archived','tombstoned')),
 product BLOB, input TEXT, chain INTEGER, reference TEXT, kind TEXT NOT NULL, step_start INTEGER, step_end INTEGER, reading_version TEXT, session TEXT, sinex_context BLOB, value_hash BLOB, material TEXT, record_hash TEXT, sinex_kind TEXT
) STRICT;
CREATE INDEX confirmed_event_chain ON confirmed_event(input,chain,committed,event);
CREATE INDEX confirmed_event_session ON confirmed_event(session,kind,state,committed);
CREATE INDEX confirmed_event_live_material ON confirmed_event(material,event) WHERE state='live';
CREATE INDEX confirmed_event_value ON confirmed_event(input,chain,reading_version,reference,value_hash,state);
CREATE INDEX confirmed_event_reading_progress ON confirmed_event(input,chain,reading_version,step_start,step_end) WHERE kind='reading_progress';
", include_str!("confirmed/read_indexes.sql"), "
CREATE TABLE confirmed_event_blob (
 event TEXT NOT NULL REFERENCES confirmed_event(event), hash BLOB NOT NULL,
 PRIMARY KEY(event,hash)
) STRICT, WITHOUT ROWID;
CREATE INDEX confirmed_event_blob_hash ON confirmed_event_blob(hash);
CREATE TABLE primary_pending (
 n INTEGER PRIMARY KEY, position INTEGER NOT NULL, event TEXT NOT NULL,
 change TEXT NOT NULL, product BLOB, source_order INTEGER NOT NULL, sinex_context BLOB,
 UNIQUE(position,event,change)
) STRICT;
CREATE INDEX primary_pending_position ON primary_pending(position,n);
CREATE TABLE primary_complete(position INTEGER PRIMARY KEY, marker BLOB) STRICT;
", include_str!("../schema/primary_refused.sql"), "
CREATE TABLE source_material (
 id TEXT NOT NULL, event TEXT NOT NULL, input TEXT, chain INTEGER, offset INTEGER,
 length INTEGER NOT NULL, hash BLOB NOT NULL, source_length INTEGER, blob_offset INTEGER NOT NULL,
 PRIMARY KEY(event,id)
) STRICT;
CREATE INDEX source_material_identity ON source_material(id);
CREATE INDEX source_material_place ON source_material(input,chain,offset);
", include_str!("confirmed/reference_index.sql"));

use super::{Confirmed, ConfirmedChange, Material, Product};
use crate::ConnExt as _;
mod replay;
use crate::{Result, StoreError};
pub(crate) use replay::save as save_replay;
pub(crate) use replay::{has_confirmed_event, replay_state};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, BTreeSet};

fn packed(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(crate::encoding::pack_bytes(&serde_json::to_vec(value)?)?)
}
fn unpacked<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    crate::encoding::unpack_json(bytes)
}

pub(crate) fn checkpoint(index: &Connection) -> Result<Option<i64>> {
    index.scalar_optional(
        "SELECT CAST(value AS INTEGER) FROM meta WHERE key='confirmed_position'",
        [],
    )
}
pub(crate) fn position(index: &Connection) -> Result<i64> {
    Ok(checkpoint(index)?.unwrap_or(0))
}

/// Each staged page may be acknowledged. Only a complete position is visible.
fn stage(index: &mut Connection, events: &[Confirmed]) -> Result<()> {
    let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let applied = position(&tx)?;
    let mut state = replay_state(&tx)?;
    for event in events {
        if let ConfirmedChange::Control(control) = &event.change {
            if !event.event.is_empty() || event.sinex_context.is_some() || event.output_index != 0 {
                return Err(StoreError::Invalid(
                    "stream controls cannot carry an event identity".into(),
                ));
            }
            replay::stage_control(&tx, &mut state, control)?;
            continue;
        }
        if let Some(context) = &event.sinex_context {
            replay::validate_context(context)?;
        }
        if event.position <= 0 {
            return Err(StoreError::Invalid(
                "a confirmed data position must be positive".into(),
            ));
        }
        if event.position <= applied {
            continue;
        }
        if event.event.is_empty() {
            return Err(StoreError::Invalid(
                "a confirmed event needs its Sinex event identity".into(),
            ));
        }
        let change = match &event.change {
            ConfirmedChange::Live(_) => "live",
            ConfirmedChange::Archived => "archived",
            ConfirmedChange::Restored => "restored",
            ConfirmedChange::Tombstoned => "tombstoned",
            ConfirmedChange::Refused { .. } => "refused",
            ConfirmedChange::Control(_) => unreachable!(),
        };
        if let ConfirmedChange::Live(product) = &event.change
            && product.reading().is_some()
            && !matches!(product.as_ref(), Product::Interpreted { .. })
        {
            return Err(StoreError::Invalid(
                "confirmed source events need their source record identity".into(),
            ));
        }
        let bytes = packed(&event.change)?;
        let context_bytes = event.sinex_context.as_ref().map(packed).transpose()?;
        let source_order = i64::try_from(event.output_index).map_err(|_| {
            StoreError::Invalid(
                "a source output position exceeds SQLite's signed integer range".into(),
            )
        })?;
        let existing:Option<(Vec<u8>,i64,Option<Vec<u8>>)>=tx.query_row("SELECT product,source_order,sinex_context FROM primary_pending WHERE position=?1 AND event=?2 AND change=?3",params![event.position,event.event,change],|row| <(_, _, _)>::try_from(row)).optional()?;
        match existing {
            Some((old, order, context)) => {
                if order != source_order || old != bytes || context != context_bytes {
                    return Err(StoreError::Invalid(
                        "a redelivered confirmed change has different contents".into(),
                    ));
                }
            }
            None => {
                tx.execute("INSERT INTO primary_pending(position,event,change,product,source_order,sinex_context) VALUES (?1,?2,?3,?4,?5,?6)",params![event.position,event.event,change,bytes,source_order,context_bytes])?;
            }
        }
    }
    replay::save(&tx, &state)?;
    crate::fault::check("primary.stage_commit", "")?;
    tx.commit()?;
    Ok(())
}
#[derive(Clone)]
struct ConfirmedEventRow {
    event: String,
    committed: i64,
    changed: i64,
    order: i64,
    state: String,
    product: Option<Vec<u8>>,
    input: Option<String>,
    chain: Option<i64>,
    reference: Option<String>,
    kind: String,
    start: Option<i64>,
    end: Option<i64>,
    version: Option<String>,
    session: Option<String>,
    sinex_context: Option<Vec<u8>>,
    value_hash: Option<Vec<u8>>,
    material: Option<String>,
    record_hash: Option<String>,
    sinex_kind: Option<String>,
}
impl ConfirmedEventRow {
    fn read(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            event: row.get(0)?,
            committed: row.get(1)?,
            changed: row.get(2)?,
            order: row.get(3)?,
            state: row.get(4)?,
            product: row.get(5)?,
            input: row.get(6)?,
            chain: row.get(7)?,
            reference: row.get(8)?,
            kind: row.get(9)?,
            start: row.get(10)?,
            end: row.get(11)?,
            version: row.get(12)?,
            session: row.get(13)?,
            sinex_context: row.get(14)?,
            value_hash: row.get(15)?,
            material: row.get(16)?,
            record_hash: row.get(17)?,
            sinex_kind: row.get(18)?,
        })
    }
    fn insert(&self, index: &Connection, table: &str) -> Result<()> {
        index.execute(&format!("INSERT OR REPLACE INTO {table}(event,committed,changed,source_order,state,product,input,chain,reference,kind,step_start,step_end,reading_version,session,sinex_context,value_hash,material,record_hash,sinex_kind) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)"),params![self.event,self.committed,self.changed,self.order,self.state,self.product,self.input,self.chain,self.reference,self.kind,self.start,self.end,self.version,self.session,self.sinex_context,self.value_hash,self.material,self.record_hash,self.sinex_kind])?;
        Ok(())
    }
    fn tombstoned(event: String, position: i64, order: i64) -> Self {
        Self {
            event,
            committed: position,
            changed: position,
            order,
            state: "tombstoned".into(),
            product: None,
            input: None,
            chain: None,
            reference: None,
            kind: "tombstoned".into(),
            start: None,
            end: None,
            version: None,
            session: None,
            sinex_context: None,
            value_hash: None,
            material: None,
            record_hash: None,
            sinex_kind: None,
        }
    }
    // The delivered event has lifecycle identity even when its body is unavailable.
    fn unreadable(event: String, position: i64, order: i64) -> Self {
        let mut row = Self::tombstoned(event, position, order);
        row.state = "live".into();
        row.kind = "refused".into();
        row
    }
    fn product(&self) -> Result<Option<Product>> {
        self.product.as_deref().map(unpacked).transpose()
    }
}

fn metadata(product: &Product) -> (Option<String>, Option<i64>, Option<String>, &'static str) {
    if let Product::Interpreted { reading, record } = product {
        return (
            Some(reading.input.id.clone()),
            Some(reading.chain),
            Some(record.reference.clone()),
            if record.family == "parse_progress" {
                "reading_progress"
            } else {
                "reading"
            },
        );
    }
    let reference = match product {
        Product::AttachmentBinding { record } => {
            Some(format!("attachment-binding:{}", record.context().message))
        }
        Product::Request(request) => Some(format!("request:{}", request.request.id)),
        Product::RestoredRequest { prepared, .. } => {
            Some(format!("request:{}", prepared.request.id))
        }
        Product::RestoredAssertion { fact, .. } => Some(fact.reference().stored_form()),
        Product::Coverage(coverage) => Some(format!("coverage:{}", coverage.machine)),
        Product::Query { object } => Some(format!("query:{}", object.id)),
        Product::QueryRun { run } => Some(format!("query-run:{}", run.id)),
        Product::QueryRunOutcome { outcome } => Some(format!("query-run-outcome:{}", outcome.run)),
        _ => None,
    };
    match product {
        Product::Input { rows, .. } => (
            Some(rows.input.clone()),
            rows.extents
                .first()
                .map(|e| e.chain)
                .or_else(|| rows.revision.as_ref().map(|r| r.chain)),
            reference,
            "input",
        ),
        Product::AttachmentBinding { record } => (
            Some(record.context().input.clone()),
            Some(record.context().chain),
            reference,
            "attachment_binding",
        ),
        Product::Blob(_) => (None, None, reference, "blob"),
        Product::Request(_)
        | Product::RestoredRequest { .. }
        | Product::RestoredAssertion { .. } => (None, None, reference, "request"),
        Product::Coverage(_) => (None, None, reference, "coverage"),
        Product::Query { .. } | Product::QueryRun { .. } | Product::QueryRunOutcome { .. } => {
            (None, None, reference, "query")
        }
        _ => unreachable!(),
    }
}

/// Staged events are kept on disk and hidden from domain reads.
fn stage_events(index: &Connection, through: i64) -> Result<()> {
    index.execute_batch("DROP VIEW IF EXISTS temp.primary_next; DROP TABLE IF EXISTS temp.primary_staged;
        CREATE TEMP TABLE primary_staged AS SELECT * FROM confirmed_event WHERE 0;
        CREATE UNIQUE INDEX temp.primary_staged_event ON primary_staged(event);
        CREATE INDEX temp.primary_staged_chain ON primary_staged(input,chain);
        CREATE TEMP VIEW primary_next AS SELECT * FROM primary_staged UNION ALL SELECT * FROM main.confirmed_event f WHERE NOT EXISTS(SELECT 1 FROM primary_staged c WHERE c.event=f.event);")?;
    let mut query = index.prepare(
        "SELECT event,product,source_order,position,sinex_context FROM primary_pending WHERE position<=?1 ORDER BY position,CASE change WHEN 'live' THEN 0 WHEN 'archived' THEN 1 WHEN 'restored' THEN 1 ELSE 2 END,n",
    )?;
    let mut rows = query.query([through])?;
    while let Some(row) = rows.next()? {
        let event: String = row.get(0)?;
        let bytes: Vec<u8> = row.get(1)?;
        let order: i64 = row.get(2)?;
        let position: i64 = row.get(3)?;
        let context_bytes: Option<Vec<u8>> = row.get(4)?;
        let context: Option<super::SinexEventContext> =
            context_bytes.as_deref().map(unpacked).transpose()?;
        let sinex_material = context
            .as_ref()
            .and_then(|context| context.location.as_ref())
            .map(|location| location.material.clone());
        let sinex_record_hash = context
            .as_ref()
            .and_then(|context| context.record_hash.clone());
        let sinex_kind = context.as_ref().map(|context| context.kind.name.clone());
        let change: ConfirmedChange = unpacked(&bytes)?;
        let old = index
            .query_row(
                "SELECT * FROM primary_next WHERE event=?1",
                [&event],
                ConfirmedEventRow::read,
            )
            .optional()?;
        let next = match change {
            ConfirmedChange::Live(product) => {
                let bytes = packed(&product)?;
                if let Some(old) = &old {
                    if old.state == "tombstoned" {
                        return Err(StoreError::Invalid(
                            "a tombstoned Sinex event cannot become live".into(),
                        ));
                    }
                    if old.kind != "refused"
                        && (old.product.as_ref() != Some(&bytes)
                            || old.sinex_context != context_bytes)
                    {
                        return Err(StoreError::Invalid(
                            "a Sinex event changed its product".into(),
                        ));
                    }
                }
                let (input, chain, reference, kind) = metadata(&product);
                let session = match product.as_ref() {
                    Product::Interpreted { record, .. } => record.session.clone(),
                    _ => None,
                };
                let reading = product.reading();
                let start = reading
                    .map(|r| i64::try_from(r.start))
                    .transpose()
                    .map_err(|_| {
                        StoreError::Invalid(
                            "a reading range exceeds SQLite's signed integer range".into(),
                        )
                    })?;
                let end = reading
                    .map(|r| i64::try_from(r.parsed_upto))
                    .transpose()
                    .map_err(|_| {
                        StoreError::Invalid(
                            "a reading range exceeds SQLite's signed integer range".into(),
                        )
                    })?;
                let version = reading.map(|r| r.reading_version.clone());
                let value_hash = match product.as_ref() {
                    Product::Interpreted { record, .. } => Some(record.value_hash.0.to_vec()),
                    _ => None,
                };
                Some(ConfirmedEventRow {
                    event,
                    committed: old.as_ref().map_or(position, |f| f.committed),
                    changed: position,
                    order,
                    state: "live".into(),
                    product: Some(bytes),
                    input,
                    chain,
                    reference,
                    kind: kind.into(),
                    start,
                    end,
                    version,
                    session,
                    sinex_context: context_bytes,
                    value_hash,
                    material: sinex_material,
                    record_hash: sinex_record_hash,
                    sinex_kind,
                })
            }
            ConfirmedChange::Archived => old.map(|mut row| {
                row.state = "archived".into();
                row.changed = position;
                row
            }),
            ConfirmedChange::Restored => match old {
                Some(mut row)
                    if row.state != "tombstoned"
                        && (row.product.is_some() || row.kind == "refused") =>
                {
                    row.state = "live".into();
                    row.changed = position;
                    Some(row)
                }
                Some(_) => {
                    return Err(StoreError::Invalid(
                        "a tombstoned Sinex event cannot be restored".into(),
                    ));
                }
                None => None,
            },
            // An event erased before this index read it is recorded as tombstoned.
            ConfirmedChange::Tombstoned => Some(match old {
                Some(mut row) => {
                    row.state = "tombstoned".into();
                    row.product = None;
                    row.changed = position;
                    row
                }
                None => ConfirmedEventRow::tombstoned(event, position, order),
            }),
            ConfirmedChange::Refused { .. } => {
                let mut row =
                    old.unwrap_or_else(|| ConfirmedEventRow::unreadable(event, position, order));
                if row.sinex_context.is_none() {
                    row.sinex_context = context_bytes;
                    row.material = sinex_material;
                    row.record_hash = sinex_record_hash;
                    row.sinex_kind = sinex_kind;
                }
                Some(row)
            }
            ConfirmedChange::Control(_) => None,
        };
        if let Some(next) = next {
            next.insert(index, "primary_staged")?;
        }
    }
    Ok(())
}

fn material(product: &Product) -> Option<&Material> {
    match product {
        Product::Input { material, .. } => material.as_ref(),
        Product::Blob(material) => Some(material),
        _ => None,
    }
}
pub(crate) fn material_at(
    index: &Connection,
    input: &str,
    chain: i64,
    start: u64,
) -> Result<Option<super::SourceLocation>> {
    let start = i64::try_from(start).map_err(|_| {
        StoreError::Invalid("a source record position exceeds SQLite's signed integer range".into())
    })?;
    Ok(index.query_row("SELECT id,offset,blob_offset FROM source_material WHERE input=?1 AND chain=?2 AND offset<=?3 AND (offset+source_length>?3 OR (source_length=0 AND offset=?3)) ORDER BY offset DESC LIMIT 1",params![input,chain,start],|row|Ok(super::SourceLocation{material:row.get(0)?,anchor:(start-row.get::<_,i64>(1)?+row.get::<_,i64>(2)?) as u64})).optional()?)
}

struct Plan {
    retractions: Vec<crate::Batch>,
    inputs: BTreeSet<String>,
    materials: BTreeSet<polylogue_model::BlobHash>,
    requests: StateChange,
    queries: StateChange,
    binding_messages: BTreeSet<String>,
    coverage: StateChange,
}

/// How authored, query or coverage state takes one position's events.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StateChange {
    Unchanged,
    /// Only new live events: each is applied after the ones already applied.
    Appended,
    /// A lifecycle change touched an applied event: the state is cleared and
    /// every live event is applied again in commit order.
    Rewritten,
}

impl StateChange {
    fn of(index: &Connection, kind: &str) -> Result<Self> {
        // A formerly unreadable event can precede already-applied authored state.
        let rewritten: bool = index.exists("SELECT EXISTS(SELECT 1 FROM primary_staged c JOIN main.confirmed_event f ON f.event=c.event WHERE (f.kind=?1 AND f.state!=c.state) OR (f.kind='refused' AND c.kind=?1))", [kind])?;
        if rewritten {
            return Ok(Self::Rewritten);
        }
        let appended: bool = index.exists("SELECT EXISTS(SELECT 1 FROM primary_staged c WHERE c.kind=?1 AND c.state='live' AND NOT EXISTS(SELECT 1 FROM main.confirmed_event f WHERE f.event=c.event))",
            [kind])?;
        Ok(if appended {
            Self::Appended
        } else {
            Self::Unchanged
        })
    }

    /// The events to apply, in commit order, or `None` when nothing changed.
    fn selection(self, kind: &str) -> Option<String> {
        match self {
            Self::Unchanged => None,
            Self::Appended => Some(format!(
                "SELECT c.product,c.committed,c.source_order FROM primary_staged c WHERE c.kind='{kind}' AND c.state='live' AND NOT EXISTS(SELECT 1 FROM main.confirmed_event f WHERE f.event=c.event) ORDER BY c.committed,c.source_order,c.event"
            )),
            Self::Rewritten => Some(format!(
                "SELECT product,committed,source_order FROM primary_next WHERE kind='{kind}' AND state='live' ORDER BY committed,source_order,event"
            )),
        }
    }
}

fn plan(index: &Connection) -> Result<Plan> {
    let mut out = Plan {
        retractions: Vec::new(),
        inputs: BTreeSet::new(),
        materials: BTreeSet::new(),
        requests: StateChange::of(index, "request")?,
        queries: StateChange::of(index, "query")?,
        binding_messages: BTreeSet::new(),
        coverage: StateChange::of(index, "coverage")?,
    };
    let mut chains = BTreeMap::<(String, i64), (crate::InputKey, u64)>::new();
    let mut query = index.prepare("SELECT * FROM primary_staged UNION ALL SELECT f.* FROM confirmed_event f JOIN primary_staged c ON c.event=f.event")?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        let event_row = ConfirmedEventRow::read(row)?;
        if let Some(product) = event_row.product()? {
            if let Product::AttachmentBinding { record } = &product {
                out.binding_messages
                    .insert(record.context().message.to_string());
            }
            if let Product::Input { rows, .. } = &product {
                out.inputs.insert(rows.input.clone());
                if event_row.changed != event_row.committed {
                    let mut states = index.prepare("SELECT p.chain,i.format,coalesce((SELECT max(length) FROM source_revision WHERE input=i.id AND chain=p.chain),0) FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1")?;
                    for row in states
                        .query_map([&rows.input], |row| <(i64, String, u64)>::try_from(row))?
                    {
                        let (chain, format, length) = row?;
                        chains.insert(
                            (rows.input.clone(), chain),
                            (
                                crate::InputKey {
                                    id: rows.input.clone(),
                                    format,
                                    route: polylogue_model::Route::Live,
                                },
                                length,
                            ),
                        );
                    }
                }
            }
            if let Some(material) = material(&product) {
                out.materials.insert(material.hash);
                out.materials.extend(
                    material
                        .descriptors()?
                        .into_iter()
                        .map(|descriptor| polylogue_model::BlobHash(descriptor.hash.0)),
                );
            }
        }
    }
    for hash in &out.materials {
        let mut affected = index.prepare("SELECT f.input,f.chain,i.format,p.parsed_upto FROM confirmed_event_blob b JOIN confirmed_event f ON f.event=b.event JOIN input_ref i ON i.id=f.input JOIN parse_state p ON p.input_pk=i.pk AND p.chain=f.chain WHERE b.hash=?1")?;
        for row in affected.query_map([hash.0.as_slice()], |row| {
            <(String, i64, String, u64)>::try_from(row)
        })? {
            let (input, chain, format, length) = row?;
            chains.insert(
                (input.clone(), chain),
                (
                    crate::InputKey {
                        id: input,
                        format,
                        route: polylogue_model::Route::Live,
                    },
                    length,
                ),
            );
        }
    }
    for ((_, chain), (input, length)) in chains {
        let mut absent = crate::ParseStep::new(Default::default(), Default::default(), 0..0, false);
        absent.source_prefix_missing = true;
        out.retractions.push(crate::Batch {
            required_outcome: None,
            input,
            change: crate::Change::Reparse { chain },
            observation: None,
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length,
            parse: vec![absent],
            acquisition: None,
            basis: None,
            release: Default::default(),
        });
    }
    Ok(out)
}

fn store_material(tx: &Connection, event: &str, product: &Product) -> Result<()> {
    let Some(material) = material(product) else {
        return Ok(());
    };
    let descriptors = material.descriptors()?;
    let (input, chain, offset, source_length, blob_offset) = match product {
        Product::Input { rows, .. } => {
            let first = rows.extents.first().ok_or_else(|| {
                StoreError::Invalid("a material input has no source extent".into())
            })?;
            let blob_offset = i64::try_from(material.offset_of(&rows.extents)?).map_err(|_| {
                StoreError::Invalid("material offset exceeds the source catalog range".into())
            })?;
            (
                Some(rows.input.as_str()),
                Some(first.chain),
                Some(first.offset),
                Some(rows.extents.iter().map(|extent| extent.length).sum::<i64>()),
                blob_offset,
            )
        }
        _ => (None, None, None, None, 0),
    };
    let length = i64::try_from(material.length).map_err(|_| {
        StoreError::Invalid("material length exceeds the source catalog range".into())
    })?;
    tx.execute("INSERT OR REPLACE INTO source_material(id,event,input,chain,offset,length,hash,source_length,blob_offset) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",params![material.id,event,input,chain,offset,length,&material.hash.0[..],source_length,blob_offset])?;
    // Source extents cite the blobs that hold the bytes: the material's own,
    // or its chunks'.
    for descriptor in descriptors {
        let (Ok(size), Ok(stored)) = (
            i64::try_from(descriptor.decoded_len),
            i64::try_from(descriptor.len),
        ) else {
            return Err(StoreError::Invalid(
                "extent length exceeds the source catalog range".into(),
            ));
        };
        tx.execute("INSERT INTO source_blob(hash,size,stored_size,storage) VALUES (?1,?2,?3,?4) ON CONFLICT(hash) DO UPDATE SET storage=excluded.storage,stored_size=excluded.stored_size",params![&descriptor.hash.0[..],size,stored,descriptor.encode()])?;
    }
    Ok(())
}

fn sources(tx: &rusqlite::Transaction<'_>, plan: &Plan) -> Result<()> {
    // Material catalog changes precede every source extent citing them.
    let mut q = tx.prepare("SELECT event,product,state FROM primary_staged")?;
    let mut rows = q.query([])?;
    while let Some(row) = rows.next()? {
        let event: String = row.get(0)?;
        let bytes: Option<Vec<u8>> = row.get(1)?;
        let state: String = row.get(2)?;
        tx.execute("DELETE FROM source_material WHERE event=?1", [&event])?;
        if state == "live"
            && let Some(bytes) = bytes
        {
            store_material(tx, &event, &unpacked(&bytes)?)?;
        }
    }
    drop(rows);
    drop(q);
    for input in &plan.inputs {
        tx.execute("DELETE FROM source_observation_container WHERE observation_id IN (SELECT id FROM source_observation WHERE input=?1)",[input])?;
        tx.execute("DELETE FROM source_observation WHERE input=?1", [input])?;
        tx.execute("DELETE FROM source_extent WHERE input=?1", [input])?;
        tx.execute("DELETE FROM source_revision WHERE input=?1", [input])?;
        tx.execute("DELETE FROM source_input WHERE id=?1", [input])?;
        let mut q=tx.prepare("SELECT product,committed FROM primary_next WHERE state='live' AND kind='input' AND input=?1 ORDER BY committed,source_order,event")?;
        let mut rows = q.query([input])?;
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(0)?;
            let committed: i64 = row.get(1)?;
            if let Product::Input { rows, .. } = unpacked(&bytes)? {
                rows.insert_confirmed(tx, committed)?;
            }
        }
    }
    for hash in &plan.materials {
        tx.execute("DELETE FROM source_blob WHERE hash=?1 AND NOT EXISTS(SELECT 1 FROM source_material WHERE hash=?1) AND NOT EXISTS(SELECT 1 FROM source_extent WHERE blob=?1)",[&hash.0[..]])?;
    }
    Ok(())
}

fn members(tx: &rusqlite::Transaction<'_>, plan: &Plan) -> Result<()> {
    for input in &plan.inputs {
        crate::assets::visit_import_members(tx, "source_", 0, Some(input), |member| {
            crate::assets::project_member(tx, &member, None, None)
        })?;
    }
    Ok(())
}

fn bindings(tx: &rusqlite::Transaction<'_>, plan: &Plan) -> Result<()> {
    for message in &plan.binding_messages {
        tx.execute("DELETE FROM attachment_binding WHERE message=?1", [message])?;
        let reference = format!("attachment-binding:{message}");
        let mut query = tx.prepare("SELECT product FROM primary_next WHERE state='live' AND kind='attachment_binding' AND reference=?1 ORDER BY committed,source_order,event")?;
        let mut rows = query.query([reference])?;
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(0)?;
            if let Product::AttachmentBinding { record } = unpacked(&bytes)? {
                crate::assets::bindings::project(tx, tx, &record)?;
            }
        }
    }
    Ok(())
}

fn requests(tx: &rusqlite::Transaction<'_>, plan: &Plan) -> Result<()> {
    let Some(selection) = plan.requests.selection("request") else {
        return Ok(());
    };
    if plan.requests == StateChange::Rewritten {
        super::authored::clear(tx)?;
    }
    let mut q = tx.prepare(&selection)?;
    let mut rows = q.query([])?;
    while let Some(row) = rows.next()? {
        let bytes: Vec<u8> = row.get(0)?;
        let committed: i64 = row.get(1)?;
        match unpacked(&bytes)? {
            Product::Request(prepared) => super::authored::apply(tx, &prepared, committed)?,
            Product::RestoredRequest { prepared } => {
                super::authored::restore(tx, &prepared, committed)?;
            }
            Product::RestoredAssertion { fact, import } => {
                crate::assertions::apply(tx, std::slice::from_ref(&fact), committed)?;
                if let Some(import) = import {
                    crate::assertions::imports::record(tx, &fact.id, &import, committed)?;
                }
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn queries(tx: &rusqlite::Transaction<'_>, plan: &Plan) -> Result<()> {
    let Some(selection) = plan.queries.selection("query") else {
        return Ok(());
    };
    if plan.queries == StateChange::Rewritten {
        crate::query_objects::clear(tx)?;
    }
    let mut query = tx.prepare(&selection)?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        let bytes: Vec<u8> = row.get(0)?;
        match unpacked(&bytes)? {
            Product::Query { object } => {
                crate::query_objects::apply_object(tx, &object, row.get(1)?)?
            }
            Product::QueryRun { run } => {
                crate::query_objects::apply_run(tx, &run, row.get(1)?, row.get::<_, u64>(2)?)?;
            }
            Product::QueryRunOutcome { outcome } => {
                crate::query_objects::apply_run_outcome(tx, &outcome)?
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn coverage(tx: &rusqlite::Transaction<'_>, plan: &Plan) -> Result<()> {
    let Some(selection) = plan.coverage.selection("coverage") else {
        return Ok(());
    };
    if plan.coverage == StateChange::Rewritten {
        crate::coverage::clear(tx)?;
    }
    let mut query = tx.prepare(&selection)?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        let bytes: Vec<u8> = row.get(0)?;
        if let Product::Coverage(coverage) = unpacked(&bytes)? {
            crate::coverage::apply(tx, &coverage)?;
        }
    }
    Ok(())
}

/// Events Polylogue could not read are recorded with their kind, position
/// and reason; the stream continues past them.
fn refusals(tx: &rusqlite::Transaction<'_>, through: i64) -> Result<()> {
    let mut query = tx.prepare(
        "SELECT event,position,product FROM primary_pending WHERE change='refused' AND position<=?1",
    )?;
    let mut rows = query.query([through])?;
    while let Some(row) = rows.next()? {
        let event: String = row.get(0)?;
        let position: i64 = row.get(1)?;
        let bytes: Vec<u8> = row.get(2)?;
        if let ConfirmedChange::Refused {
            kind,
            reason,
            unreadable_input,
        } = unpacked(&bytes)?
        {
            tx.execute(
                "INSERT INTO primary_refused(event,position,kind,reason,unreadable_input) VALUES (?1,?2,?3,?4,?5) ON CONFLICT(event) DO UPDATE SET position=excluded.position,kind=excluded.kind,reason=excluded.reason,unreadable_input=excluded.unreadable_input",
                params![event, position, kind, reason, unreadable_input],
            )?;
        }
    }
    tx.execute("DELETE FROM primary_refused WHERE EXISTS(SELECT 1 FROM primary_next f WHERE f.event=primary_refused.event AND (f.product IS NOT NULL OR f.state='tombstoned'))", [])?;
    Ok(())
}

fn remember_blobs(tx: &rusqlite::Transaction<'_>) -> Result<()> {
    tx.execute(
        "DELETE FROM confirmed_event_blob WHERE event IN (SELECT event FROM primary_staged)",
        [],
    )?;
    let mut query =
        tx.prepare("SELECT event,product FROM primary_staged WHERE product IS NOT NULL")?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        let event: String = row.get(0)?;
        let bytes: Vec<u8> = row.get(1)?;
        let product: Product = unpacked(&bytes)?;
        let mut hashes = BTreeSet::new();
        if let Product::Interpreted { record, .. } = product {
            hashes.extend(record.blobs);
        }
        for hash in hashes {
            tx.execute(
                "INSERT INTO confirmed_event_blob(event,hash) VALUES (?1,?2)",
                params![event, &hash.0[..]],
            )?;
        }
    }
    Ok(())
}

/// The next complete position, prepared outside the serial writer. The pending
/// rows stay the saved copy until application commits.
pub(crate) struct Preparation {
    position: i64,
    state: super::ReplayState,
    generation: Option<String>,
    pending: Option<i64>,
    completion: Option<Vec<u8>>,
    pub(crate) retractions: Vec<crate::Batch>,
}

pub(crate) enum Step {
    Complete(i64),
    Prepare(Box<Preparation>),
}

fn generation(index: &Connection) -> Result<Option<String>> {
    crate::meta_get(index, "generation")
}

fn pending(index: &Connection) -> Result<Option<i64>> {
    index.scalar("SELECT max(n) FROM primary_pending", [])
}

fn next_position(index: &Connection) -> Result<Option<i64>> {
    index.scalar(
        "SELECT min(position) FROM primary_complete WHERE position>=?1",
        [checkpoint(index)?.unwrap_or(-1)],
    )
}

fn completion(index: &Connection, position: i64) -> Result<Option<Vec<u8>>> {
    index.scalar(
        "SELECT marker FROM primary_complete WHERE position=?1",
        [position],
    )
}

fn next(index: &Connection) -> Result<Step> {
    let position = next_position(index)?;
    let Some(position) = position else {
        return Ok(Step::Complete(self::position(index)?));
    };
    stage_events(index, position)?;
    Ok(Step::Prepare(Box::new(Preparation {
        position,
        state: replay_state(index)?,
        generation: generation(index)?,
        pending: pending(index)?,
        completion: completion(index, position)?,
        retractions: plan(index)?.retractions,
    })))
}

/// Each bounded page is saved before its worker waits for overlapping key locks.
pub(crate) fn stage_next(index: &mut Connection, events: &[Confirmed]) -> Result<Step> {
    stage(index, events)?;
    next(index)
}

/// Recheck Sinex's state and the local chain set, then commit lifecycle,
/// source retractions and completion together. No key-lock wait runs on this thread.
pub(crate) fn apply_prepared(
    texts: &crate::text_rows::TextIds,
    index: &mut Connection,
    prepared: Preparation,
) -> Result<Step> {
    let text_transaction = texts.transaction();
    let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
    super::scope::require_readable(&tx)?;
    // Staged rows never change: redelivery compares their entire contents.
    // Applied progress and the pending row boundary therefore fix Sinex's
    // admission, live state and value hashes without another copy of the events.
    if next_position(&tx)? != Some(prepared.position)
        || completion(&tx, prepared.position)? != prepared.completion
        || replay_state(&tx)? != prepared.state
        || generation(&tx)? != prepared.generation
        || pending(&tx)? != prepared.pending
    {
        return Err(StoreError::ScopeChanged);
    }
    let next = prepared.position;
    stage_events(&tx, next)?;
    let mut plan = plan(&tx)?;
    if plan.retractions.len() != prepared.retractions.len()
        || !plan
            .retractions
            .iter()
            .zip(&prepared.retractions)
            .all(|(current, prepared)| {
                current.input == prepared.input
                    && current.change == prepared.change
                    && current.length == prepared.length
            })
    {
        return Err(StoreError::ScopeChanged);
    }
    plan.retractions = prepared.retractions;
    let counter = crate::archive::bump_counter(&tx)?;
    // Lifecycle changes can remove or restore sources committed before a
    // position fixed before I/O. New commits keep that position valid; changes
    // to existing rows do not.
    let source_lifecycle: bool = tx.exists(
        "SELECT EXISTS(SELECT 1 FROM primary_staged WHERE kind='input' AND changed!=committed)",
        [],
    )?;
    if source_lifecycle {
        crate::archive::generation::rotate(&tx)?;
    }
    sources(&tx, &plan)?;
    members(&tx, &plan)?;
    requests(&tx, &plan)?;
    queries(&tx, &plan)?;
    coverage(&tx, &plan)?;
    refusals(&tx, next)?;
    let changes = crate::change_summary::ChangeSummary::start(&tx)?;
    crate::writer::apply_source_retractions(texts, &tx, &plan.retractions, counter, &changes)?;
    bindings(&tx, &plan)?;
    // Variants are authored records, so only a request can change them.
    if plan.requests != StateChange::Unchanged {
        crate::variants::project(&tx)?;
    }
    tx.execute(
        "INSERT OR REPLACE INTO confirmed_event SELECT * FROM primary_staged",
        [],
    )?;
    remember_blobs(&tx)?;
    replay::complete(&tx, next)?;
    tx.execute("INSERT INTO source_meta(key,value) VALUES ('write_counter',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[next.to_string()])?;
    crate::meta_set(&tx, "confirmed_position", next.to_string())?;
    tx.execute("DELETE FROM primary_pending WHERE position<=?1", [next])?;
    tx.execute("DELETE FROM primary_complete WHERE position<=?1", [next])?;
    // Confirmed events also change sessions outside the row application.
    let touched = tx
        .prepare("SELECT pk FROM session WHERE changed = ?1")?
        .query_map([counter], |row| row.get(0))?
        .collect::<rusqlite::Result<Vec<i64>>>()?;
    for session in touched {
        changes.whole_session(session);
    }
    crate::writer::finish_rows(
        &tx,
        &changes,
        &crate::writer::resolve::IndexUpkeep::PerTransaction,
        texts,
        counter,
    )?;
    crate::fault::check("primary.before_commit", "")?;
    tx.commit()?;
    text_transaction.commit();
    drop(plan);
    self::next(index)
}

#[cfg(test)]
pub(crate) fn apply(
    texts: &crate::text_rows::TextIds,
    index: &mut Connection,
    resolver: &std::sync::Arc<crate::writer::resolve::Resolver>,
    events: &[Confirmed],
) -> Result<i64> {
    let mut step = stage_next(index, events)?;
    loop {
        match step {
            Step::Complete(position) => return Ok(position),
            Step::Prepare(mut prepared) => {
                crate::writer::prepare_source_retractions(resolver, &mut prepared.retractions)?;
                step = apply_prepared(texts, index, *prepared)?;
            }
        }
    }
}

#[cfg(test)]
mod tests;
