//! The primary cache applies complete confirmed positions, never producer acknowledgments.
pub(crate) const SCHEMA: &str = "
CREATE TABLE primary_fact (
 event TEXT PRIMARY KEY, admitted INTEGER NOT NULL, changed INTEGER NOT NULL, source_order INTEGER NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('live','archived','tombstoned')),
 product BLOB, input TEXT, chain INTEGER, reference TEXT, kind TEXT NOT NULL, step_start INTEGER, step_end INTEGER, reading_version TEXT, session TEXT
) STRICT;
CREATE INDEX primary_fact_chain ON primary_fact(input,chain,admitted,event);
CREATE INDEX primary_fact_session ON primary_fact(session,kind,state,admitted);
CREATE TABLE primary_fact_blob (
 event TEXT NOT NULL REFERENCES primary_fact(event), hash BLOB NOT NULL,
 PRIMARY KEY(event,hash)
) STRICT, WITHOUT ROWID;
CREATE INDEX primary_fact_blob_hash ON primary_fact_blob(hash);
CREATE TABLE primary_pending (
 n INTEGER PRIMARY KEY, position INTEGER NOT NULL, event TEXT NOT NULL,
 change TEXT NOT NULL, product BLOB, source_order INTEGER NOT NULL,
 UNIQUE(position,event,change)
) STRICT;
CREATE INDEX primary_pending_position ON primary_pending(position,n);
CREATE TABLE primary_complete(position INTEGER PRIMARY KEY) STRICT;
CREATE TABLE source_material (
 id TEXT NOT NULL, event TEXT NOT NULL, input TEXT, chain INTEGER, offset INTEGER,
 length INTEGER NOT NULL, hash BLOB NOT NULL, source_length INTEGER, blob_offset INTEGER NOT NULL,
 PRIMARY KEY(event,id)
) STRICT;
CREATE INDEX source_material_identity ON source_material(id);
CREATE INDEX source_material_place ON source_material(input,chain,offset);
";

use super::{Confirmed, ConfirmedChange, Material, Product};
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

fn packed(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(crate::encoding::pack_bytes(&serde_json::to_vec(value)?)?)
}
fn unpacked<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    crate::encoding::unpack_json(bytes)
}

pub(crate) fn checkpoint(index: &Connection) -> Result<Option<i64>> {
    Ok(index
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key='confirmed_position'",
            [],
            |row| row.get(0),
        )
        .optional()?)
}
pub(crate) fn position(index: &Connection) -> Result<i64> {
    Ok(checkpoint(index)?.unwrap_or(0))
}

/// Each successfully staged page may be acknowledged. Only a complete position is visible.
fn stage(index: &mut Connection, events: &[Confirmed]) -> Result<()> {
    let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let applied = position(&tx)?;
    for event in events {
        if event.position == 0 && matches!(event.change, ConfirmedChange::PositionComplete) {
            tx.execute("INSERT INTO meta(key,value) VALUES ('confirmed_position','0') ON CONFLICT(key) DO NOTHING",[])?;
            continue;
        }
        if event.position <= 0 {
            return Err(StoreError::Invalid(
                "a confirmed data position must be positive".into(),
            ));
        }
        if event.position <= applied {
            continue;
        }
        if matches!(event.change, ConfirmedChange::PositionComplete) {
            tx.execute(
                "INSERT OR IGNORE INTO primary_complete(position) VALUES (?1)",
                [event.position],
            )?;
            continue;
        }
        if matches!(event.change, ConfirmedChange::Ignored) {
            continue;
        }
        if event.event.is_empty() {
            return Err(StoreError::Invalid(
                "a confirmed fact needs its Sinex event identity".into(),
            ));
        }
        let change = match &event.change {
            ConfirmedChange::Live(_) => "live",
            ConfirmedChange::Archived => "archived",
            ConfirmedChange::Restored => "restored",
            ConfirmedChange::Tombstoned => "tombstoned",
            _ => unreachable!(),
        };
        let bytes = packed(&event.change)?;
        let source_order = i64::try_from(event.output_index).map_err(|_| {
            StoreError::Invalid(
                "a source output position exceeds SQLite's signed integer range".into(),
            )
        })?;
        let existing:Option<(Vec<u8>,i64)>=tx.query_row("SELECT product,source_order FROM primary_pending WHERE position=?1 AND event=?2 AND change=?3",params![event.position,event.event,change],|row|Ok((row.get(0)?,row.get(1)?))).optional()?;
        if let Some((old, order)) = existing {
            if old != bytes || order != source_order {
                return Err(StoreError::Invalid(
                    "a redelivered confirmed change has different contents".into(),
                ));
            }
        } else {
            tx.execute("INSERT INTO primary_pending(position,event,change,product,source_order) VALUES (?1,?2,?3,?4,?5)",params![event.position,event.event,change,bytes,source_order])?;
        }
    }
    crate::fault::check("primary.stage_commit", "")?;
    tx.commit()?;
    Ok(())
}

#[derive(Clone)]
struct Fact {
    event: String,
    admitted: i64,
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
}
impl Fact {
    fn read(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            event: row.get(0)?,
            admitted: row.get(1)?,
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
        })
    }
    fn insert(&self, index: &Connection, table: &str) -> Result<()> {
        index.execute(&format!("INSERT OR REPLACE INTO {table}(event,admitted,changed,source_order,state,product,input,chain,reference,kind,step_start,step_end,reading_version,session) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)"),params![self.event,self.admitted,self.changed,self.order,self.state,self.product,self.input,self.chain,self.reference,self.kind,self.start,self.end,self.version,self.session])?;
        Ok(())
    }
    fn product(&self) -> Result<Option<Product>> {
        self.product.as_deref().map(unpacked).transpose()
    }
}

fn metadata(product: &Product) -> (Option<String>, Option<i64>, Option<String>, &'static str) {
    let reference = match product {
        Product::Session { facts, .. } => Some(format!("session:{}", facts.id)),
        Product::Message { facts, .. } => Some(format!("message:{}", facts.message.id)),
        Product::Document { facts, .. } => Some(format!("doc:{}", facts.document.id)),
        Product::Request(request) => Some(format!("request:{}", request.request.id)),
        Product::RestoredRequest { prepared, .. } => {
            Some(format!("request:{}", prepared.request.id))
        }
        Product::Coverage(coverage) => Some(format!("coverage:{}", coverage.machine)),
        _ => None,
    };
    if let Some(reading) = product.reading() {
        return (
            Some(reading.input.id.clone()),
            Some(reading.chain),
            reference,
            "parsed",
        );
    }
    match product {
        Product::Input { rows, .. } => (
            Some(rows.input.clone()),
            rows.extent
                .as_ref()
                .map(|e| e.chain)
                .or_else(|| rows.revision.as_ref().map(|r| r.chain)),
            reference,
            "input",
        ),
        Product::Blob(_) => (None, None, reference, "blob"),
        Product::Request(_) | Product::RestoredRequest { .. } => (None, None, reference, "request"),
        Product::Coverage(_) => (None, None, reference, "coverage"),
        Product::WorkSnapshot { .. } => (None, None, reference, "derived"),
        _ => unreachable!(),
    }
}

/// Candidate facts are disk-backed and hidden from ordinary domain reads.
fn candidate(index: &Connection, through: i64) -> Result<()> {
    index.execute_batch("DROP VIEW IF EXISTS temp.primary_next; DROP TABLE IF EXISTS temp.primary_candidate;
        CREATE TEMP TABLE primary_candidate AS SELECT * FROM primary_fact WHERE 0;
        CREATE UNIQUE INDEX temp.primary_candidate_event ON primary_candidate(event);
        CREATE INDEX temp.primary_candidate_chain ON primary_candidate(input,chain);
        CREATE TEMP VIEW primary_next AS SELECT * FROM primary_candidate UNION ALL SELECT * FROM main.primary_fact f WHERE NOT EXISTS(SELECT 1 FROM primary_candidate c WHERE c.event=f.event);")?;
    let mut query = index.prepare(
        "SELECT event,product,source_order,position FROM primary_pending WHERE position<=?1 ORDER BY position,CASE change WHEN 'live' THEN 0 WHEN 'archived' THEN 1 WHEN 'restored' THEN 1 ELSE 2 END,n",
    )?;
    let mut rows = query.query([through])?;
    while let Some(row) = rows.next()? {
        let event: String = row.get(0)?;
        let bytes: Vec<u8> = row.get(1)?;
        let order: i64 = row.get(2)?;
        let position: i64 = row.get(3)?;
        let change: ConfirmedChange = unpacked(&bytes)?;
        let old = index
            .query_row(
                "SELECT * FROM primary_next WHERE event=?1",
                [&event],
                Fact::read,
            )
            .optional()?;
        let next = match change {
            ConfirmedChange::Live(product) => {
                let bytes = packed(&product)?;
                if let Some(old) = &old {
                    if old
                        .product
                        .as_ref()
                        .is_some_and(|previous| previous != &bytes)
                    {
                        return Err(StoreError::Invalid(
                            "a Sinex event changed its immutable product facts".into(),
                        ));
                    }
                    if old.state == "tombstoned" {
                        return Err(StoreError::Invalid(
                            "a tombstoned Sinex event cannot become live".into(),
                        ));
                    }
                }
                let (input, chain, reference, kind) = metadata(&product);
                let session = match &product {
                    Product::Session { facts, .. } => Some(facts.id.to_string()),
                    Product::Message { facts, .. } => Some(facts.message.session.to_string()),
                    Product::Event { facts, .. } => facts.session.as_ref().map(ToString::to_string),
                    Product::Background { facts, .. } => Some(facts.session.to_string()),
                    Product::WorkSnapshot { snapshot } => Some(snapshot.session.id.to_string()),
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
                Some(Fact {
                    event,
                    admitted: old.as_ref().map_or(position, |f| f.admitted),
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
                })
            }
            ConfirmedChange::Archived => old.map(|mut fact| {
                fact.state = "archived".into();
                fact.changed = position;
                fact
            }),
            ConfirmedChange::Restored => match old {
                Some(mut fact) if fact.state != "tombstoned" && fact.product.is_some() => {
                    fact.state = "live".into();
                    fact.changed = position;
                    Some(fact)
                }
                Some(_) => {
                    return Err(StoreError::Invalid(
                        "a tombstoned Sinex event cannot be restored".into(),
                    ));
                }
                None => None,
            },
            ConfirmedChange::Tombstoned => old.map(|mut fact| {
                fact.state = "tombstoned".into();
                fact.product = None;
                fact.changed = position;
                fact
            }),
            ConfirmedChange::Ignored | ConfirmedChange::PositionComplete => None,
        };
        if let Some(next) = next {
            next.insert(index, "primary_candidate")?;
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
fn descriptor(material: &Material) -> Result<hashed_blobs::Extent> {
    let [bytes] = material.extents.as_slice() else {
        return Err(StoreError::Invalid(
            "a primary material needs one committed blob extent".into(),
        ));
    };
    let extent = hashed_blobs::Extent::decode(bytes)?;
    if extent.hash.0 != material.hash.0 || extent.decoded_len != material.length || extent.len == 0
    {
        return Err(StoreError::Invalid(
            "a primary material differs from its committed extent".into(),
        ));
    }
    Ok(extent)
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
    readings: Vec<(crate::InputKey, i64, u64, crate::writer::ChainReading)>,
    inputs: BTreeSet<String>,
    materials: BTreeSet<polylogue_model::BlobHash>,
    available: HashMap<polylogue_model::BlobHash, Option<u64>>,
    requests: bool,
    coverage: bool,
}

fn attachments(
    blocks: &[polylogue_model::Block],
    hashes: &mut BTreeSet<polylogue_model::BlobHash>,
) {
    use polylogue_model::Block;
    for block in blocks {
        match block {
            Block::Image(value) | Block::File(value) => hashes.extend(value.blob),
            Block::ToolResult { attachments, .. } => {
                hashes.extend(attachments.iter().filter_map(|a| a.blob))
            }
            _ => {}
        }
    }
}

fn step(
    reading: &super::Reading,
    output: polylogue_model::Output,
    counts: polylogue_model::RecordCounts,
) -> crate::ParseStep {
    let mut step = crate::ParseStep::new(
        output,
        counts,
        reading.start..reading.parsed_upto,
        reading.at_end,
    );
    step.order_from = reading.order_from;
    step.reading_version = reading.reading_version.clone();
    step.repeats_whole_conversations = reading.repeats_whole_conversations;
    step.no_parser_yet = reading.no_parser_yet;
    step
}

fn finish_step(
    file: &mut crate::writer::ChainReading,
    reading: &super::Reading,
    output: polylogue_model::Output,
    counts: polylogue_model::RecordCounts,
    receipt: Option<crate::wire_shapes::ReadingReceipt>,
    published_reset: bool,
    comparison: Option<crate::tool_databases::Comparison>,
    through: &mut u64,
    missing: &mut bool,
) -> Result<()> {
    if published_reset {
        *missing = false;
    } else if reading.start > *through {
        *missing = true;
    }
    let mut step = step(reading, output, counts);
    step.receipt = receipt;
    step.reset = published_reset;
    step.comparison = comparison;
    step.source_prefix_missing = *missing || step.receipt.is_none();
    *through = (*through).max(reading.parsed_upto);
    file.push_confirmed(&mut step)
}

fn plan(index: &Connection) -> Result<Plan> {
    let mut out = Plan {
        readings: Vec::new(),
        inputs: BTreeSet::new(),
        materials: BTreeSet::new(),
        available: HashMap::new(),
        requests: false,
        coverage: false,
    };
    let mut chains = BTreeMap::<(String, i64), crate::InputKey>::new();
    let mut new_materials = HashMap::new();
    let mut q=index.prepare("SELECT * FROM primary_candidate UNION ALL SELECT f.* FROM primary_fact f JOIN primary_candidate c ON c.event=f.event")?;
    let mut rows = q.query([])?;
    while let Some(row) = rows.next()? {
        let fact = Fact::read(row)?;
        if let Some(product) = fact.product()? {
            if let Some(reading) = product.reading() {
                chains.insert(
                    (reading.input.id.clone(), reading.chain),
                    reading.input.clone(),
                );
            }
            if let Product::Input { rows, .. } = &product {
                out.inputs.insert(rows.input.clone());
            }
            if matches!(
                product,
                Product::Request(_) | Product::RestoredRequest { .. }
            ) {
                out.requests = true;
            }
            if matches!(product, Product::Coverage(_)) {
                out.coverage = true;
            }
            if let Some(material) = material(&product) {
                descriptor(material)?;
                out.materials.insert(material.hash);
                let live: bool = index.query_row(
                    "SELECT state='live' FROM primary_next WHERE event=?1",
                    [&fact.event],
                    |row| row.get(0),
                )?;
                if live {
                    new_materials.insert(material.hash, material.length);
                }
            }
        }
    }
    drop(rows);
    drop(q);
    for hash in &out.materials {
        let mut query = index.prepare("SELECT f.product FROM primary_fact_blob b JOIN primary_next f ON f.event=b.event WHERE b.hash=?1 AND f.state='live' AND f.kind='parsed'")?;
        let mut rows = query.query([&hash.0[..]])?;
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(0)?;
            let product: Product = unpacked(&bytes)?;
            let reading = product.reading().expect("parsed fact has reading");
            chains.insert(
                (reading.input.id.clone(), reading.chain),
                reading.input.clone(),
            );
        }
    }
    let mut needed = BTreeSet::new();
    for ((input, chain), key) in chains {
        let mut reading_file = crate::writer::ChainReading::new()?;
        let mut current: Option<super::Reading> = None;
        let mut output = polylogue_model::Output::default();
        let mut counts = polylogue_model::RecordCounts::default();
        let mut receipt = None;
        let mut published_reset = false;
        let mut comparison = None;
        let mut through = 0;
        let mut missing = false;
        let mut length = index.query_row(
            "SELECT coalesce(max(length),0) FROM source_revision WHERE input=?1 AND chain=?2",
            params![input, chain],
            |row| unsigned_column(row, 0),
        )?;
        let mut query=index.prepare("SELECT product,event,admitted FROM primary_next WHERE state='live' AND kind='parsed' AND input=?1 AND chain=?2 ORDER BY step_start,step_end,reading_version,admitted,source_order,event")?;
        let mut rows = query.query(params![input, chain])?;
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(0)?;
            let product: Product = unpacked(&bytes)?;
            let reading = product.reading().expect("parsed fact has reading").clone();
            if current.as_ref().is_some_and(|old| old != &reading) {
                let old = current.take().expect("reading exists");
                finish_step(
                    &mut reading_file,
                    &old,
                    std::mem::take(&mut output),
                    std::mem::take(&mut counts),
                    receipt.take(),
                    published_reset,
                    comparison.take(),
                    &mut through,
                    &mut missing,
                )?;
                published_reset = false;
            }
            current = Some(reading.clone());
            length = length.max(reading.parsed_upto);
            match product {
                Product::Session { facts, .. } => output.sessions.push(facts),
                Product::Message {
                    facts,
                    order,
                    projection,
                    ..
                } => {
                    if let Some(projection) = projection {
                        output
                            .message_projections
                            .insert(output.messages.len(), projection);
                    }
                    if let Some(order) = order {
                        output.message_order.insert(output.messages.len(), order);
                    }
                    attachments(&facts.message.blocks, &mut needed);
                    output.messages.push(facts);
                }
                Product::Document { facts, .. } => {
                    attachments(&facts.document.blocks, &mut needed);
                    output.documents.push(facts);
                }
                Product::Event {
                    id,
                    facts,
                    imported,
                    ..
                } => {
                    output.event_ids.insert(output.events.len(), id);
                    if let Some(imported) = imported {
                        output.imported_events.insert(output.events.len(), imported);
                    }
                    output.events.push(facts);
                }
                Product::Link { facts, .. } => output.links.push(facts),
                Product::Background { facts, .. } => output.background.push(facts),
                Product::Repository { facts, .. } => output.repo_lookups.push(facts),
                Product::ParseProgress {
                    counts: record_counts,
                    reset,
                    comparison: source_comparison,
                    ..
                } => {
                    counts = record_counts;
                    published_reset = reset;
                    comparison = source_comparison;
                    receipt = Some(crate::wire_shapes::ReadingReceipt {
                        sequence: row.get(2)?,
                        id: Some(row.get(1)?),
                        observation: None,
                        observed_at_ms: None,
                    });
                }
                _ => unreachable!(),
            }
        }
        if let Some(reading) = current {
            finish_step(
                &mut reading_file,
                &reading,
                output,
                counts,
                receipt,
                published_reset,
                comparison,
                &mut through,
                &mut missing,
            )?;
        } else {
            let mut absent =
                crate::ParseStep::new(Default::default(), Default::default(), 0..0, false);
            absent.source_prefix_missing = true;
            reading_file.push_confirmed(&mut absent)?;
        }
        reading_file.finish()?;
        out.readings.push((key, chain, length, reading_file));
    }
    for hash in needed {
        let size=match new_materials.get(&hash) {
            Some(size)=>Some(*size),
            None=>index.query_row("SELECT m.length FROM source_material m JOIN primary_next f ON f.event=m.event WHERE f.state='live' AND m.hash=?1 LIMIT 1",[&hash.0[..]],|row|unsigned_column(row,0)).optional()?,
        };
        out.available.insert(hash, size);
    }
    Ok(out)
}

fn store_material(tx: &Connection, event: &str, product: &Product) -> Result<()> {
    let Some(material) = material(product) else {
        return Ok(());
    };
    let extent = descriptor(material)?;
    let (input, chain, offset, source_length, blob_offset) = match product {
        Product::Input { rows, .. } => {
            let source = rows.extent.as_ref().ok_or_else(|| {
                StoreError::Invalid("material input facts have no source extent".into())
            })?;
            if source.blob != material.hash
                || source.length < 0
                || source.offset < 0
                || source.blob_offset < 0
                || (source.blob_offset as u64)
                    .checked_add(source.length as u64)
                    .is_none_or(|end| end > material.length)
            {
                return Err(StoreError::Invalid(
                    "input extent differs from its material".into(),
                ));
            }
            (
                Some(rows.input.as_str()),
                Some(source.chain),
                Some(source.offset),
                Some(source.length),
                source.blob_offset,
            )
        }
        _ => (None, None, None, None, 0),
    };
    let length = i64::try_from(material.length).map_err(|_| {
        StoreError::Invalid("material length exceeds the source catalog range".into())
    })?;
    let stored = i64::try_from(extent.len).map_err(|_| {
        StoreError::Invalid("extent length exceeds the source catalog range".into())
    })?;
    tx.execute("INSERT OR REPLACE INTO source_material(id,event,input,chain,offset,length,hash,source_length,blob_offset) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",params![material.id,event,input,chain,offset,length,&material.hash.0[..],source_length,blob_offset])?;
    tx.execute("INSERT INTO source_blob(hash,size,stored_size,storage) VALUES (?1,?2,?3,?4) ON CONFLICT(hash) DO UPDATE SET storage=excluded.storage,stored_size=excluded.stored_size",params![&material.hash.0[..],length,stored,extent.encode()])?;
    Ok(())
}

fn unsigned_column(row: &rusqlite::Row<'_>, column: usize) -> rusqlite::Result<u64> {
    let value: i64 = row.get(column)?;
    u64::try_from(value).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(column, value))
}

fn sources(tx: &rusqlite::Transaction<'_>, plan: &Plan) -> Result<()> {
    // Material catalog changes precede every source extent citing them.
    let mut q = tx.prepare("SELECT event,product,state FROM primary_candidate")?;
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
        let mut q=tx.prepare("SELECT product,admitted FROM primary_next WHERE state='live' AND kind='input' AND input=?1 ORDER BY admitted,source_order,event")?;
        let mut rows = q.query([input])?;
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(0)?;
            let admitted: i64 = row.get(1)?;
            if let Product::Input { rows, .. } = unpacked(&bytes)? {
                rows.insert(tx, admitted)?;
            }
        }
    }
    for hash in &plan.materials {
        tx.execute("DELETE FROM source_blob WHERE hash=?1 AND NOT EXISTS(SELECT 1 FROM source_material WHERE hash=?1) AND NOT EXISTS(SELECT 1 FROM source_extent WHERE blob=?1)",[&hash.0[..]])?;
    }
    Ok(())
}

fn requests(tx: &rusqlite::Transaction<'_>, plan: &Plan) -> Result<()> {
    if !plan.requests {
        return Ok(());
    }
    super::authored::clear(tx)?;
    let mut q=tx.prepare("SELECT product,admitted FROM primary_next WHERE state='live' AND kind='request' ORDER BY admitted,source_order,event")?;
    let mut rows = q.query([])?;
    while let Some(row) = rows.next()? {
        let bytes: Vec<u8> = row.get(0)?;
        let admitted: i64 = row.get(1)?;
        match unpacked(&bytes)? {
            Product::Request(prepared) => super::authored::apply(tx, &prepared, admitted)?,
            Product::RestoredRequest {
                prepared,
                claims,
                current,
            } => {
                super::authored::restore(tx, &prepared, &claims, &current, admitted)?;
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn remember_blobs(tx: &rusqlite::Transaction<'_>) -> Result<()> {
    tx.execute(
        "DELETE FROM primary_fact_blob WHERE event IN (SELECT event FROM primary_candidate)",
        [],
    )?;
    let mut query =
        tx.prepare("SELECT event,product FROM primary_candidate WHERE product IS NOT NULL")?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        let event: String = row.get(0)?;
        let bytes: Vec<u8> = row.get(1)?;
        let product: Product = unpacked(&bytes)?;
        let mut hashes = BTreeSet::new();
        match product {
            Product::Message { facts, .. } => attachments(&facts.message.blocks, &mut hashes),
            Product::Document { facts, .. } => attachments(&facts.document.blocks, &mut hashes),
            _ => {}
        }
        for hash in hashes {
            tx.execute(
                "INSERT INTO primary_fact_blob(event,hash) VALUES (?1,?2)",
                params![event, &hash.0[..]],
            )?;
        }
    }
    Ok(())
}

/// Stages bounded pages and atomically applies each complete global position.
pub(crate) fn apply(
    index: &mut Connection,
    resolver: &Arc<crate::writer::resolve::Resolver>,
    events: &[Confirmed],
) -> Result<i64> {
    if index.pragma_query_value::<i64, _>(None, "temp_store", |row| row.get(0))? != 1 {
        index.pragma_update(None, "temp_store", "file")?;
        index.pragma_update(Some("temp"), "cache_size", -8192)?;
    }
    stage(index, events)?;
    loop {
        let applied = position(index)?;
        let next: Option<i64> = index.query_row(
            "SELECT min(position) FROM primary_complete WHERE position>?1",
            [applied],
            |row| row.get(0),
        )?;
        let Some(next) = next else {
            return Ok(applied);
        };
        candidate(index, next)?;
        let mut plan = plan(index)?;
        resolver.confirmed_readings(&mut plan.readings, &plan.available)?;
        let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = crate::archive::bump_counter(&tx)?;
        sources(&tx, &plan)?;
        requests(&tx, &plan)?;
        if plan.coverage {
            crate::coverage::clear(&tx)?;
            let mut q=tx.prepare("SELECT product FROM primary_next WHERE state='live' AND kind='coverage' ORDER BY admitted,source_order,event")?;
            let mut rows = q.query([])?;
            while let Some(row) = rows.next()? {
                let bytes: Vec<u8> = row.get(0)?;
                if let Product::Coverage(coverage) = unpacked(&bytes)? {
                    crate::coverage::apply(&tx, &coverage)?;
                }
            }
        }
        crate::writer::apply_confirmed_readings(&tx, &mut plan.readings, counter)?;
        tx.execute(
            "INSERT OR REPLACE INTO primary_fact SELECT * FROM primary_candidate",
            [],
        )?;
        remember_blobs(&tx)?;
        tx.execute("INSERT INTO source_meta(key,value) VALUES ('write_counter',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[next.to_string()])?;
        tx.execute("INSERT INTO meta(key,value) VALUES ('confirmed_position',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[next.to_string()])?;
        tx.execute("DELETE FROM primary_pending WHERE position<=?1", [next])?;
        tx.execute("DELETE FROM primary_complete WHERE position<=?1", [next])?;
        crate::fault::check("primary.before_commit", "")?;
        tx.commit()?;
        drop(plan);
    }
}

#[cfg(test)]
mod tests;
