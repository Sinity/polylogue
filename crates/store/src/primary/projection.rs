//! The primary cache applies complete confirmed positions, never producer acknowledgments.
pub(crate) const SCHEMA: &str = concat!("
ALTER TABLE parse_state ADD COLUMN native_publication BLOB;
CREATE TABLE primary_fact (
 event TEXT PRIMARY KEY, admitted INTEGER NOT NULL, changed INTEGER NOT NULL, source_order INTEGER NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('live','archived','tombstoned')),
 product BLOB, input TEXT, chain INTEGER, reference TEXT, kind TEXT NOT NULL, step_start INTEGER, step_end INTEGER, reading_version TEXT, session TEXT, native_context BLOB
) STRICT;
CREATE INDEX primary_fact_chain ON primary_fact(input,chain,admitted,event);
CREATE INDEX primary_fact_session ON primary_fact(session,kind,state,admitted);
", include_str!("projection/read_indexes.sql"), "
CREATE TABLE primary_fact_blob (
 event TEXT NOT NULL REFERENCES primary_fact(event), hash BLOB NOT NULL,
 PRIMARY KEY(event,hash)
) STRICT, WITHOUT ROWID;
CREATE INDEX primary_fact_blob_hash ON primary_fact_blob(hash);
CREATE TABLE primary_pending (
 n INTEGER PRIMARY KEY, position INTEGER NOT NULL, event TEXT NOT NULL,
 change TEXT NOT NULL, product BLOB, source_order INTEGER NOT NULL, native_context BLOB,
 UNIQUE(position,event,change)
) STRICT;
CREATE INDEX primary_pending_position ON primary_pending(position,n);
CREATE TABLE primary_complete(position INTEGER PRIMARY KEY, evidence BLOB) STRICT;
", include_str!("../schema/primary_refused.sql"), "
CREATE TABLE source_material (
 id TEXT NOT NULL, event TEXT NOT NULL, input TEXT, chain INTEGER, offset INTEGER,
 length INTEGER NOT NULL, hash BLOB NOT NULL, source_length INTEGER, blob_offset INTEGER NOT NULL,
 PRIMARY KEY(event,id)
) STRICT;
CREATE INDEX source_material_identity ON source_material(id);
CREATE INDEX source_material_place ON source_material(input,chain,offset);
", include_str!("projection/reference_index.sql"));

use super::{Confirmed, ConfirmedChange, Material, Product};
mod replay;
use crate::{Result, StoreError};
pub(crate) use replay::save as save_replay;
pub(crate) use replay::{has_fact, native_context, processed_source, replay_state};
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

/// Each staged page may be acknowledged. Only a complete position is visible.
fn stage(index: &mut Connection, events: &[Confirmed]) -> Result<()> {
    let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let applied = position(&tx)?;
    let mut state = replay_state(&tx)?;
    for event in events {
        if let ConfirmedChange::Control(control) = &event.change {
            if !event.event.is_empty() || event.native.is_some() || event.output_index != 0 {
                return Err(StoreError::Invalid(
                    "stream controls cannot carry a fact identity".into(),
                ));
            }
            replay::stage_control(&tx, &mut state, control)?;
            continue;
        }
        if let Some(context) = &event.native {
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
                "a confirmed fact needs its Sinex event identity".into(),
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
        let bytes = packed(&event.change)?;
        let native = event.native.as_ref().map(packed).transpose()?;
        let source_order = i64::try_from(event.output_index).map_err(|_| {
            StoreError::Invalid(
                "a source output position exceeds SQLite's signed integer range".into(),
            )
        })?;
        let existing:Option<(Vec<u8>,i64,Option<Vec<u8>>)>=tx.query_row("SELECT product,source_order,native_context FROM primary_pending WHERE position=?1 AND event=?2 AND change=?3",params![event.position,event.event,change],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?))).optional()?;
        match existing {
            Some((old, order, context)) => {
                if order != source_order || old != bytes || context != native {
                    return Err(StoreError::Invalid(
                        "a redelivered confirmed change has different contents".into(),
                    ));
                }
            }
            None => {
                tx.execute("INSERT INTO primary_pending(position,event,change,product,source_order,native_context) VALUES (?1,?2,?3,?4,?5,?6)",params![event.position,event.event,change,bytes,source_order,native])?;
            }
        }
    }
    replay::save(&tx, &state)?;
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
    native: Option<Vec<u8>>,
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
            native: row.get(14)?,
        })
    }
    fn insert(&self, index: &Connection, table: &str) -> Result<()> {
        index.execute(&format!("INSERT OR REPLACE INTO {table}(event,admitted,changed,source_order,state,product,input,chain,reference,kind,step_start,step_end,reading_version,session,native_context) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)"),params![self.event,self.admitted,self.changed,self.order,self.state,self.product,self.input,self.chain,self.reference,self.kind,self.start,self.end,self.version,self.session,self.native])?;
        Ok(())
    }
    fn tombstoned(event: String, position: i64, order: i64) -> Self {
        Self {
            event,
            admitted: position,
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
            native: None,
        }
    }
    // The delivered event has lifecycle identity even when its body is unavailable.
    fn unreadable(event: String, position: i64, order: i64) -> Self {
        let mut fact = Self::tombstoned(event, position, order);
        fact.state = "live".into();
        fact.kind = "refused".into();
        fact
    }
    fn product(&self) -> Result<Option<Product>> {
        self.product.as_deref().map(unpacked).transpose()
    }
}

fn metadata(product: &Product) -> (Option<String>, Option<i64>, Option<String>, &'static str) {
    if let Product::ToolCall { reading, facts } = product {
        return (
            Some(reading.input.id.clone()),
            Some(reading.chain),
            Some(format!("call:{}:{}", facts.message, facts.block)),
            "tool_call",
        );
    }
    let reference = match product {
        Product::AttachmentBinding { record } => {
            Some(format!("attachment-binding:{}", record.context().message))
        }
        Product::Link { facts, .. } => Some(facts.selection_tag()),
        Product::Session { facts, .. } => Some(format!("session:{}", facts.id)),
        Product::Message { facts, .. } => Some(format!("message:{}", facts.message.id)),
        Product::Document { facts, .. } => Some(format!("doc:{}", facts.document.id)),
        Product::Request(request) => Some(format!("request:{}", request.request.id)),
        Product::RestoredRequest { prepared, .. } => {
            Some(format!("request:{}", prepared.request.id))
        }
        Product::RestoredAssertion { fact, .. } => {
            Some(format!("assertion:{}@{}", fact.root, fact.id))
        }
        Product::Coverage(coverage) => Some(format!("coverage:{}", coverage.machine)),
        Product::Query { object } => Some(format!("query:{}", object.id)),
        Product::QueryRun { run } => Some(format!("query-run:{}", run.id)),
        Product::QueryRunOutcome { outcome } => Some(format!("query-run-outcome:{}", outcome.run)),
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

/// Candidate facts are disk-backed and hidden from ordinary domain reads.
fn candidate(index: &Connection, through: i64) -> Result<()> {
    index.execute_batch("DROP VIEW IF EXISTS temp.primary_next; DROP TABLE IF EXISTS temp.primary_candidate;
        CREATE TEMP TABLE primary_candidate AS SELECT * FROM primary_fact WHERE 0;
        CREATE UNIQUE INDEX temp.primary_candidate_event ON primary_candidate(event);
        CREATE INDEX temp.primary_candidate_chain ON primary_candidate(input,chain);
        CREATE TEMP VIEW primary_next AS SELECT * FROM primary_candidate UNION ALL SELECT * FROM main.primary_fact f WHERE NOT EXISTS(SELECT 1 FROM primary_candidate c WHERE c.event=f.event);")?;
    let mut query = index.prepare(
        "SELECT event,product,source_order,position,native_context FROM primary_pending WHERE position<=?1 ORDER BY position,CASE change WHEN 'live' THEN 0 WHEN 'archived' THEN 1 WHEN 'restored' THEN 1 ELSE 2 END,n",
    )?;
    let mut rows = query.query([through])?;
    while let Some(row) = rows.next()? {
        let event: String = row.get(0)?;
        let bytes: Vec<u8> = row.get(1)?;
        let order: i64 = row.get(2)?;
        let position: i64 = row.get(3)?;
        let native: Option<Vec<u8>> = row.get(4)?;
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
                    if old.state == "tombstoned" {
                        return Err(StoreError::Invalid(
                            "a tombstoned Sinex event cannot become live".into(),
                        ));
                    }
                    if old.kind != "refused"
                        && (old.product.as_ref() != Some(&bytes) || old.native != native)
                    {
                        return Err(StoreError::Invalid(
                            "a Sinex event changed its immutable product facts".into(),
                        ));
                    }
                }
                let (input, chain, reference, kind) = metadata(&product);
                let session = match product.as_ref() {
                    Product::Session { facts, .. } => Some(facts.id.to_string()),
                    Product::Message { facts, .. } => Some(facts.message.session.to_string()),
                    Product::Event { facts, .. } => facts.session.as_ref().map(ToString::to_string),
                    Product::Background { facts, .. } => Some(facts.session.to_string()),
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
                    native,
                })
            }
            ConfirmedChange::Archived => old.map(|mut fact| {
                fact.state = "archived".into();
                fact.changed = position;
                fact
            }),
            ConfirmedChange::Restored => match old {
                Some(mut fact)
                    if fact.state != "tombstoned"
                        && (fact.product.is_some() || fact.kind == "refused") =>
                {
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
            // An event erased before this index read it is recorded as tombstoned.
            ConfirmedChange::Tombstoned => Some(match old {
                Some(mut fact) => {
                    fact.state = "tombstoned".into();
                    fact.product = None;
                    fact.changed = position;
                    fact
                }
                None => Fact::tombstoned(event, position, order),
            }),
            ConfirmedChange::Refused { .. } => {
                Some(old.unwrap_or_else(|| Fact::unreadable(event, position, order)))
            }
            ConfirmedChange::Control(_) => None,
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
    requests: StateChange,
    queries: StateChange,
    binding_messages: BTreeSet<String>,
    coverage: StateChange,
}

/// How authored, query or coverage state takes one position's facts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StateChange {
    Unchanged,
    /// Only new live facts: each is applied after the ones already applied.
    Appended,
    /// A lifecycle change touched an applied fact: the state is cleared and
    /// every live fact is applied again in commit order.
    Rewritten,
}

impl StateChange {
    fn of(index: &Connection, kind: &str) -> Result<Self> {
        // A formerly unreadable fact can precede already-applied authored state.
        let rewritten: bool = index.query_row(
            "SELECT EXISTS(SELECT 1 FROM primary_candidate c JOIN main.primary_fact f ON f.event=c.event WHERE (f.kind=?1 AND f.state!=c.state) OR (f.kind='refused' AND c.kind=?1))",
            [kind],
            |row| row.get(0),
        )?;
        if rewritten {
            return Ok(Self::Rewritten);
        }
        let appended: bool = index.query_row(
            "SELECT EXISTS(SELECT 1 FROM primary_candidate c WHERE c.kind=?1 AND c.state='live' AND NOT EXISTS(SELECT 1 FROM main.primary_fact f WHERE f.event=c.event))",
            [kind],
            |row| row.get(0),
        )?;
        Ok(if appended {
            Self::Appended
        } else {
            Self::Unchanged
        })
    }

    /// The facts to apply, in commit order, or `None` when nothing changed.
    fn selection(self, kind: &str) -> Option<String> {
        match self {
            Self::Unchanged => None,
            Self::Appended => Some(format!(
                "SELECT c.product,c.admitted,c.source_order FROM primary_candidate c WHERE c.kind='{kind}' AND c.state='live' AND NOT EXISTS(SELECT 1 FROM main.primary_fact f WHERE f.event=c.event) ORDER BY c.admitted,c.source_order,c.event"
            )),
            Self::Rewritten => Some(format!(
                "SELECT product,admitted,source_order FROM primary_next WHERE kind='{kind}' AND state='live' ORDER BY admitted,source_order,event"
            )),
        }
    }
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
    let mut step = crate::ParseStep::new(output, counts, reading.start..reading.parsed_upto, false);
    step.progress = reading.progress.clone();
    step.delivery = reading.delivery;
    step.order_from = reading.order_from;
    step.reading_version = reading.reading_version.clone();
    step.repeats_whole_conversations = reading.repeats_whole_conversations;
    step
}

type ReadingStep = super::products::AssembledReading;
#[derive(Default)]
struct SourceSpans {
    through: u64,
    missing: bool,
}
fn finish_step(
    file: &mut crate::writer::ChainReading,
    reading: &super::Reading,
    facts: ReadingStep,
    spans: &mut SourceSpans,
) -> Result<()> {
    if let Some(checkpoint) = &facts.checkpoint
        && !checkpoint.matches(&reading.reading_version, reading.parsed_upto)
    {
        return Err(StoreError::Invalid(
            "parser continuation differs from its confirmed reading cursor".into(),
        ));
    }
    if facts.reset {
        spans.missing = false;
    } else if reading.start > spans.through {
        spans.missing = true;
    }
    let mut output = facts.output;
    let mut confirmed_event_ids = None;
    if reading.delivery.is_some() {
        let semantic_event_ids = facts.semantic_event_ids.ok_or_else(|| {
            StoreError::Invalid(
                "export progress omitted its original semantic event identities".into(),
            )
        })?;
        // The explicit map restores the sealed callback body. The admitted
        // Event products also carry generated identities, which must survive
        // payload normalization when the confirmed reading is applied.
        if semantic_event_ids
            .iter()
            .any(|(index, id)| output.event_ids.get(index) != Some(id))
        {
            return Err(StoreError::Invalid(
                "export event identities differ from their explicit callback identities".into(),
            ));
        }
        confirmed_event_ids = Some(
            (0..output.events.len())
                .map(|index| {
                    output.event_ids.get(&index).cloned().ok_or_else(|| {
                        StoreError::Invalid(
                            "confirmed export event omitted its admitted identity".into(),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        );
        output.event_ids = semantic_event_ids;
    }
    let mut step = step(reading, output, facts.counts);
    step.confirmed_event_ids = confirmed_event_ids;
    step.receipt = facts.receipt;
    step.reset = facts.reset;
    step.comparison = facts.comparison;
    step.checkpoint = facts.checkpoint;
    step.stopped_attempt = facts.stopped_attempt;
    step.source_prefix_missing = spans.missing || step.receipt.is_none();
    spans.through = spans.through.max(reading.parsed_upto);
    file.push_confirmed(&mut step)
}

fn plan(index: &Connection) -> Result<Plan> {
    let mut out = Plan {
        readings: Vec::new(),
        inputs: BTreeSet::new(),
        materials: BTreeSet::new(),
        available: HashMap::new(),
        requests: StateChange::of(index, "request")?,
        queries: StateChange::of(index, "query")?,
        binding_messages: BTreeSet::new(),
        coverage: StateChange::of(index, "coverage")?,
    };
    let mut chains = BTreeMap::<(String, i64), crate::InputKey>::new();
    let mut new_materials = HashMap::new();
    let mut q=index.prepare("SELECT * FROM primary_candidate UNION ALL SELECT f.* FROM primary_fact f JOIN primary_candidate c ON c.event=f.event")?;
    let mut rows = q.query([])?;
    while let Some(row) = rows.next()? {
        let fact = Fact::read(row)?;
        if let Some(product) = fact.product()? {
            if fact.kind == "parsed"
                && let Some(reading) = product.reading()
            {
                chains.insert(
                    (reading.input.id.clone(), reading.chain),
                    reading.input.clone(),
                );
            }
            if let Product::AttachmentBinding { record } = &product {
                out.binding_messages
                    .insert(record.context().message.to_string());
            }
            if let Product::Input { rows, .. } = &product {
                out.inputs.insert(rows.input.clone());
            }
            if let Some(material) = material(&product) {
                for descriptor in material.descriptors()? {
                    out.materials
                        .insert(polylogue_model::BlobHash(descriptor.hash.0));
                }
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
        let mut pending = ReadingStep::default();
        let mut spans = SourceSpans::default();
        let mut length = index.query_row(
            "SELECT coalesce(max(length),0) FROM source_revision WHERE input=?1 AND chain=?2",
            params![input, chain],
            |row| row.get::<_, u64>(0),
        )?;
        // Appends on the same immutable chain create a new length-bound export
        // reading instance. Select its actual reset control in original owner
        // admission order, never the largest length or callback ordinal. Older
        // instances remain facts, not callbacks of the current scan.
        let mut export_lengths = BTreeMap::<String, u64>::new();
        let mut export_instances = BTreeSet::new();
        let mut lengths = index.prepare("SELECT product FROM primary_next WHERE state='live' AND kind='parsed' AND input=?1 AND chain=?2 ORDER BY admitted,source_order,event")?;
        let mut length_rows = lengths.query(params![input, chain])?;
        while let Some(row) = length_rows.next()? {
            let bytes: Vec<u8> = row.get(0)?;
            let product: Product = unpacked(&bytes)?;
            let reading = product.reading().expect("parsed fact has reading");
            if let Product::ParseProgress { reset: true, .. } = &product
                && reading
                    .delivery
                    .is_some_and(|delivery| delivery.ordinal == 0)
            {
                let bound = reading.source_length.ok_or_else(|| {
                    StoreError::Invalid(
                        "confirmed export callback has no actual source length".into(),
                    )
                })?;
                let key = super::ExportReadingKey {
                    input: reading.input.clone(),
                    chain: reading.chain,
                    reading_version: reading.reading_version.clone(),
                    length: bound,
                };
                key.validate()?;
                let mut bound_to_source = false;
                let mut source_records = index.prepare("SELECT product FROM primary_next WHERE state!='tombstoned' AND kind='input' AND input=?1 AND chain=?2 ORDER BY admitted,source_order,event")?;
                let mut source_rows = source_records.query(params![input, chain])?;
                while let Some(row) = source_rows.next()? {
                    let bytes: Vec<u8> = row.get(0)?;
                    if let Product::Input { rows, .. } = unpacked(&bytes)?
                        && rows.input == reading.input.id
                        && rows.format == reading.input.format
                        && rows.revision.as_ref().is_some_and(|revision| {
                            revision.chain == chain
                                && u64::try_from(revision.length).ok() == Some(bound)
                        })
                    {
                        bound_to_source = true;
                        break;
                    }
                }
                if !bound_to_source {
                    return Err(StoreError::Invalid(
                        "export reset length has no retained confirmed source revision".into(),
                    ));
                }
                export_instances.insert((reading.reading_version.clone(), bound));
                export_lengths.insert(reading.reading_version.clone(), bound);
            }
        }
        drop(length_rows);
        drop(lengths);
        let mut query=index.prepare("SELECT product,event,admitted FROM primary_next WHERE state='live' AND kind='parsed' AND input=?1 AND chain=?2 ORDER BY step_start,step_end,reading_version,admitted,source_order,event")?;
        let mut rows = query.query(params![input, chain])?;
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(0)?;
            let product: Product = unpacked(&bytes)?;
            let admitted = row.get::<_, u64>(2)?;
            if admitted == 0 {
                return Err(StoreError::Invalid(
                    "a canonical product has no positive admission receipt".into(),
                ));
            }
            let reading = product.reading().expect("parsed fact has reading").clone();
            if reading.delivery.is_some() {
                if !reading.source_length.is_some_and(|bound| {
                    export_instances.contains(&(reading.reading_version.clone(), bound))
                }) {
                    return Err(StoreError::Invalid(
                        "export callback has no matching source-bound reset control".into(),
                    ));
                }
                if export_lengths.get(&reading.reading_version).copied() != reading.source_length {
                    continue;
                }
            }
            if current.as_ref().is_some_and(|old| old != &reading) {
                let old = current.take().expect("reading exists");
                finish_step(
                    &mut reading_file,
                    &old,
                    std::mem::take(&mut pending),
                    &mut spans,
                )?;
            }
            current = Some(reading.clone());
            length = length.max(reading.source_length.unwrap_or(reading.parsed_upto));
            match &product {
                Product::Message { facts, .. } => attachments(&facts.message.blocks, &mut needed),
                Product::Document { facts, .. } => attachments(&facts.document.blocks, &mut needed),
                _ => {}
            }
            pending.push(
                product,
                Some(admitted),
                Some(crate::wire_shapes::ReadingReceipt {
                    sequence: row.get(2)?,
                    id: Some(row.get(1)?),
                    observation: None,
                    observed_at_ms: None,
                }),
            )?;
        }
        if let Some(reading) = current {
            finish_step(&mut reading_file, &reading, pending, &mut spans)?;
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
            None=>index.query_row("SELECT m.length FROM source_material m JOIN primary_next f ON f.event=m.event WHERE f.state='live' AND m.hash=?1 LIMIT 1",[&hash.0[..]],|row|row.get::<_, u64>(0)).optional()?,
        };
        out.available.insert(hash, size);
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
                StoreError::Invalid("material input facts have no source extent".into())
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
                rows.insert_confirmed(tx, admitted)?;
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
        let mut query = tx.prepare("SELECT product FROM primary_next WHERE state='live' AND kind='attachment_binding' AND reference=?1 ORDER BY admitted,source_order,event")?;
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
        let admitted: i64 = row.get(1)?;
        match unpacked(&bytes)? {
            Product::Request(prepared) => super::authored::apply(tx, &prepared, admitted)?,
            Product::RestoredRequest { prepared } => {
                super::authored::restore(tx, &prepared, admitted)?;
            }
            Product::RestoredAssertion { fact } => {
                crate::assertions::apply(tx, std::slice::from_ref(&fact), admitted)?;
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
        let applied = checkpoint(index)?.unwrap_or(-1);
        let next: Option<i64> = index.query_row(
            "SELECT min(position) FROM primary_complete WHERE position>=?1",
            [applied],
            |row| row.get(0),
        )?;
        let Some(next) = next else {
            return position(index);
        };
        candidate(index, next)?;
        let mut plan = plan(index)?;
        resolver.confirmed_readings(&mut plan.readings, &plan.available)?;
        let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = crate::archive::bump_counter(&tx)?;
        // Lifecycle changes can remove or restore source admissions older than
        // a pre-I/O cut. New admissions preserve that cut; retained-row changes do not.
        let source_lifecycle: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM primary_candidate WHERE kind='input' AND changed!=admitted)",
            [], |row| row.get(0),
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
        crate::writer::apply_confirmed_readings(&tx, &mut plan.readings, counter, &changes)?;
        bindings(&tx, &plan)?;
        // Variants are authored records, so only a request can change them.
        if plan.requests != StateChange::Unchanged {
            crate::variants::project(&tx)?;
        }
        tx.execute(
            "INSERT OR REPLACE INTO primary_fact SELECT * FROM primary_candidate",
            [],
        )?;
        remember_blobs(&tx)?;
        replay::complete(&tx, next)?;
        tx.execute("INSERT INTO source_meta(key,value) VALUES ('write_counter',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[next.to_string()])?;
        tx.execute("INSERT INTO meta(key,value) VALUES ('confirmed_position',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[next.to_string()])?;
        tx.execute("DELETE FROM primary_pending WHERE position<=?1", [next])?;
        tx.execute("DELETE FROM primary_complete WHERE position<=?1", [next])?;
        // Confirmed facts also change sessions outside the row application.
        let touched = tx
            .prepare("SELECT pk FROM session WHERE changed = ?1")?
            .query_map([counter], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<i64>>>()?;
        for session in touched {
            changes.whole_session(session);
        }
        crate::writer::finish_rows(&tx, &changes, false)?;
        crate::fault::check("primary.before_commit", "")?;
        tx.commit()?;
        drop(plan);
    }
}

#[cfg(test)]
mod tests;
