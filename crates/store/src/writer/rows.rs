//! `index.db` rows for parsed records: sessions, messages and their
//! sources, blocks, tool calls, events, links and parse positions.

use super::*;
use crate::first_build::FirstBuild;
use crate::versions::VersionKind;

mod assertions;
mod retract;
pub(super) use retract::prune_sessions;

/// Writes a parse step's rows; returns the number of message rows written.
pub(super) fn write_rows(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    chain: i64,
    counter: i64,
    first_build: Option<&FirstBuild>,
    receipt: Option<&crate::wire_shapes::ReadingReceipt>,
) -> Result<u64> {
    apply_rows(
        tx,
        b,
        p,
        ReadingWrite {
            chain,
            counter,
            first_build,
            confirmed: false,
            receipt,
        },
    )
}

/// Apply confirmed canonical facts to the derived cache; source byte admission
/// was already proved by the producer and lifecycle may hide earlier facts.
pub(super) fn write_confirmed_rows(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    chain: i64,
    counter: i64,
    first_build: Option<&FirstBuild>,
    receipt: Option<&crate::wire_shapes::ReadingReceipt>,
) -> Result<u64> {
    apply_rows(
        tx,
        b,
        p,
        ReadingWrite {
            chain,
            counter,
            first_build,
            confirmed: true,
            receipt,
        },
    )
}

struct ReadingWrite<'a> {
    chain: i64,
    counter: i64,
    first_build: Option<&'a FirstBuild>,
    confirmed: bool,
    receipt: Option<&'a crate::wire_shapes::ReadingReceipt>,
}

fn apply_rows(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    context: ReadingWrite<'_>,
) -> Result<u64> {
    let ReadingWrite {
        chain,
        counter,
        first_build,
        confirmed,
        receipt,
    } = context;
    p.counts.check()?;
    p.prepared.validate(&p.output)?;
    let keys = p.prepared.keys.as_deref().ok_or_else(|| {
        StoreError::Invalid("an indexed reading has no resolved natural keys".into())
    })?;
    let choices =
        p.prepared.choices.as_ref().ok_or_else(|| {
            StoreError::Invalid("an indexed reading has no resolved versions".into())
        })?;
    if p.prepared.chain != Some(chain) {
        return Err(StoreError::Invalid(
            "prepared reading has another input chain".into(),
        ));
    }
    let input_pk = keys.get(resolve::Kind::Input, b.input.id.as_bytes())?.pk;
    validate_range(b, p)?;
    if !confirmed && !admit_step(tx, input_pk, chain, p)? {
        return Ok(0);
    }
    tx.execute("INSERT INTO input_ref(pk,id,format) VALUES (?1,?2,?3) ON CONFLICT(pk) DO UPDATE SET format=excluded.format WHERE format IS NOT excluded.format",params![input_pk,b.input.id,b.input.format])?;
    if let Some(update) = &p.comparison {
        update.write(tx, input_pk)?;
    }
    let rows = RowWriter {
        tx,
        input_pk,
        chain,
        rank: i64::from(b.input.arrival_rank),
        counter,
        packed: &p.prepared.texts,
        first_build,
        keys,
        choices,
    };
    if p.reset {
        rows.retract()?;
    }
    let mut sessions: HashMap<String, i64> = HashMap::new();
    for s in &p.output.sessions {
        let (pk, _) = rows.upsert_session(s)?;
        sessions.insert(s.id.to_string(), pk);
    }
    let source = crate::documents::Source {
        input_pk,
        chain,
        rank: i64::from(b.input.arrival_rank),
        counter,
        packed: &p.prepared.texts,
        first_build,
        keys: Some(keys),
    };
    for (i, row) in p.output.documents.iter().enumerate() {
        if p.prepared
            .choices
            .as_ref()
            .is_some_and(|choices| !choices.keep_documents[i])
        {
            continue;
        }
        let prepared = &p.prepared.documents[i];
        crate::documents::write(tx, &source, row, prepared)?;
    }
    let mut written = 0;
    let mut activity: HashMap<i64, i64> = HashMap::new();
    // Messages one record holds share its offset; their order in the
    // record breaks the tie.
    let mut in_record: HashMap<u64, i64> = HashMap::new();
    for (i, row) in p.output.messages.iter().enumerate() {
        if p.prepared
            .choices
            .as_ref()
            .is_some_and(|choices| !choices.keep_messages[i])
        {
            continue;
        }
        let session_pk = rows.session_pk(&mut sessions, &row.message.session)?;
        let prepared = &p.prepared.messages[i];
        let position = match p.message_position(i)? {
            Some(order) => (order, 0),
            None => {
                let next = in_record.entry(row.offset).or_default();
                *next += 1;
                (row.offset as i64, *next - 1)
            }
        };
        // A kind this step states needs no lookup.
        let main = choices.message_mains[i];
        written += u64::from(rows.write_message(row, session_pk, prepared, position, main)?);
        if let Some(at) = ms(row.message.at) {
            let last = activity.entry(session_pk).or_insert(at);
            *last = (*last).max(at);
        }
    }
    for (session_pk, at) in activity {
        tx.prepare_cached(
            "UPDATE session SET last_activity_at = MAX(COALESCE(last_activity_at, ?2), ?2),
                 changed = ?3
             WHERE pk = ?1",
        )?
        .execute(params![session_pk, at, counter])?;
    }
    for (i, e) in p.output.events.iter().enumerate() {
        if let Some(id) = &e.session {
            rows.session_pk(&mut sessions, id)?;
        }
        rows.write_event(i)?;
    }
    for (i, l) in p.output.links.iter().enumerate() {
        rows.write_link(l, i)?;
    }
    for l in &p.output.repo_lookups {
        crate::repo_lookup::upsert_lookup(tx, l, input_pk, chain, counter)?;
    }
    {
        for pk in choices.messages.keys() {
            written += u64::from(rows.refresh_message(*pk)?);
        }
        for (pk, choice) in &choices.documents {
            crate::documents::apply_choice(tx, *pk, choice.as_ref(), counter, first_build, keys)?;
        }
        for pk in choices.events.keys() {
            rows.refresh_event(*pk)?;
        }
        for pk in choices.links.keys() {
            rows.refresh_link(*pk)?;
        }
    }
    // Every session these records touched is derived again.
    for pk in sessions.values() {
        tx.prepare_cached("UPDATE session SET changed = ?2 WHERE pk = ?1 AND changed IS NOT ?2")?
            .execute(params![pk, counter])?;
    }
    // Hooks name the selected tool-call rows, which message projection just wrote.
    for (i, e) in p.output.events.iter().enumerate() {
        let session_pk = e
            .session
            .as_ref()
            .map(|id| rows.session_pk(&mut sessions, id))
            .transpose()?;
        if let Some(session_pk) = session_pk.filter(|_| e.kind.starts_with("hook_")) {
            tx.prepare_cached(
                "UPDATE session SET live_reparse=1 WHERE pk=?1 AND ?2<=live_last_event_at",
            )?
            .execute(params![session_pk, ms(e.at)])?;
            // Prepared events release their JSON payload before writing. Live
            // facts must read the same payload as the stored event.
            let event = choices.live_events[i].clone();
            crate::live::apply(tx, session_pk, &event, counter)?;
        }
    }
    for u in &p.output.background {
        let (outcome, reason, exit) = outcome_columns(&u.outcome);
        tx.prepare_cached(
            "UPDATE tool_call SET outcome = ?1, outcome_reason = ?2, exit_code = ?3,result_at=?6,reported_duration_ms=?7
             WHERE session_pk = (SELECT pk FROM session WHERE id = ?4) AND call_id = ?5",
        )?
        .execute(params![
            outcome,
            reason,
            exit,
            u.session.to_string(),
            u.call_id,
            ms(u.result_at),
            u.reported_duration_ms.map(|ms|ms as i64)
        ])?;
    }
    for pk in sessions.values() {
        tx.prepare_cached(
            "INSERT OR IGNORE INTO parsed(input_pk, chain, session_pk) VALUES (?1, ?2, ?3)",
        )?
        .execute(params![input_pk, chain, pk])?;
    }
    let receipt = receipt.or(p.receipt.as_ref());
    if confirmed && receipt.is_none() {
        crate::wire_shapes::unobserved(tx, input_pk, chain)?;
        tx.execute(
            "DELETE FROM parse_state WHERE input_pk=?1 AND chain=?2",
            params![input_pk, chain],
        )?;
        return Ok(written);
    }
    let receipt = receipt.ok_or_else(|| {
        StoreError::Invalid("an indexed reading has no committed source receipt".into())
    })?;
    crate::wire_shapes::apply(
        tx,
        crate::wire_shapes::Source {
            input_pk,
            input: &b.input.id,
            format: &b.input.format,
            chain,
            length: b.length,
            receipt: receipt.sequence,
            receipt_id: receipt.id.as_deref(),
            observation: receipt.observation.as_deref(),
            observed_at_ms: receipt.observed_at_ms,
        },
        p,
    )?;
    rows.write_parse_state(b, p)?;
    Ok(written)
}

fn validate_range(batch: &Batch, step: &ParseStep) -> Result<()> {
    if step.start > step.parsed_upto
        || step.parsed_upto > batch.length
        || i64::try_from(batch.length).is_err()
    {
        return Err(StoreError::Invalid(
            "reading range is outside its stored input".into(),
        ));
    }
    Ok(())
}

/// A continuing reading can only extend its committed prefix. Explicit resets
/// replace a complete reading, including row dumps whose accounting starts later.
fn admit_step(tx: &Transaction<'_>, input: i64, chain: i64, step: &ParseStep) -> Result<bool> {
    if step.reset {
        return Ok(true);
    }
    let previous: Option<(u64, String)> = tx
        .query_row(
            "SELECT parsed_upto, reading_version FROM parse_state WHERE input_pk=?1 AND chain=?2",
            params![input, chain],
            |row| Ok((row.get::<_, i64>(0)? as u64, row.get(1)?)),
        )
        .optional()?;
    let cursor = previous.as_ref().map_or(0, |(cursor, _)| *cursor);
    if step.start < cursor
        && step.parsed_upto <= cursor
        && previous
            .as_ref()
            .is_some_and(|(_, version)| version == &step.reading_version)
    {
        return Ok(false);
    }
    if step.start != cursor {
        return Err(StoreError::Invalid(format!(
            "reading starts at {} after committed position {cursor}",
            step.start
        )));
    }
    Ok(true)
}

/// A block's packed body and structured value, as `block` stores them.
#[derive(Clone, Copy)]
struct StoredBlock<'a> {
    body: &'a [u8],
    data: Option<&'a [u8]>,
    text_id: Option<i64>,
}

pub(super) struct RowWriter<'a, 't> {
    tx: &'a Transaction<'t>,
    input_pk: i64,
    chain: i64,
    rank: i64,
    counter: i64,
    packed: &'a HashMap<[u8; 32], Vec<u8>>,
    /// Set during a first build ([`Writer::start`]): search entries,
    /// trigram entries and secondary indexes are built once, in bulk, when
    /// it ends ([`Inner::build_deferred`]), and rows are found by natural
    /// key through it ([`FirstBuild`]).
    first_build: Option<&'a FirstBuild>,
    keys: &'a resolve::ResolvedKeys,
    choices: &'a resolve::versions::Choices,
}

/// A message reading's source and shown fields.
pub(super) struct Version {
    time: Vec<u8>,
    input_pk: i64,
    chain: i64,
    start: i64,
    end: i64,
    complete: bool,
    rank: i64,
    hash: Vec<u8>,
    /// The most any block of the version holds of PEM private key blocks
    /// ([`crate::ordered_blocks::pem`]). Written with the version, never
    /// compared.
    pem: i64,
}

pub(super) fn ms(t: Option<Timestamp>) -> Option<i64> {
    t.map(Timestamp::unix_ms)
}

pub(super) fn name<T: serde::Serialize>(v: &T) -> String {
    match serde_json::to_value(v) {
        Ok(serde_json::Value::String(s)) => s,
        _ => String::new(),
    }
}

impl RowWriter<'_, '_> {
    /// The message with natural ID `id`.
    fn message_pk(&self, id: &str) -> Result<Option<i64>> {
        let pk = self.keys.get(resolve::Kind::Message, id.as_bytes())?.pk;
        Ok(self
            .tx
            .prepare_cached("SELECT pk FROM message WHERE pk=?1")?
            .query_row([pk], |row| row.get(0))
            .optional()?)
    }
    fn session_pk(&self, cache: &mut HashMap<String, i64>, id: &SessionId) -> Result<i64> {
        let key = id.to_string();
        if let Some(pk) = cache.get(&key) {
            return Ok(*pk);
        }
        let reserved = self.keys.get(resolve::Kind::Session, key.as_bytes())?.pk;
        let existing = self
            .tx
            .prepare_cached("SELECT pk FROM session WHERE pk=?1")?
            .query_row([reserved], |row| row.get(0))
            .optional()?;
        let pk = match existing {
            Some(pk) => pk,
            // Named by an event before any of its own records arrived: a
            // stub, whose kind the session's first real facts decide
            // (index.sql, session.stub), whichever file is read first.
            None => self.session_row(id, true)?,
        };
        cache.insert(key, pk);
        Ok(pk)
    }

    /// Stores the facts this input states about a session, merged with
    /// what the same input chain stated before (one input's records arrive
    /// in file order), and creates the session row on first sight. The
    /// session's own columns are derived from every input's facts
    /// (derive/session.rs), so no arrival order decides them. Pending links
    /// naming the session as their child are resolved here.
    fn upsert_session(&self, s: &Session) -> Result<(i64, bool)> {
        let pk = self.session_row(&s.id, false)?;
        let merged = self
            .choices
            .sessions
            .get(&pk)
            .ok_or_else(|| StoreError::Invalid("unresolved session facts".into()))?;
        self.tx
            .prepare_cached(
                "INSERT INTO session_record(session_pk, input_pk, chain, facts) VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(session_pk, input_pk, chain) DO UPDATE SET facts = excluded.facts",
            )?
            .execute(params![pk, self.input_pk, self.chain, serde_json::to_string(&merged)?])?;
        let main = merged.kind == polylogue_model::SessionKind::Main;
        self.tx.execute("UPDATE message_source SET main=?4 WHERE session_pk=?1 AND input_pk=?2 AND chain=?3 AND main<>?4",params![pk,self.input_pk,self.chain,main])?;
        self.tx
            .prepare_cached("UPDATE session SET changed = ?2, stub = 0 WHERE pk = ?1")?
            .execute(params![pk, self.counter])?;
        Ok((pk, main))
    }

    /// The session's row, created on first sight (`stub` when no record of
    /// its own has arrived yet).
    fn session_row(&self, id: &SessionId, stub: bool) -> Result<i64> {
        let key = id.to_string();
        let pk = self.keys.get(resolve::Kind::Session, key.as_bytes())?.pk;
        if self
            .tx
            .prepare_cached("SELECT pk FROM session WHERE pk=?1")?
            .exists([pk])?
        {
            return Ok(pk);
        }
        self.tx.prepare_cached("INSERT INTO session(pk,id,origin,native_id,kind,cwd_json,changed,harness,stub) VALUES (?1,?2,?3,?4,'main','[]',?5,?3,?6)")?.execute(params![pk,key,id.origin().as_str(),id.native().as_str(),self.counter,stub])?;
        self.tx
            .prepare_cached(
                "UPDATE link SET child_pk = ?1 WHERE child_id = ?2 AND child_pk IS NULL",
            )?
            .execute(params![pk, key])?;
        Ok(pk)
    }

    /// Stores a chain's reading before choosing across inputs.
    fn write_message(
        &self,
        row: &MessageRow,
        session_pk: i64,
        prepared: &PreparedMessage,
        position: (i64, i64),
        main: bool,
    ) -> Result<bool> {
        let m = &row.message;
        let id = m.id.to_string();
        let pk = self.message_pk(&id)?;
        let new = pk.is_none();
        let (version_message, version_prepared) = self
            .choices
            .message_versions
            .get(&BlobHash(prepared.hash))
            .ok_or_else(|| {
                StoreError::Invalid("an incoming message has no resolved value".into())
            })?;
        crate::versions::store_message(
            self.tx,
            version_message,
            version_prepared,
            crate::versions::TextRows {
                packed: self.packed,
                first_build: self.first_build,
                keys: self.keys,
            },
        )?;
        let v = Version {
            time: prepared.time.clone(),
            input_pk: self.input_pk,
            chain: self.chain,
            start: row.offset as i64,
            end: row.end.max(row.offset) as i64,
            complete: !(m.role == Role::Assistant && m.stop_reason.is_none()),
            rank: self.rank,
            hash: prepared.hash.to_vec(),
            pem: prepared.pem.iter().copied().max().unwrap_or(0),
        };
        let pk = match pk {
            Some(pk) => pk,
            None => {
                let pk = self.insert_message(m, session_pk, &v, (self.input_pk, position))?;
                if let Some(first) = self.first_build {
                    first.add_message(&id, pk);
                }
                pk
            }
        };
        if new {
            self.tx.prepare_cached(
                "INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,shown,complete,version_hash,session_pk,main,arrival_rank,seq,record_position)
                 VALUES (?1,?2,?3,?4,?5,?13,?6,?7,?8,?9,?10,?11,?12)"
            )?.execute(params![pk,v.input_pk,v.chain,v.start,v.end,v.complete,v.hash,session_pk,main,self.rank,position.0,position.1,false])?;
            return Ok(false);
        }
        self.tx.prepare_cached(
            "INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,shown,complete,version_hash,session_pk,main,arrival_rank,seq,record_position)
             VALUES (?1,?2,?3,?4,?5,0,?6,?7,?8,?9,?10,?11,?12)
             ON CONFLICT(message_pk,input_pk,chain) DO UPDATE SET source_start=excluded.source_start,
             source_end=excluded.source_end,complete=excluded.complete,version_hash=excluded.version_hash,
             shown=CASE WHEN message_source.version_hash=excluded.version_hash THEN message_source.shown ELSE 0 END,
             session_pk=excluded.session_pk,main=excluded.main,arrival_rank=excluded.arrival_rank,
             seq=excluded.seq,record_position=excluded.record_position"
        )?.execute(params![pk,v.input_pk,v.chain,v.start,v.end,v.complete,v.hash,session_pk,main,self.rank,position.0,position.1])?;
        Ok(false)
    }

    fn refresh_message(&self, pk: i64) -> Result<bool> {
        let choice = self
            .choices
            .messages
            .get(&pk)
            .ok_or_else(|| StoreError::Invalid("unresolved message version".into()))?;
        let winner = choice.as_ref().map(|choice| {
            (
                Version {
                    time: choice
                        .prepared
                        .as_ref()
                        .map(|prepared| prepared.time.clone())
                        .unwrap_or_default(),
                    input_pk: choice.input,
                    chain: choice.chain,
                    start: choice.start,
                    end: choice.end,
                    complete: choice.complete,
                    rank: choice.rank,
                    hash: choice.hash.clone(),
                    pem: 0,
                },
                choice.session,
            )
        });
        let Some((mut v, session_pk)) = winner else {
            if !self.tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1)",
                [pk],
                |r| r.get::<_, bool>(0),
            )? {
                return Ok(false);
            }
            self.clear_blocks(pk)?;
            self.update_event_anchors(pk, false)?;
            self.tx.execute("DELETE FROM message WHERE pk=?1", [pk])?;
            return Ok(true);
        };
        let choice = choice.as_ref().expect("selected message");
        let order = choice.order;
        if !choice.changed {
            self.tx.execute("UPDATE message SET input_pk=?2,source_chain=?3,source_start=?4,source_end=?5,arrival_rank=?6,
                order_input_pk=?7,order_input_id=(SELECT id FROM input_ref WHERE pk=?7),seq=?8,record_position=?9 WHERE pk=?1",
                params![pk,v.input_pk,v.chain,v.start,v.end,v.rank,order.0,order.1.0,order.1.1])?;
            self.tx.execute(
                "UPDATE session SET changed=?2 WHERE pk=?1",
                params![session_pk, self.counter],
            )?;
            self.mark_shown(pk, &v)?;
            self.update_event_anchors(pk, true)?;
            return Ok(false);
        }
        let m = choice.message.as_ref().ok_or_else(|| {
            StoreError::Invalid("a changed version has no prepared message".into())
        })?;
        let prepared = choice
            .prepared
            .as_ref()
            .ok_or_else(|| StoreError::Invalid("a changed version has no prepared body".into()))?;
        v.pem = prepared.pem.iter().copied().max().unwrap_or(0);
        self.tx.execute("UPDATE session SET changed=?2 WHERE pk IN (SELECT session_pk FROM message WHERE pk=?1) OR pk=?3",params![pk,self.counter,session_pk])?;
        self.clear_blocks(pk)?;
        self.update_message(pk, m, session_pk, &v, order)?;
        self.insert_blocks(pk, session_pk, m, prepared)?;
        self.mark_shown(pk, &v)?;
        Ok(true)
    }

    fn mark_shown(&self, pk: i64, v: &Version) -> Result<()> {
        self.tx.execute(
            "UPDATE message_source SET shown=(input_pk=?2 AND chain=?3) WHERE message_pk=?1",
            params![pk, v.input_pk, v.chain],
        )?;
        Ok(())
    }

    fn insert_message(
        &self,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: (i64, (i64, i64)),
    ) -> Result<i64> {
        self.set_message(None, m, session_pk, v, order)
    }

    fn update_message(
        &self,
        pk: i64,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: (i64, (i64, i64)),
    ) -> Result<()> {
        self.set_message(Some(pk), m, session_pk, v, order)
            .map(drop)
    }

    /// Writes every column of the shown version in one statement: an
    /// insert for a new message, an update when another version won.
    fn set_message(
        &self,
        pk: Option<i64>,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: (i64, (i64, i64)),
    ) -> Result<i64> {
        let written_by = polylogue_model::WrittenBy::of(m.role, m.prompt_kind).as_str();
        let words: i64 = m
            .blocks
            .iter()
            .filter_map(Block::text)
            .map(|t| t.split_whitespace().count() as i64)
            .sum();
        let attrs = {
            let mut a = serde_json::Map::new();
            if !m.attrs.is_empty() {
                a.insert("attrs".into(), Json::Object(m.attrs.clone()));
            }
            if let Some(c) = &m.reported_cost_usd {
                a.insert("reported_cost_usd".into(), c.clone().into());
            }
            if !m.production.is_empty() {
                a.insert(
                    "production".into(),
                    serde_json::to_value(&m.production).unwrap_or_default(),
                );
            }
            (!a.is_empty()).then(|| serde_json::Value::Object(a).to_string())
        };
        let u = &m.usage;
        let n = |x: Option<u64>| x.map(|x| x as i64);
        let sql = if pk.is_some() {
            "UPDATE message SET session_pk = ?2, order_input_pk = ?3, seq = ?4, arrival_rank = ?5,
                 role = ?6, written_by = ?7, prompt_kind = ?8, model = ?9, at = ?10,
                 parent_id = ?11, input_pk = ?12, source_chain = ?13, source_start = ?14,
                 source_end = ?26, in_progress = ?15, input_tokens = ?16, output_tokens = ?17,
                 cache_read_tokens = ?18, cache_write_tokens = ?19, reasoning_tokens = ?20,
                 service_tier = ?21, stop_reason = ?22, duration_ms = ?23, attrs_json = ?24,
                 words = ?25, cost_nusd = ?33, provider_cost_nusd = ?27,
                 cache_write_5m_tokens = ?28, cache_write_1h_tokens = ?29, speed = ?30,
                 record_position = ?31, pem = ?32, at_time = ?35,
                 order_input_id = (SELECT id FROM input_ref WHERE pk = ?3)
             WHERE pk = ?1 AND (?34 IS NULL OR pk=?34)"
        } else {
            "INSERT INTO message(pk, id, session_pk, order_input_pk, seq, arrival_rank, role, written_by,
                 prompt_kind, model, at, parent_id, input_pk, source_chain, source_start,
                 source_end, in_progress, input_tokens, output_tokens, cache_read_tokens,
                 cache_write_tokens, reasoning_tokens, service_tier, stop_reason, duration_ms,
                 attrs_json, words, provider_cost_nusd,
                 cache_write_5m_tokens, cache_write_1h_tokens, speed, record_position, pem,
                 cost_nusd, order_input_id, at_time)
             VALUES (?34, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?26, ?15, ?16,
                 ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?27, ?28, ?29, ?30, ?31, ?32, ?33,
                 (SELECT id FROM input_ref WHERE pk = ?3), ?35)"
        };
        let first: rusqlite::types::Value = match pk {
            Some(pk) => pk.into(),
            None => m.id.to_string().into(),
        };
        self.tx.prepare_cached(sql)?.execute(params![
            first,
            session_pk,
            order.0,
            order.1.0,
            v.rank,
            name(&m.role),
            written_by,
            m.prompt_kind.map(|k| name(&k)),
            m.model,
            ms(m.at),
            m.parent.as_ref().map(ToString::to_string),
            v.input_pk,
            v.chain,
            v.start,
            i64::from(!v.complete),
            n(u.input),
            n(u.output),
            n(u.cached_input),
            n(u.cache_write),
            n(u.reasoning),
            u.service_tier,
            m.stop_reason,
            n(m.duration_ms),
            attrs,
            words,
            v.end,
            reported_nusd(m.reported_cost_usd.as_ref()),
            // The five-minute part is known only when the one-hour part is.
            u.cache_write_1h
                .and(u.cache_write)
                .map(|w| w.saturating_sub(u.cache_write_1h.unwrap_or(0)) as i64),
            n(u.cache_write_1h),
            u.speed,
            order.1.1,
            v.pem,
            // The cost derivation's value, written with the row so a first
            // build's derive does not rewrite every priced message.
            polylogue_model::Catalog::builtin().cost_nusd(m.model.as_deref(), u),
            Some(
                self.keys
                    .get(resolve::Kind::Message, m.id.to_string().as_bytes())?
                    .pk
            ),
            v.time,
        ])?;
        let pk = pk.unwrap_or_else(|| self.tx.last_insert_rowid());
        self.update_event_anchors(pk, true)?;
        Ok(pk)
    }

    /// Events can precede their message, or survive its replacement.
    fn update_event_anchors(&self, message_pk: i64, present: bool) -> Result<()> {
        let (id, seq): (String, i64) = self
            .tx
            .prepare_cached("SELECT id,seq FROM message WHERE pk=?1")?
            .query_row([message_pk], |r| Ok((r.get(0)?, r.get(1)?)))?;
        let pk = present.then_some(message_pk);
        let seq = if present { seq } else { 0 };
        let sessions = self
            .tx
            .prepare_cached(
                "UPDATE event SET after_message_pk=?2,
             seq=CASE WHEN seq_from_message=1 THEN ?3 ELSE seq END
             WHERE after_message_id=?1 AND (after_message_pk IS NOT ?2
                OR (seq_from_message=1 AND seq IS NOT ?3)) RETURNING session_pk",
            )?
            .query_map(params![id, pk, seq], |r| r.get::<_, Option<i64>>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for session in sessions.into_iter().flatten() {
            self.tx
                .prepare_cached("UPDATE session SET changed=?2 WHERE pk=?1 AND changed IS NOT ?2")?
                .execute(params![session, self.counter])?;
        }
        Ok(())
    }

    /// Removes a message's blocks, keeping search entries and tool-call
    /// pairs in step, before another version replaces them. It runs before
    /// the message row changes: a version from another session moves the
    /// message, and its old blocks must not outlive the move.
    fn clear_blocks(&self, message_pk: i64) -> Result<()> {
        let texts: Vec<i64> = self
            .tx
            .prepare_cached(
                "SELECT DISTINCT text_id FROM block WHERE message_pk=?1 AND text_id IS NOT NULL",
            )?
            .query_map([message_pk], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        // Calls of the old blocks go, with their trigram entries; results of
        // the old blocks are unpaired from their calls.
        let old_calls: Vec<(i64, Option<String>, Option<String>)> = self
            .tx
            .prepare_cached(
                "SELECT t.pk, t.command, t.path FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
                 WHERE b.message_pk = ?1",
            )?
            .query_map([message_pk], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (pk, command, path) in old_calls {
            if self.first_build.is_none() {
                self.tx
                    .prepare_cached(
                        "INSERT INTO tool_call_trigram(tool_call_trigram, rowid, command, path)
                         VALUES ('delete', ?1, ?2, ?3)",
                    )?
                    .execute(params![pk, command, path])?;
            }
            self.tx
                .prepare_cached("DELETE FROM tool_call WHERE pk = ?1")?
                .execute([pk])?;
        }
        self.tx
            .prepare_cached(
                "UPDATE tool_call SET result_block_pk = NULL,result_at=NULL,reported_duration_ms=NULL, outcome = 'unknown',
                     outcome_reason = 'no_result', exit_code = NULL
                 WHERE result_block_pk IN (SELECT pk FROM block WHERE message_pk = ?1)",
            )?
            .execute([message_pk])?;
        self.tx
            .prepare_cached("DELETE FROM block WHERE message_pk = ?1")?
            .execute([message_pk])?;
        for text in texts {
            crate::text_rows::unindex_unused(self.tx, text, self.first_build.is_some())?;
        }
        Ok(())
    }

    fn insert_blocks(
        &self,
        message_pk: i64,
        session_pk: i64,
        m: &Message,
        prepared: &PreparedMessage,
    ) -> Result<()> {
        if prepared.blocks.len() != m.blocks.len() {
            return Err(StoreError::Invalid(
                "prepared message differs from its parsed blocks".into(),
            ));
        }
        prepared.projection.validate(m.blocks.len())?;
        for (idx, ((block, body), data)) in m
            .blocks
            .iter()
            .zip(&prepared.blocks)
            .zip(&prepared.data)
            .enumerate()
        {
            let text_id = prepared.texts[idx]
                .as_ref()
                .map(|hash| self.keys.get(resolve::Kind::Text, hash).map(|key| key.pk))
                .transpose()?;
            let stored = StoredBlock {
                body,
                data: data.as_deref(),
                text_id,
            };
            self.insert_block(
                message_pk,
                session_pk,
                (idx as i64, prepared.projection.block_ordinals[idx]),
                block,
                stored,
                ms(m.at),
            )?;
        }
        Ok(())
    }

    fn insert_block(
        &self,
        message_pk: i64,
        session_pk: i64,
        position: (i64, u64),
        block: &Block,
        StoredBlock {
            body,
            data,
            text_id,
        }: StoredBlock<'_>,
        at: Option<i64>,
    ) -> Result<()> {
        let (idx, ordinal) = position;
        let bit = if matches!(block, Block::ToolResult { .. }) {
            2
        } else {
            1
        };
        if let Some(id) = text_id {
            crate::text_rows::activate(self.tx, id, bit, self.first_build.is_some())?;
        }
        let (kind, call_id, attachment) = match block {
            Block::Text { .. } => ("text", None, None),
            Block::Thinking { .. } => ("thinking", None, None),
            Block::ToolCall { call_id, .. } => ("tool_call", call_id.as_deref(), None),
            Block::ToolResult { call_id, .. } => ("tool_result", call_id.as_deref(), None),
            Block::Image(a) => ("image", None, Some(a)),
            Block::File(a) => ("file", None, Some(a)),
            Block::Other { .. } => ("other", None, None),
        };
        let (signature, encrypted) = match block {
            Block::Thinking {
                signature,
                redacted,
                ..
            } => (signature.as_deref(), *redacted),
            _ => (None, false),
        };
        let other_type = match block {
            Block::Other { type_name, .. } => Some(type_name.as_str()),
            _ => None,
        };
        self.tx
            .prepare_cached(
                "INSERT INTO block(message_pk, session_pk, idx, kind, text_id, body, call_id,
                     blob_hash, media_type, file_name, size, url, direction, signature, encrypted,
                     other_type, data, ordinal)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            )?
            .execute(params![
                message_pk,
                session_pk,
                idx,
                kind,
                text_id,
                body,
                call_id,
                attachment.and_then(|a| a.blob).map(|h| h.0.to_vec()),
                attachment.and_then(|a| a.media_type.as_deref()),
                attachment.and_then(|a| a.name.as_deref()),
                attachment.and_then(|a| a.size).map(|s| s as i64),
                attachment.and_then(|a| a.url.as_deref()),
                attachment.map(|a| name(&a.direction)),
                signature,
                i64::from(encrypted),
                other_type,
                data,
                ordinal.to_string(),
            ])?;
        let block_pk = self.tx.last_insert_rowid();

        match block {
            Block::ToolCall {
                call_id,
                tool,
                action,
                command,
                path,
                run_by_script,
                ..
            } => self.insert_call(
                block_pk,
                session_pk,
                call_id,
                tool,
                *action,
                command,
                path,
                at,
                *run_by_script,
            ),
            Block::ToolResult {
                call_id: Some(call_id),
                outcome,
                reported_duration_ms,
                ..
            } => {
                let (outcome, reason, exit) = outcome_columns(outcome);
                self.tx
                    .prepare_cached(
                        "UPDATE tool_call SET result_block_pk = ?1, outcome = ?2, outcome_reason = ?3,
                             exit_code = ?4, result_at=?7,reported_duration_ms=?8
                         WHERE session_pk = ?5 AND call_id = ?6",
                    )?
                    .execute(params![block_pk, outcome, reason, exit, session_pk, call_id,at,reported_duration_ms.map(|ms|ms as i64)])?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_call(
        &self,
        block_pk: i64,
        session_pk: i64,
        call_id: &Option<String>,
        tool: &Option<String>,
        action: ToolAction,
        command: &Option<String>,
        path: &Option<String>,
        at: Option<i64>,
        run_by_script: bool,
    ) -> Result<()> {
        let tool = tool.as_deref().unwrap_or("unknown");
        self.tx
            .prepare_cached(
                "INSERT INTO tool_call(session_pk, call_block_pk, call_id, at, tool, action,
                     command, path, outcome, outcome_reason, run_by_script, command_norm)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'unknown', 'no_result', ?9, ?10)",
            )?
            .execute(params![
                session_pk,
                block_pk,
                call_id,
                at,
                tool,
                action.as_str(),
                command,
                path,
                run_by_script,
                // The tool-call derivation's value, written with the row so a
                // first build's derive does not rewrite every shell call.
                (action == polylogue_model::ToolAction::Shell)
                    .then(|| command
                        .as_deref()
                        .and_then(polylogue_model::command::normalize))
                    .flatten()
            ])?;
        let pk = self.tx.last_insert_rowid();
        if self.first_build.is_none() {
            self.tx
                .prepare_cached(
                    "INSERT INTO tool_call_trigram(rowid, command, path) VALUES (?1, ?2, ?3)",
                )?
                .execute(params![pk, command, path])?;
        }
        // The result may have been stored first (a late call record, or a
        // message replaced after its result arrived).
        let Some(call_id) = call_id else {
            return Ok(());
        };
        let result: Option<(i64, Vec<u8>, Option<i64>)> = self
            .tx
            .prepare_cached(
                "SELECT b.pk,b.body,m.at FROM block b JOIN message m ON m.pk=b.message_pk
                 WHERE b.session_pk = ?1 AND b.call_id = ?2 AND b.kind = 'tool_result'
                 ORDER BY b.pk LIMIT 1",
            )?
            .query_row(params![session_pk, call_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .optional()?;
        if let Some((result_pk, body, result_at)) = result
            && let Ok(Block::ToolResult {
                outcome,
                reported_duration_ms,
                ..
            }) = unpack_block(&body)
        {
            let (outcome, reason, exit) = outcome_columns(&outcome);
            self.tx
                .prepare_cached(
                    "UPDATE tool_call SET result_block_pk = ?2, outcome = ?3, outcome_reason = ?4,
                         exit_code = ?5,result_at=?6,reported_duration_ms=?7
                     WHERE pk = ?1",
                )?
                .execute(params![
                    pk,
                    result_pk,
                    outcome,
                    reason,
                    exit,
                    result_at,
                    reported_duration_ms.map(|ms| ms as i64)
                ])?;
        }
        Ok(())
    }

    fn write_event(&self, index: usize) -> Result<()> {
        let id = &self.choices.event_ids[index];
        let (pk, source) = &self.choices.event_sources[index];
        self.tx.prepare_cached("INSERT OR IGNORE INTO event(pk,id,session_pk,seq,kind,at,input_pk,source_chain,source_start,source_end,fields,data)
            VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)")?
            .execute(params![pk,id,source.session,source.seq.unwrap_or(0),source.kind,source.at,self.input_pk,self.chain,source.start,source.end,source.fields,source.body])?;
        if let Some(first) = self.first_build {
            first.add_event(id, *pk);
        }
        self.store_event_value(index)
    }

    fn write_link(&self, l: &Link, index: usize) -> Result<()> {
        self.store_link(l, index)
    }

    /// Parse position and record counts for the chain, committed with the rows
    /// they cover. A step from offset 0 replaces the chain's counts; a
    /// continuing step adds to them.
    fn write_parse_state(&self, b: &Batch, p: &ParseStep) -> Result<()> {
        let a = &p.counts;
        let used: u64 = a.used.values().sum();
        let skipped: u64 = a.skipped.values().sum();
        let unreadable: u64 = a.unreadable.values().sum();
        let status = if p.source_prefix_missing {
            "partial"
        } else if p.at_end && p.parsed_upto < b.length {
            "waiting"
        } else if p.no_parser_yet {
            "no_parser_yet"
        } else if p.parsed_upto >= b.length {
            "done"
        } else if p.at_end {
            "waiting"
        } else {
            "parsing"
        };
        let detail = (status == "waiting").then(|| b.length.to_string());
        let whole = p.reset;
        let sql = if whole {
            "INSERT INTO parse_state(input_pk, chain, status, parsed_upto, records,
                 used, skipped, unreadable, detail, reading_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(input_pk, chain) DO UPDATE SET
                 status = excluded.status, parsed_upto = excluded.parsed_upto, detail = excluded.detail, reading_version=excluded.reading_version,
                 records = excluded.records, used = excluded.used,
                 skipped = excluded.skipped, unreadable = excluded.unreadable"
        } else {
            "INSERT INTO parse_state(input_pk, chain, status, parsed_upto, records,
                 used, skipped, unreadable, detail, reading_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(input_pk, chain) DO UPDATE SET
                 status = excluded.status, parsed_upto = excluded.parsed_upto, detail = excluded.detail, reading_version=excluded.reading_version,
                 records = records + excluded.records, used = used + excluded.used,
                 skipped = skipped + excluded.skipped,
                 unreadable = unreadable + excluded.unreadable"
        };
        self.tx.prepare_cached(sql)?.execute(params![
            self.input_pk,
            self.chain,
            status,
            p.parsed_upto as i64,
            a.records as i64,
            used as i64,
            skipped as i64,
            unreadable as i64,
            detail,
            p.reading_version.as_str(),
        ])?;
        let groups = [
            ("skipped:", &a.skipped, 1),
            ("unreadable:", &a.unreadable, 0),
            ("field:", &a.unread_fields, 0),
        ];
        for (prefix, map, counted_only) in groups {
            for (what, count) in map {
                self.tx
                    .prepare_cached(
                        "INSERT INTO unused(input_pk, chain, format, what, count, counted_only)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                         ON CONFLICT(input_pk, chain, what) DO UPDATE SET count = count + excluded.count",
                    )?
                    .execute(params![
                        self.input_pk,
                        self.chain,
                        b.input.format,
                        format!("{prefix}{what}"),
                        *count as i64,
                        counted_only,
                    ])?;
            }
        }
        Ok(())
    }
}

pub(super) fn outcome_columns(o: &ToolOutcome) -> (&'static str, Option<String>, Option<i64>) {
    let (reason, exit) = match o {
        ToolOutcome::Ok | ToolOutcome::Error { reason: None } => (None, None),
        ToolOutcome::Error {
            reason: Some(reason),
        } => match reason {
            ErrorReason::ExitCode(code) => (Some("exit_code".into()), Some(*code)),
            ErrorReason::Denied(label) => (Some(format!("denied:{label}")), None),
            ErrorReason::Interrupted => (Some("interrupted".into()), None),
            ErrorReason::TimedOut => (Some("timed_out".into()), None),
        },
        ToolOutcome::NoResult => (Some("no_result".into()), None),
        ToolOutcome::Unknown { reason } => (Some(name(reason)), None),
    };
    (o.stored().as_str(), reason, exit)
}

/// A provider's reported amount is an observed fact, separate from catalog pricing.
fn reported_nusd(usd: Option<&serde_json::Number>) -> Option<i64> {
    let usd = usd?.as_f64()?;
    usd.is_finite().then(|| (usd * 1e9).round() as i64)
}
