//! `index.db` rows for parsed records: sessions, messages and their
//! sources, blocks, tool calls, events, links and parse positions.

use super::*;
use crate::first_build::FirstBuild;
use std::collections::BTreeMap;

/// Writes a parse step's rows; returns the number of message rows written.
pub(super) fn write_rows(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    chain: i64,
    counter: i64,
    first_build: Option<&FirstBuild>,
) -> Result<u64> {
    p.prepared.validate(&p.output)?;
    tx.prepare_cached("INSERT INTO input_ref(id,format) VALUES (?1,?2) ON CONFLICT(id) DO UPDATE SET format=excluded.format WHERE format IS NOT excluded.format")?
        .execute(params![&b.input.id,&b.input.format])?;
    let input_pk: i64 = tx
        .prepare_cached("SELECT pk FROM input_ref WHERE id = ?1")?
        .query_row([&b.input.id], |r| r.get(0))?;
    let rows = RowWriter {
        tx,
        input_pk,
        input_id: &b.input.id,
        chain,
        repeats_whole_conversations: p.repeats_whole_conversations,
        rank: i64::from(b.input.arrival_rank),
        counter,
        packed: &p.prepared.texts,
        first_build,
    };
    // A step parsed from the chain's first byte restates everything the
    // chain said: its earlier events and session facts go first.
    if p.reset {
        tx.prepare_cached(
            "DELETE FROM event WHERE input_pk = ?1 AND (source_chain = ?2 OR source_chain IS NULL)",
        )?
        .execute(params![input_pk, chain])?;
        tx.prepare_cached("DELETE FROM session_record WHERE input_pk = ?1 AND chain = ?2")?
            .execute(params![input_pk, chain])?;
    }
    let mut sessions: HashMap<String, i64> = HashMap::new();
    for s in &p.output.sessions {
        let pk = rows.upsert_session(s)?;
        sessions.insert(s.id.to_string(), pk);
    }
    let source = crate::documents::Source {
        input_pk,
        input_id: &b.input.id,
        chain,
        repeats_whole_conversations: p.repeats_whole_conversations,
        rank: i64::from(b.input.arrival_rank),
        counter,
        packed: &p.prepared.texts,
        first_build,
    };
    for (i, row) in p.output.documents.iter().enumerate() {
        let fallback;
        let prepared = match p.prepared.documents.get(i) {
            Some(prepared) => prepared,
            None => {
                fallback = PreparedDocument::new(&row.document)?;
                &fallback
            }
        };
        crate::documents::write(tx, &source, row, prepared)?;
    }
    let mut written = 0;
    let mut activity: HashMap<i64, i64> = HashMap::new();
    // Messages one record holds share its offset; their order in the
    // record breaks the tie.
    let mut in_record: HashMap<u64, i64> = HashMap::new();
    for (i, row) in p.output.messages.iter().enumerate() {
        let session_pk = rows.session_pk(&mut sessions, &row.message.session)?;
        let fallback;
        let prepared = match p.prepared.messages.get(i) {
            Some(prepared) => prepared,
            None => {
                fallback = PreparedMessage::new(&row.message)?;
                &fallback
            }
        };
        let position = match p.order_from {
            Some(from) => ((from + i as u64) as i64, 0),
            None if p.repeats_whole_conversations => (i as i64, 0),
            None => {
                let next = in_record.entry(row.offset).or_default();
                *next += 1;
                (row.offset as i64, *next - 1)
            }
        };
        // A kind this step states needs no lookup.
        let main = match p
            .output
            .sessions
            .iter()
            .find(|s| s.id == row.message.session)
        {
            Some(session) => session.kind == polylogue_model::SessionKind::Main,
            None => rows.session_kind(session_pk)? == "main",
        };
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
    let mut event_occurrences = BTreeMap::<(u64, u64), u64>::new();
    for (i, e) in p.output.events.iter().enumerate() {
        let occurrence = if e.session.is_none() {
            if rows.input_id.is_empty() {
                return Err(StoreError::Invalid(
                    "an event about its input has no input identity".into(),
                ));
            }
            let range = e.range.filter(|(start, end)| start < end).ok_or_else(|| {
                StoreError::Invalid("an event about its input has no record range".into())
            })?;
            if e.after.is_some() {
                return Err(StoreError::Invalid(
                    "an event about its input cannot follow a conversation message".into(),
                ));
            }
            let next = event_occurrences.entry(range).or_default();
            let occurrence = *next;
            *next = next
                .checked_add(1)
                .ok_or_else(|| StoreError::Invalid("too many events in one record".into()))?;
            occurrence
        } else {
            0
        };
        let session_pk = e
            .session
            .as_ref()
            .map(|id| rows.session_pk(&mut sessions, id))
            .transpose()?;
        let prepared;
        let packed = match p.prepared.events.get(i) {
            Some(prepared) if p.prepared.events.len() == p.output.events.len() => prepared,
            _ => {
                prepared = pack_event(e)?;
                &prepared
            }
        };
        rows.write_event(e, session_pk, packed, p.order_from, p.start, occurrence)?;
        if let Some(session_pk) = session_pk.filter(|_| e.kind.starts_with("hook_")) {
            // Prepared events release their JSON payload before writing. Live
            // facts must read the same payload as the stored event.
            let event = Event {
                data: crate::encoding::unpack_json(&packed.1)?,
                ..e.clone()
            };
            crate::live::apply(tx, session_pk, &event, counter)?;
        }
    }
    for l in &p.output.links {
        rows.write_link(l)?;
    }
    for l in &p.output.repo_lookups {
        crate::repo_lookup::upsert_lookup(tx, l, input_pk, chain, counter)?;
    }
    // Every session these records touched is derived again.
    for pk in sessions.values() {
        tx.prepare_cached("UPDATE session SET changed = ?2 WHERE pk = ?1 AND changed IS NOT ?2")?
            .execute(params![pk, counter])?;
    }
    for u in &p.output.background {
        let (outcome, reason, exit) = outcome_columns(&u.outcome);
        tx.prepare_cached(
            "UPDATE tool_call SET outcome = ?1, outcome_reason = ?2, exit_code = ?3
             WHERE session_pk = (SELECT pk FROM session WHERE id = ?4) AND call_id = ?5",
        )?
        .execute(params![
            outcome,
            reason,
            exit,
            u.session.to_string(),
            u.call_id
        ])?;
    }
    for pk in sessions.values() {
        tx.prepare_cached(
            "INSERT OR IGNORE INTO parsed(input_pk, chain, session_pk) VALUES (?1, ?2, ?3)",
        )?
        .execute(params![input_pk, chain, pk])?;
    }
    rows.write_parse_state(b, p)?;
    Ok(written)
}

/// A block's packed body and structured value, as `block` stores them.
#[derive(Clone, Copy)]
struct StoredBlock<'a> {
    body: &'a [u8],
    data: Option<&'a [u8]>,
}

pub(super) struct RowWriter<'a, 't> {
    tx: &'a Transaction<'t>,
    input_pk: i64,
    input_id: &'a str,
    chain: i64,
    /// The step's records restate whole conversations ([`ParseStep::restates`]).
    repeats_whole_conversations: bool,
    rank: i64,
    counter: i64,
    packed: &'a HashMap<[u8; 32], Vec<u8>>,
    /// Set during a first build ([`Writer::start`]): search entries,
    /// trigram entries and secondary indexes are built once, in bulk, when
    /// it ends ([`Inner::build_deferred`]), and rows are found by natural
    /// key through it ([`FirstBuild`]).
    first_build: Option<&'a FirstBuild>,
}

/// The shown version of a message, as compared with a new one.
pub(super) struct Version {
    input_pk: i64,
    chain: i64,
    start: i64,
    end: i64,
    /// Whether the version's session is a main session. A spawned
    /// session's file can repeat its parent's messages; the parent's own
    /// record is the original.
    main: bool,
    complete: bool,
    rank: i64,
    hash: Vec<u8>,
    /// The most any block of the version holds of PEM private key blocks
    /// ([`crate::ordered_blocks::pem`]). Written with the version, never
    /// compared.
    pem: i64,
}

impl Version {
    /// Architecture 5.5 and 5.5a. Within one input a newer revision chain
    /// wins (a rewrite), and within a chain the caller has already kept the
    /// right record. Across inputs: the originating session's own record
    /// over a copy in a session it spawned, then a complete version, then
    /// the more faithful arrival, then the smaller content hash, so the choice
    /// does not depend on which input arrived first.
    fn beats(&self, cur: &Version) -> bool {
        if self.input_pk == cur.input_pk {
            return self.chain >= cur.chain;
        }
        if self.main != cur.main {
            return self.main;
        }
        if self.complete != cur.complete {
            return self.complete;
        }
        if self.rank != cur.rank {
            return self.rank < cur.rank;
        }
        self.hash < cur.hash
    }
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
        match self.first_build {
            Some(first_build) => first_build.message(self.tx, id),
            None => Ok(self
                .tx
                .prepare_cached("SELECT pk FROM message WHERE id = ?1")?
                .query_row([id], |r| r.get(0))
                .optional()?),
        }
    }

    /// The event with ID `id`.
    fn event_pk(&self, id: &str) -> Result<Option<i64>> {
        match self.first_build {
            Some(first_build) => first_build.event(self.tx, id),
            None => Ok(self
                .tx
                .prepare_cached("SELECT pk FROM event WHERE id = ?1")?
                .query_row([id], |r| r.get(0))
                .optional()?),
        }
    }

    fn session_pk(&self, cache: &mut HashMap<String, i64>, id: &SessionId) -> Result<i64> {
        let key = id.to_string();
        if let Some(pk) = cache.get(&key) {
            return Ok(*pk);
        }
        let existing = self
            .tx
            .prepare_cached("SELECT pk FROM session WHERE id = ?1")?
            .query_row([&key], |r| r.get::<_, i64>(0))
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
    fn upsert_session(&self, s: &Session) -> Result<i64> {
        let pk = self.session_row(&s.id, false)?;
        let stored: Option<String> = self
            .tx
            .prepare_cached(
                "SELECT facts FROM session_record WHERE session_pk = ?1 AND input_pk = ?2 AND chain = ?3",
            )?
            .query_row(params![pk, self.input_pk, self.chain], |r| r.get(0))
            .optional()?;
        let merged = match stored
            .as_deref()
            .map(serde_json::from_str::<Session>)
            .transpose()?
        {
            Some(mut cur) => {
                let before = cur.clone();
                cur.merge(s);
                if cur == before {
                    return Ok(pk);
                }
                cur
            }
            None => s.clone(),
        };
        self.tx
            .prepare_cached(
                "INSERT INTO session_record(session_pk, input_pk, chain, facts) VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(session_pk, input_pk, chain) DO UPDATE SET facts = excluded.facts",
            )?
            .execute(params![pk, self.input_pk, self.chain, serde_json::to_string(&merged)?])?;
        self.tx
            .prepare_cached("UPDATE session SET changed = ?2, stub = 0 WHERE pk = ?1")?
            .execute(params![pk, self.counter])?;
        Ok(pk)
    }

    /// The session's row, created on first sight (`stub` when no record of
    /// its own has arrived yet).
    fn session_row(&self, id: &SessionId, stub: bool) -> Result<i64> {
        let key = id.to_string();
        if let Some(pk) = self
            .tx
            .prepare_cached("SELECT pk FROM session WHERE id = ?1")?
            .query_row([&key], |r| r.get::<_, i64>(0))
            .optional()?
        {
            return Ok(pk);
        }
        self.tx
            .prepare_cached(
                "INSERT INTO session(id, origin, native_id, kind, cwd_json, changed, harness, stub)
                 VALUES (?1, ?2, ?3, 'main', '[]', ?4, ?2, ?5)",
            )?
            .execute(params![
                key,
                id.origin().as_str(),
                id.native().as_str(),
                self.counter,
                stub
            ])?;
        let pk = self.tx.last_insert_rowid();
        self.tx
            .prepare_cached(
                "UPDATE link SET child_pk = ?1 WHERE child_id = ?2 AND child_pk IS NULL",
            )?
            .execute(params![pk, key])?;
        Ok(pk)
    }

    /// Records one version of a message and makes it the shown one when it
    /// beats the current one. Returns whether the message row was written.
    fn write_message(
        &self,
        row: &MessageRow,
        session_pk: i64,
        prepared: &PreparedMessage,
        position: (i64, i64),
        main: bool,
    ) -> Result<bool> {
        let m = &row.message;
        let new = Version {
            input_pk: self.input_pk,
            chain: self.chain,
            start: row.offset as i64,
            end: row.end.max(row.offset) as i64,
            main,
            complete: !(m.role == Role::Assistant && m.stop_reason.is_none()),
            rank: self.rank,
            hash: prepared.hash.to_vec(),
            pem: prepared.pem.iter().copied().max().unwrap_or(0),
        };
        let id = m.id.to_string();
        let cur = match self.message_pk(&id)? {
            None => None,
            Some(pk) => self
                .tx
                .prepare_cached(
                    "SELECT m.pk, m.order_input_pk, m.seq, m.order_input_id, m.input_pk, m.source_chain,
                        m.source_start, m.in_progress, m.arrival_rank, s.version_hash, m.session_pk, ss.kind,
                        m.record_position
                 FROM message m LEFT JOIN message_source s
                   ON s.message_pk = m.pk AND s.shown = 1
                 JOIN session ss ON ss.pk = m.session_pk
                 WHERE m.pk = ?1",
                )?
                .query_row([pk], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    (
                        r.get::<_, i64>(1)?,
                        (r.get::<_, i64>(2)?, r.get::<_, i64>(12)?),
                        r.get::<_, String>(3)?,
                    ),
                    Version {
                        input_pk: r.get(4)?,
                        chain: r.get::<_, Option<i64>>(5)?.unwrap_or(0),
                        start: r.get::<_, Option<i64>>(6)?.unwrap_or(0),
                        end: 0,
                        main: r.get::<_, String>(11)? == "main",
                        complete: r.get::<_, i64>(7)? == 0,
                        rank: r.get(8)?,
                        hash: r.get::<_, Option<Vec<u8>>>(9)?.unwrap_or_default(),
                        pem: 0,
                    },
                    (r.get::<_, i64>(10)?, r.get::<_, String>(11)?),
                ))
                })
                .optional()?,
        };

        let Some((pk, (order_input, order_seq, order_id), shown, (cur_session, cur_kind))) = cur
        else {
            let pk = self.insert_message(m, session_pk, &new, (self.input_pk, position))?;
            if let Some(first_build) = self.first_build {
                first_build.add_message(&id, pk);
            }
            self.upsert_source(pk, &new, true)?;
            self.insert_blocks(pk, session_pk, m, prepared)?;
            return Ok(true);
        };

        // Within one chain the earliest record of a message wins: a later
        // duplicate (a repeated uuid) is not another version. In a chain of
        // repeating whole conversations records each record restates the conversation, so the later
        // record replaces the earlier one.
        let prior: Option<i64> = self
            .tx
            .prepare_cached(
                "SELECT source_start FROM message_source
                 WHERE message_pk = ?1 AND input_pk = ?2 AND chain = ?3",
            )?
            .query_row(params![pk, self.input_pk, self.chain], |r| r.get(0))
            .optional()?;
        if prior.is_some_and(|start| {
            if self.repeats_whole_conversations {
                new.start < start
            } else {
                new.start > start
            }
        }) {
            return Ok(false);
        }
        let order = (order_input, order_seq);
        let new_order = if (self.repeats_whole_conversations && order_id == self.input_id)
            || (self.input_id, position) < (order_id.as_str(), order_seq)
        {
            (self.input_pk, position)
        } else {
            order
        };
        if !new.beats(&shown) {
            self.upsert_source(pk, &new, false)?;
            if new_order != order {
                self.tx
                    .prepare_cached(
                        "UPDATE message SET order_input_pk = ?2, seq = ?3, record_position = ?4,
                             order_input_id = (SELECT id FROM input_ref WHERE pk = ?2) WHERE pk = ?1",
                    )?
                    .execute(params![pk, new_order.0, new_order.1.0, new_order.1.1])?;
            }
            return Ok(false);
        }
        // A message is stored once, in the session where it originated
        //. A spawned session's file can repeat its
        // parent's messages (a compaction call, a side question replaying
        // the history); such a copy may become the shown version, but it
        // never moves the message out of a main session.
        let declared = session_pk;
        let session_pk = if cur_session != session_pk && cur_kind == "main" && !new.main {
            cur_session
        } else {
            session_pk
        };
        if cur_session != declared {
            // Both sessions change: the one the version came in and the
            // one the message was in. Each is derived again and its reads
            // see the change, and the input is recorded as one the
            // message's session was parsed from.
            self.tx
                .prepare_cached("UPDATE session SET changed = ?3 WHERE pk IN (?1, ?2)")?
                .execute(params![cur_session, declared, self.counter])?;
            self.tx
                .prepare_cached(
                    "INSERT OR IGNORE INTO parsed(input_pk, chain, session_pk) VALUES (?1, ?2, ?3)",
                )?
                .execute(params![self.input_pk, self.chain, session_pk])?;
        }
        self.tx
            .prepare_cached("UPDATE message_source SET shown = 0 WHERE message_pk = ?1")?
            .execute([pk])?;
        self.upsert_source(pk, &new, true)?;
        let unchanged = new.hash == shown.hash && cur_session == session_pk;
        if !unchanged {
            self.clear_blocks(pk)?;
        }
        self.update_message(pk, m, session_pk, &new, new_order)?;
        if !unchanged {
            self.insert_blocks(pk, session_pk, m, prepared)?;
        }
        Ok(true)
    }

    fn upsert_source(&self, pk: i64, v: &Version, shown: bool) -> Result<()> {
        self.tx
            .prepare_cached(
                "INSERT INTO message_source(message_pk, input_pk, chain, source_start, source_end,
                     shown, complete, version_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(message_pk, input_pk, chain) DO UPDATE SET
                     source_start = excluded.source_start, source_end = excluded.source_end,
                     shown = excluded.shown, complete = excluded.complete,
                     version_hash = excluded.version_hash",
            )?
            .execute(params![
                pk,
                v.input_pk,
                v.chain,
                v.start,
                v.end,
                i64::from(shown),
                i64::from(v.complete),
                v.hash
            ])?;
        Ok(())
    }

    // Placement uses this input's declared kind before the merged session
    // is derived; another input's kind cannot classify these rows.
    fn session_kind(&self, session_pk: i64) -> Result<String> {
        Ok(self
            .tx
            .prepare_cached(
                "SELECT COALESCE((SELECT json_extract(facts,'$.kind') FROM session_record
                 WHERE session_pk=?1 AND input_pk=?2 AND chain=?3),kind)
                 FROM session WHERE pk=?1",
            )?
            .query_row(params![session_pk, self.input_pk, self.chain], |r| r.get(0))?)
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
                 record_position = ?31, pem = ?32,
                 order_input_id = (SELECT id FROM input_ref WHERE pk = ?3)
             WHERE pk = ?1"
        } else {
            "INSERT INTO message(id, session_pk, order_input_pk, seq, arrival_rank, role, written_by,
                 prompt_kind, model, at, parent_id, input_pk, source_chain, source_start,
                 source_end, in_progress, input_tokens, output_tokens, cache_read_tokens,
                 cache_write_tokens, reasoning_tokens, service_tier, stop_reason, duration_ms,
                 attrs_json, words, provider_cost_nusd,
                 cache_write_5m_tokens, cache_write_1h_tokens, speed, record_position, pem,
                 cost_nusd, order_input_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?26, ?15, ?16,
                 ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?27, ?28, ?29, ?30, ?31, ?32, ?33,
                 (SELECT id FROM input_ref WHERE pk = ?3))"
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
        ])?;
        Ok(pk.unwrap_or_else(|| self.tx.last_insert_rowid()))
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
                "UPDATE tool_call SET result_block_pk = NULL, outcome = 'unknown',
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
        for (idx, ((block, body), data)) in m
            .blocks
            .iter()
            .zip(&prepared.blocks)
            .zip(&prepared.data)
            .enumerate()
        {
            let stored = StoredBlock {
                body,
                data: data.as_deref(),
            };
            self.insert_block(message_pk, session_pk, idx as i64, block, stored, ms(m.at))?;
        }
        Ok(())
    }

    fn insert_block(
        &self,
        message_pk: i64,
        session_pk: i64,
        idx: i64,
        block: &Block,
        StoredBlock { body, data }: StoredBlock<'_>,
        at: Option<i64>,
    ) -> Result<()> {
        let bit = if matches!(block, Block::ToolResult { .. }) {
            2
        } else {
            1
        };
        let text_id = block_text(block)
            .map(|text| {
                crate::text_rows::intern(self.tx, text, bit, Some(self.packed), self.first_build)
            })
            .transpose()?;
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
                     other_type, data)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
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
                ..
            } => {
                let (outcome, reason, exit) = outcome_columns(outcome);
                self.tx
                    .prepare_cached(
                        "UPDATE tool_call SET result_block_pk = ?1, outcome = ?2, outcome_reason = ?3,
                             exit_code = ?4
                         WHERE session_pk = ?5 AND call_id = ?6",
                    )?
                    .execute(params![block_pk, outcome, reason, exit, session_pk, call_id])?;
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
        let result: Option<(i64, Vec<u8>)> = self
            .tx
            .prepare_cached(
                "SELECT pk, body FROM block
                 WHERE session_pk = ?1 AND call_id = ?2 AND kind = 'tool_result'
                 ORDER BY pk LIMIT 1",
            )?
            .query_row(params![session_pk, call_id], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?;
        if let Some((result_pk, body)) = result
            && let Ok(Block::ToolResult { outcome, .. }) = unpack_block(&body)
        {
            let (outcome, reason, exit) = outcome_columns(&outcome);
            self.tx
                .prepare_cached(
                    "UPDATE tool_call SET result_block_pk = ?2, outcome = ?3, outcome_reason = ?4,
                         exit_code = ?5
                     WHERE pk = ?1",
                )?
                .execute(params![pk, result_pk, outcome, reason, exit])?;
        }
        Ok(())
    }

    /// `packed` is the event's id and stored payload ([`pack_event`]).
    fn write_event(
        &self,
        e: &Event,
        session_pk: Option<i64>,
        packed: &(String, Vec<u8>),
        order: Option<u64>,
        step_start: u64,
        occurrence: u64,
    ) -> Result<()> {
        let (content_id, data) = packed;
        // An event about a session keeps its content identity across inputs.
        // An event about its input belongs to one exact input record
        // occurrence, including repeated identical events from that record.
        let id = if e.session.is_none() {
            let (start, end) = e.range.ok_or_else(|| {
                StoreError::Invalid("an event about its input has no record range".into())
            })?;
            let mut hash = blake3::Hasher::new_derive_key("polylogue input event v1");
            hash.update(&(self.input_id.len() as u64).to_le_bytes());
            hash.update(self.input_id.as_bytes());
            hash.update(&self.chain.to_le_bytes());
            hash.update(&start.to_le_bytes());
            hash.update(&end.to_le_bytes());
            hash.update(&occurrence.to_le_bytes());
            hash.update(content_id.as_bytes());
            std::borrow::Cow::Owned(hash.finalize().to_hex().to_string())
        } else {
            std::borrow::Cow::Borrowed(content_id.as_str())
        };
        let after: Option<(i64, i64)> = match &e.after {
            Some(id) => match self.message_pk(&id.to_string())? {
                Some(pk) => Some((
                    pk,
                    self.tx
                        .prepare_cached("SELECT seq FROM message WHERE pk = ?1")?
                        .query_row([pk], |r| r.get(0))?,
                )),
                None => None,
            },
            None => None,
        };
        // An event with its own byte range sorts by it among the input's
        // messages (whose `seq` is their record offset); otherwise it takes
        // the position of the message it follows.
        // In a repeating whole conversations step an event without a range takes the step's place.
        let seq = order
            .map(|order| order as i64)
            .or(e.range.map(|(start, _)| start as i64))
            .or(self
                .repeats_whole_conversations
                .then_some(step_start as i64))
            .or(after.map(|a| a.1))
            .unwrap_or(0);
        let source_chain = e.range.map(|_| self.chain);
        // The first input to state an event keeps it; within that input the
        // later position (chain, then place in the chain) wins.
        let pk = match self.event_pk(&id)? {
            Some(pk) => {
                self.tx
                    .prepare_cached(
                        "UPDATE event SET seq = ?2, after_message_pk = ?3, source_chain = ?4,
                             source_start = ?5, source_end = ?6
                         WHERE pk = ?1 AND input_pk = ?7
                           AND (COALESCE(?4, 0), ?2) > (COALESCE(source_chain, 0), seq)",
                    )?
                    .execute(params![
                        pk,
                        seq,
                        after.map(|a| a.0),
                        source_chain,
                        e.range.map(|r| r.0 as i64),
                        e.range.map(|r| r.1 as i64),
                        self.input_pk,
                    ])?;
                pk
            }
            None => {
                self.tx
                    .prepare_cached(
                        "INSERT INTO event(id, session_pk, seq, after_message_pk, kind, at, input_pk,
                             source_chain, source_start, source_end, data)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                    )?
                    .execute(params![
                        &*id,
                        session_pk,
                        seq,
                        after.map(|a| a.0),
                        e.kind,
                        ms(e.at),
                        self.input_pk,
                        source_chain,
                        e.range.map(|r| r.0 as i64),
                        e.range.map(|r| r.1 as i64),
                        data,
                    ])?;
                let pk = self.tx.last_insert_rowid();
                if let Some(first_build) = self.first_build {
                    first_build.add_event(&id, pk);
                }
                pk
            }
        };
        if crate::encoding::reading_confirmed(&e.data) {
            // A copy of a reading that reports a measured completion
            // replaces one stored without it.
            let stored: Vec<u8> = self
                .tx
                .prepare_cached("SELECT data FROM event WHERE pk = ?1")?
                .query_row([pk], |r| r.get(0))?;
            let stored: serde_json::Value = crate::encoding::unpack_json(&stored)?;
            if !crate::encoding::reading_confirmed(&stored) {
                self.tx
                    .prepare_cached("UPDATE event SET data = ?2 WHERE pk = ?1")?
                    .execute(params![pk, data])?;
            }
        }
        Ok(())
    }

    /// One link per kind, parent and child; parts one file lacks are filled
    /// from another (the SQL form of [`Link::merge`]).
    /// Upserts a link. Several inputs state the same link; what it records
    /// must not depend on which arrived first, so where two
    /// statements differ the smaller value is kept, and the evidence keeps
    /// the input that stated it.
    fn write_link(&self, l: &Link) -> Result<()> {
        let (kind, how) = l.kind.columns();
        let child = l.child.to_string();
        let child_pk: Option<i64> = self
            .tx
            .prepare_cached("SELECT pk FROM session WHERE id = ?1")?
            .query_row([&child], |r| r.get(0))
            .optional()?;
        self.tx
            .prepare_cached(
                "INSERT INTO link(parent_id, child_id, child_pk, kind, how, branch_point,
                     spawning_call, evidence, input_pk)
                 VALUES (?1, ?2, ?3, ?4, ?9, ?5, ?6, ?7, ?8)
                 ON CONFLICT(parent_id, child_id, kind, how) DO UPDATE SET
                     child_pk = COALESCE(link.child_pk, excluded.child_pk),
                     branch_point = min(COALESCE(link.branch_point, excluded.branch_point),
                                        COALESCE(excluded.branch_point, link.branch_point)),
                     spawning_call = min(COALESCE(link.spawning_call, excluded.spawning_call),
                                         COALESCE(excluded.spawning_call, link.spawning_call)),
                     input_pk = CASE WHEN excluded.evidence < link.evidence
                                     THEN excluded.input_pk ELSE link.input_pk END,
                     evidence = min(link.evidence, excluded.evidence)",
            )?
            .execute(params![
                l.parent.to_string(),
                child,
                child_pk,
                kind,
                l.at.as_ref().map(ToString::to_string),
                l.call_id,
                l.evidence,
                self.input_pk,
                how,
            ])?;
        Ok(())
    }

    /// Parse position and record counts for the chain, committed with the rows
    /// they cover. A step from offset 0 replaces the chain's counts; a
    /// continuing step adds to them.
    fn write_parse_state(&self, b: &Batch, p: &ParseStep) -> Result<()> {
        let a = &p.counts;
        let used: u64 = a.used.values().sum();
        let skipped: u64 = a.skipped.values().sum();
        let unreadable: u64 = a.unreadable.values().sum();
        let status = if p.parsed_upto >= b.length {
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
                 used, skipped, unreadable, detail)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(input_pk, chain) DO UPDATE SET
                 status = excluded.status, parsed_upto = excluded.parsed_upto, detail = excluded.detail,
                 records = excluded.records, used = excluded.used,
                 skipped = excluded.skipped, unreadable = excluded.unreadable"
        } else {
            "INSERT INTO parse_state(input_pk, chain, status, parsed_upto, records,
                 used, skipped, unreadable, detail)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(input_pk, chain) DO UPDATE SET
                 status = excluded.status, parsed_upto = excluded.parsed_upto, detail = excluded.detail,
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
        ])?;
        if whole {
            self.tx
                .prepare_cached("DELETE FROM unused WHERE input_pk = ?1 AND chain = ?2")?
                .execute(params![self.input_pk, self.chain])?;
        }
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
