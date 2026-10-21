//! `index.db` rows for parsed records: sessions, messages and their
//! versions, blocks, tool calls, events, links and parse positions.

mod reparse;
mod snapshots;

use super::*;

/// Writes a parse step's rows; returns the number of message rows written.
pub(super) fn write_rows(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    chain: i64,
    counter: i64,
    bulk: bool,
) -> Result<u64> {
    tx.prepare_cached("INSERT INTO input_ref(id,format) VALUES (?1,?2) ON CONFLICT(id) DO UPDATE SET format=excluded.format WHERE format IS NOT excluded.format")?
        .execute(params![&b.input.id,&b.input.format])?;
    let input_pk: i64 = tx
        .prepare_cached("SELECT pk FROM input_ref WHERE id = ?1")?
        .query_row([&b.input.id], |r| r.get(0))?;
    let newer: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM parse_state WHERE input_pk=?1 AND chain=?2 AND parser_version>?3)",
        params![input_pk, chain, p.parser_version], |r| r.get(0),
    )?;
    if newer && p.snapshot.is_none() {
        return Ok(0);
    }
    if let Some(snapshot) = &p.snapshot {
        let newer_scope: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM snapshot WHERE input_pk=?1 AND chain=?2 AND source_start=?3 AND parser_version>?4)",params![input_pk,chain,snapshot.start as i64,p.parser_version],|r|r.get(0))?;
        if newer_scope {
            return Ok(0);
        }
    }
    let ctx = Ctx {
        tx,
        input_pk,
        input_id: &b.input.id,
        chain,
        snapshot: p.snapshot.as_ref(),
        rank: i64::from(b.input.route_rank),
        parser_version: i64::from(p.parser_version),
        counter,
        packed: &p.packed,
        bulk,
    };
    let affected = ctx.begin_snapshot(p)?;
    let mut sessions: HashMap<String, i64> = HashMap::new();
    for s in &p.output.sessions {
        let pk = ctx.session_row(&s.id, false)?;
        if p.snapshot.is_none() {
            ctx.upsert_session(s)?;
        }
        sessions.insert(s.id.to_string(), pk);
    }
    let mut written = 0;
    let mut activity: HashMap<i64, i64> = HashMap::new();
    for (i, row) in p.output.messages.iter().enumerate() {
        let session_pk = ctx.session_pk(&mut sessions, &row.message.session)?;
        let hash = p
            .hashes
            .get(i)
            .copied()
            .unwrap_or_else(|| message_hash(&row.message));
        let bodies = match p.bodies.get(i) {
            Some(b) if b.len() == row.message.blocks.len() => b.clone(),
            _ => row
                .message
                .blocks
                .iter()
                .map(pack_block)
                .collect::<Result<_>>()?,
        };
        written += u64::from(
            ctx.write_message(
                row,
                session_pk,
                hash,
                &bodies,
                if let Some(from) = p.order_from {
                    (from + i as u64) as i64
                } else if p.snapshot.is_some() {
                    i as i64
                } else {
                    row.offset as i64
                },
                p.output
                    .sessions
                    .iter()
                    .find(|s| s.id == row.message.session)
                    .map_or(ctx.session_kind(session_pk)? == "main", |s| {
                        s.kind == polylogue_model::SessionKind::Main
                    }),
            )?,
        );
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
        let session_pk = ctx.session_pk(&mut sessions, &e.session)?;
        let packed = match p.events.get(i) {
            Some(prepared) if p.events.len() == p.output.events.len() => prepared.clone(),
            _ => pack_event(e)?,
        };
        ctx.write_event(e, session_pk, &packed, p.order_from)?;
        if ctx.snapshot_current(session_pk)? {
            crate::live::apply(tx, session_pk, e, counter)?;
        }
    }
    for l in &p.output.links {
        ctx.write_link(l)?;
    }
    for l in &p.output.repo_lookups {
        crate::repo_lookup::upsert_lookup(tx, l, input_pk, chain, ctx.parser_version, counter)?;
    }
    if let Some(snapshot) = &p.snapshot {
        for id in &snapshot.covered {
            ctx.session_pk(&mut sessions, id)?;
        }
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
            "INSERT INTO parsed(input_pk, chain, session_pk, parser_version)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(input_pk,chain,session_pk) DO UPDATE SET parser_version=excluded.parser_version",
        )?
        .execute(params![input_pk, chain, pk, ctx.parser_version])?;
    }
    ctx.finish_snapshot(p, &affected)?;
    ctx.write_parse_state(b, p)?;
    ctx.finish_reparse(p, b.length)?;
    Ok(written)
}

pub(crate) fn promote_current(tx: &Transaction<'_>, messages: &[i64], counter: i64) -> Result<()> {
    let packed = HashMap::new();
    let ctx = Ctx {
        tx,
        input_pk: 0,
        input_id: "",
        chain: 0,
        snapshot: None,
        rank: 0,
        parser_version: 0,
        counter,
        packed: &packed,
        bulk: false,
    };
    for message in messages {
        ctx.promote_current(*message)?;
    }
    Ok(())
}

pub(super) struct Ctx<'a, 't> {
    tx: &'a Transaction<'t>,
    input_pk: i64,
    input_id: &'a str,
    chain: i64,
    snapshot: Option<&'a polylogue_model::Snapshot>,
    rank: i64,
    parser_version: i64,
    counter: i64,
    packed: &'a HashMap<[u8; 32], Vec<u8>>,
    /// A first build ([`Writer::start`]): search entries, trigram entries
    /// and secondary indexes are built once, in bulk, when it ends
    /// ([`Inner::build_deferred`]).
    bulk: bool,
}

/// The preferred version of a message, as compared with a new one.
pub(super) struct Version {
    input_pk: i64,
    session_pk: i64,
    chain: i64,
    snapshot: i64,
    order: i64,
    parser_version: i64,
    current: bool,
    start: i64,
    end: i64,
    /// Whether the version's session is a main session. A spawned
    /// session's file can repeat its parent's messages; the parent's own
    /// record is the original.
    main: bool,
    complete: bool,
    rank: i64,
    hash: Vec<u8>,
}

impl Version {
    /// Architecture 5.5 and 5.5a. Within one input: a newer revision chain
    /// wins (a rewrite); snapshot scopes prefer the latest observation.
    /// For ordinary records within a chain the earliest wire record wins;
    /// the same record sent again (a response still being written) replaces
    /// itself. Across inputs: the originating session's own record over a
    /// copy in a session it spawned, then a complete version, then the
    /// more faithful route, then the smaller content hash, so the choice
    /// does not depend on which input arrived first.
    fn beats(&self, cur: &Version) -> bool {
        if self.current != cur.current {
            return self.current;
        }
        if self.input_pk == cur.input_pk {
            if self.chain != cur.chain {
                return self.chain > cur.chain;
            }
            if self.snapshot != cur.snapshot {
                return self.snapshot > cur.snapshot;
            }
            if self.parser_version != cur.parser_version {
                return self.parser_version > cur.parser_version;
            }
            return self.snapshot != -1 || self.order <= cur.order;
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

impl Ctx<'_, '_> {
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
    /// what the same input stated before (one input's records arrive in
    /// file order), and creates the session row on first sight. The
    /// session's own columns are derived from every input's facts
    /// (derive/session.rs), so no arrival order decides them. Pending links
    /// naming the session as their child are resolved here.
    fn upsert_session(&self, s: &Session) -> Result<i64> {
        let pk = self.session_row(&s.id, false)?;
        let stored: Option<String> = self
            .tx
            .prepare_cached(
                "SELECT facts FROM session_record WHERE session_pk = ?1 AND input_pk = ?2 AND chain=?3 AND parser_version=?4",
            )?
            .query_row(params![pk, self.input_pk, self.chain, self.parser_version], |r| r.get(0))
            .optional()?;
        let merged = match stored.as_deref().map(serde_json::from_str::<Session>) {
            Some(Ok(mut cur)) => {
                let before = cur.clone();
                cur.merge(s);
                if cur == before {
                    return Ok(pk);
                }
                cur
            }
            _ => s.clone(),
        };
        self.tx
            .prepare_cached(
                "INSERT INTO session_record(session_pk, input_pk, facts, chain, parser_version) VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(session_pk, input_pk, chain, parser_version) DO UPDATE SET facts = excluded.facts",
            )?
            .execute(params![pk, self.input_pk, serde_json::to_string(&merged)?, self.chain, self.parser_version])?;
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

    /// Stores one version of a message and makes it the shown one when it
    /// beats the current one. Returns whether the message row was written.
    fn write_message(
        &self,
        row: &MessageRow,
        session_pk: i64,
        hash: [u8; 32],
        bodies: &[Vec<u8>],
        position: i64,
        main: bool,
    ) -> Result<bool> {
        let m = &row.message;
        let offset = row.offset as i64;
        let new = Version {
            input_pk: self.input_pk,
            session_pk,
            chain: self.chain,
            snapshot: self.snapshot.map_or(-1, |snapshot| snapshot.start as i64),
            order: position,
            parser_version: self.parser_version,
            current: self.snapshot_current(session_pk)?,
            start: offset,
            end: row.end.max(row.offset) as i64,
            main,
            complete: !(m.role == Role::Assistant && m.stop_reason.is_none()),
            rank: self.rank,
            hash: hash.to_vec(),
        };
        let cur = self
            .tx
            .prepare_cached(
                "SELECT m.pk, m.order_input_pk, m.seq, m.input_pk, m.source_chain, m.source_start,
                        m.in_progress, m.route_rank, s.version_hash, m.session_pk, ss.kind, m.order_input_id, s.source_snapshot, s.current, s.session_pk, s.main, COALESCE(s.source_order,m.seq), COALESCE(s.parser_version,m.parser_version)
                 FROM message m LEFT JOIN message_source s
                   ON s.message_pk = m.pk AND s.preferred = 1
                 JOIN session ss ON ss.pk = m.session_pk
                 WHERE m.id = ?1",
            )?
            .query_row([m.id.to_string()], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    (r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(11)?),
                    Version {
                        input_pk: r.get(3)?,
                        session_pk:r.get::<_,Option<i64>>(14)?.unwrap_or(r.get(9)?),
                        chain: r.get::<_, Option<i64>>(4)?.unwrap_or(0),
                        snapshot: r.get::<_, Option<i64>>(12)?.unwrap_or(-1),
                        order: r.get(16)?,
                        parser_version: r.get(17)?,
                        current: r.get::<_, Option<i64>>(13)?.unwrap_or(1) != 0,
                        start: r.get::<_, Option<i64>>(5)?.unwrap_or(0),
                        end: 0,
                        main: r.get::<_,Option<bool>>(15)?.unwrap_or(r.get::<_, String>(10)? == "main"),
                        complete: r.get::<_, i64>(6)? == 0,
                        rank: r.get(7)?,
                        hash: r.get::<_, Option<Vec<u8>>>(8)?.unwrap_or_default(),
                    },
                    (r.get::<_, i64>(9)?, r.get::<_, String>(10)?),
                ))
            })
            .optional()?;

        let Some((pk, (order_input, order_seq, order_id), preferred, (cur_session, cur_kind))) =
            cur
        else {
            crate::versions::retain(self.tx, &new.hash, m, bodies, self.packed, self.bulk)?;
            let pk = self.insert_message(m, session_pk, &new, (self.input_pk, position))?;
            self.upsert_source(pk, &new, true)?;
            self.insert_blocks(pk, session_pk, m, bodies)?;
            self.tx.execute(
                "UPDATE message SET present=?2 WHERE pk=?1",
                params![pk, i64::from(new.current)],
            )?;
            return Ok(true);
        };

        // Earliest duplicate wins within one parser. A newer parser may move
        // the same occurrence; replay by an older parser cannot replace it.
        let prior: Option<(i64,i64)> = self
            .tx
            .prepare_cached(
                "SELECT source_order,parser_version FROM message_source
                 WHERE message_pk = ?1 AND input_pk = ?2 AND chain = ?3 AND source_snapshot = ?4 AND session_pk=?5",
            )?
            .query_row(params![pk, self.input_pk, self.chain, new.snapshot, new.session_pk], |r| {
                Ok((r.get(0)?,r.get(1)?))
            })
            .optional()?;
        if prior.is_some_and(|(order, version)| {
            version > self.parser_version || (version == self.parser_version && position > order)
        }) {
            return Ok(false);
        }
        crate::versions::retain(self.tx, &new.hash, m, bodies, self.packed, self.bulk)?;
        if !new.current {
            self.upsert_source(pk, &new, false)?;
            return Ok(false);
        }
        let order = (order_input, order_seq);
        let reinterpreted = prior.is_some_and(|(_, version)| version != self.parser_version);
        let new_order = if ((self.snapshot.is_some()
            || (reinterpreted
                && self.chain == preferred.chain
                && new.snapshot == preferred.snapshot))
            && order_id == self.input_id)
            || (self.input_id, position) < (order_id.as_str(), order_seq)
        {
            (self.input_pk, position)
        } else {
            order
        };
        if !new.beats(&preferred) {
            self.upsert_source(pk, &new, false)?;
            if new_order != order {
                self.tx
                    .prepare_cached(
                        "UPDATE message SET order_input_pk = ?2, seq = ?3,
                             order_input_id = (SELECT id FROM input_ref WHERE pk = ?2) WHERE pk = ?1",
                    )?
                    .execute(params![pk, new_order.0, new_order.1])?;
            }
            return Ok(false);
        }
        // A message is stored once, in the session where it originated
        // (architecture 5.4). A spawned session's file can repeat its
        // parent's messages (a compaction call, a side question replaying
        // the history); such a copy may become the shown version, but it
        // never moves the message out of a main session.
        let session_pk = if cur_session != session_pk && cur_kind == "main" && !new.main {
            cur_session
        } else {
            session_pk
        };
        self.tx
            .prepare_cached("UPDATE message_source SET preferred = 0 WHERE message_pk = ?1")?
            .execute([pk])?;
        self.upsert_source(pk, &new, true)?;
        let unchanged = new.hash == preferred.hash && cur_session == session_pk;
        if !unchanged {
            self.clear_blocks(pk)?;
        }
        self.update_message(pk, m, session_pk, &new, new_order)?;
        self.tx
            .execute("UPDATE message SET present = 1 WHERE pk = ?1", [pk])?;
        if !unchanged {
            self.insert_blocks(pk, session_pk, m, bodies)?;
        }
        Ok(true)
    }

    fn upsert_source(&self, pk: i64, v: &Version, preferred: bool) -> Result<()> {
        let old:Option<Vec<u8>>=self.tx.query_row("SELECT version_hash FROM message_source WHERE message_pk=?1 AND input_pk=?2 AND chain=?3 AND source_snapshot=?4 AND session_pk=?5",params![pk,v.input_pk,v.chain,v.snapshot,v.session_pk],|r|r.get(0)).optional()?.flatten();
        self.tx.execute(
            "INSERT INTO message_source(message_pk, input_pk, chain, source_snapshot, source_start, source_end, session_pk,
                 preferred, complete, version_hash, current, source_order, route_rank, main, parser_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?14, ?7, ?8, ?9, ?13, ?10, ?11, ?12, ?15)
             ON CONFLICT(message_pk, input_pk, chain, source_snapshot, session_pk) DO UPDATE SET
                 source_start=excluded.source_start, source_end=excluded.source_end, session_pk=excluded.session_pk, preferred=excluded.preferred,
                 complete=excluded.complete, version_hash=excluded.version_hash, current=excluded.current,
                 source_order=excluded.source_order, route_rank=excluded.route_rank, main=excluded.main,
                 parser_version=excluded.parser_version",
            params![pk, v.input_pk, v.chain, v.snapshot, v.start, v.end, i64::from(preferred), i64::from(v.complete), v.hash, v.order, v.rank, i64::from(v.main), i64::from(v.current),v.session_pk,self.parser_version],
        )?;
        if let Some(old) = old.filter(|old| old != &v.hash) {
            self.tx.execute("DELETE FROM message_version WHERE hash=?1 AND NOT EXISTS (SELECT 1 FROM message_source WHERE version_hash=?1)",[old])?;
        }
        Ok(())
    }

    fn session_kind(&self, session_pk: i64) -> Result<String> {
        Ok(self
            .tx
            .prepare_cached("SELECT kind FROM session WHERE pk = ?1")?
            .query_row([session_pk], |r| r.get(0))?)
    }

    fn insert_message(
        &self,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: (i64, i64),
    ) -> Result<i64> {
        self.set_message(None, m, session_pk, v, order)
    }

    fn update_message(
        &self,
        pk: i64,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: (i64, i64),
    ) -> Result<()> {
        self.set_message(Some(pk), m, session_pk, v, order)
            .map(drop)
    }

    /// Writes every column of the preferred version in one statement: an
    /// insert for a new message, an update when another version won.
    fn set_message(
        &self,
        pk: Option<i64>,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: (i64, i64),
    ) -> Result<i64> {
        let written_by = match (m.role, m.prompt_kind) {
            (Role::Assistant, _) | (Role::User, Some(PromptKind::Model)) => "model",
            (Role::System, _) | (Role::User, Some(PromptKind::Harness)) => "harness",
            (Role::User, Some(_)) => "human",
            (Role::User, None) => "unknown",
        };
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
            "UPDATE message SET session_pk = ?2, order_input_pk = ?3, seq = ?4, route_rank = ?5,
                 role = ?6, written_by = ?7, prompt_kind = ?8, model = ?9, at = ?10,
                 parent_id = ?11, input_pk = ?12, source_chain = ?13, source_start = ?14,
                 source_end = ?27, in_progress = ?15, input_tokens = ?16, output_tokens = ?17,
                 cache_read_tokens = ?18, cache_write_tokens = ?19, reasoning_tokens = ?20,
                 service_tier = ?21, stop_reason = ?22, duration_ms = ?23, attrs_json = ?24,
                 words = ?25, parser_version = ?26, cost_nusd = NULL, provider_cost_nusd = ?28,
                 cache_write_5m_tokens = ?29, cache_write_1h_tokens = ?30, speed = ?31,
                 order_input_id = (SELECT id FROM input_ref WHERE pk = ?3)
             WHERE pk = ?1"
        } else {
            "INSERT INTO message(id, session_pk, order_input_pk, seq, route_rank, role, written_by,
                 prompt_kind, model, at, parent_id, input_pk, source_chain, source_start,
                 source_end, in_progress, input_tokens, output_tokens, cache_read_tokens,
                 cache_write_tokens, reasoning_tokens, service_tier, stop_reason, duration_ms,
                 attrs_json, words, parser_version, provider_cost_nusd,
                 cache_write_5m_tokens, cache_write_1h_tokens, speed, order_input_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?27, ?15, ?16,
                 ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?28, ?29, ?30, ?31,
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
            order.1,
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
            v.parser_version,
            v.end,
            reported_nusd(m.reported_cost_usd.as_ref()),
            // The five-minute part is known only when the one-hour part is.
            u.cache_write_1h
                .and(u.cache_write)
                .map(|w| w.saturating_sub(u.cache_write_1h.unwrap_or(0)) as i64),
            n(u.cache_write_1h),
            u.speed,
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
            if !self.bulk {
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
            crate::text_rows::unindex_unused(self.tx, text, self.bulk)?;
        }
        Ok(())
    }

    /// Inserts a message's blocks; `bodies` are their stored forms
    /// ([`pack_block`]), in the same order.
    fn insert_blocks(
        &self,
        message_pk: i64,
        session_pk: i64,
        m: &Message,
        bodies: &[Vec<u8>],
    ) -> Result<()> {
        for (idx, (block, body)) in m.blocks.iter().zip(bodies).enumerate() {
            self.insert_block(message_pk, session_pk, idx as i64, block, body, ms(m.at))?;
        }
        Ok(())
    }

    fn insert_block(
        &self,
        message_pk: i64,
        session_pk: i64,
        idx: i64,
        block: &Block,
        body: &[u8],
        at: Option<i64>,
    ) -> Result<()> {
        let bit = if matches!(block, Block::ToolResult { .. }) {
            2
        } else {
            1
        };
        let text_id = match block_text(block) {
            Some(t) => Some(crate::text_rows::intern(
                self.tx,
                t,
                bit,
                Some(self.packed),
                self.bulk,
            )?),
            None => None,
        };
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
                     other_type)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
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
            ])?;
        let block_pk = self.tx.last_insert_rowid();

        match block {
            Block::ToolCall {
                call_id,
                tool,
                action,
                command,
                path,
                ..
            } => self.insert_call(
                block_pk, session_pk, call_id, tool, *action, command, path, at,
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
    ) -> Result<()> {
        let tool = tool.as_deref().unwrap_or("unknown");
        self.tx
            .prepare_cached(
                "INSERT INTO tool_call(session_pk, call_block_pk, call_id, at, tool, action,
                     command, path, outcome, outcome_reason)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'unknown', 'no_result')",
            )?
            .execute(params![
                session_pk,
                block_pk,
                call_id,
                at,
                tool,
                action.as_str(),
                command,
                path
            ])?;
        let pk = self.tx.last_insert_rowid();
        if !self.bulk {
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
        session_pk: i64,
        packed: &(String, Vec<u8>),
        order: Option<u64>,
    ) -> Result<()> {
        let (id, data) = packed;
        let after: Option<(i64, i64)> = match &e.after {
            Some(id) => self
                .tx
                .prepare_cached("SELECT pk, seq FROM message WHERE id = ?1")?
                .query_row([id.to_string()], |r| Ok((r.get(0)?, r.get(1)?)))
                .optional()?,
            None => None,
        };
        // An event with its own byte range sorts by it among the input's
        // messages (whose `seq` is their record offset); otherwise it takes
        // the position of the message it follows.
        let seq = e
            .range
            .map(|(start, _)| start as i64)
            .or(after.map(|a| a.1))
            .unwrap_or(0);
        self.tx
            .prepare_cached(
                "INSERT OR IGNORE INTO event(id,session_pk,after_message_pk,kind,at,data)
             VALUES (?1,?2,?3,?4,?5,?6)",
            )?
            .execute(params![
                id,
                session_pk,
                after.map(|a| a.0),
                e.kind,
                ms(e.at),
                data
            ])?;
        self.tx.prepare_cached(
            "INSERT INTO event_source(event_id,input_pk,chain,source_snapshot,current,parser_version,source_order,source_start,source_end)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)
             ON CONFLICT(event_id,input_pk,chain,source_snapshot) DO UPDATE SET
               current=excluded.current,
               source_start=CASE WHEN event_source.parser_version<>excluded.parser_version OR excluded.source_order>=event_source.source_order THEN excluded.source_start ELSE event_source.source_start END,
               source_end=CASE WHEN event_source.parser_version<>excluded.parser_version OR excluded.source_order>=event_source.source_order THEN excluded.source_end ELSE event_source.source_end END,
               source_order=CASE WHEN event_source.parser_version<>excluded.parser_version OR excluded.source_order>=event_source.source_order THEN excluded.source_order ELSE event_source.source_order END,
               parser_version=excluded.parser_version"
        )?.execute(params![id,self.input_pk,self.chain,self.snapshot.map_or(-1,|snapshot|snapshot.start as i64),
            i64::from(self.snapshot_current(session_pk)?),self.parser_version,order.map_or(seq,|order|order as i64),
            e.range.map(|range|range.0 as i64),e.range.map(|range|range.1 as i64)])?;
        Ok(())
    }

    /// One link per kind, parent and child; parts one file lacks are filled
    /// from another (the SQL form of [`Link::merge`]).
    /// Upserts a link. Several inputs state the same link; what it records
    /// must not depend on which arrived first (law 41), so where two
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
                     spawning_call, evidence, input_pk, parser_version)
                 VALUES (?1, ?2, ?3, ?4, ?10, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(parent_id, child_id, kind, how) DO UPDATE SET
                     child_pk = COALESCE(link.child_pk, excluded.child_pk),
                     branch_point = min(COALESCE(link.branch_point, excluded.branch_point),
                                        COALESCE(excluded.branch_point, link.branch_point)),
                     spawning_call = min(COALESCE(link.spawning_call, excluded.spawning_call),
                                         COALESCE(excluded.spawning_call, link.spawning_call)),
                     input_pk = CASE WHEN excluded.evidence < link.evidence
                                     THEN excluded.input_pk ELSE link.input_pk END,
                     parser_version = CASE WHEN excluded.evidence < link.evidence
                                           THEN excluded.parser_version ELSE link.parser_version END,
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
                self.parser_version,
                how,
            ])?;
        let pk: i64 = self.tx.query_row(
            "SELECT pk FROM link WHERE parent_id=?1 AND child_id=?2 AND kind=?3 AND how=?4",
            params![l.parent.to_string(), l.child.to_string(), kind, how],
            |row| row.get(0),
        )?;
        let owner = self.snapshot.and_then(|scope| {
            scope
                .covered
                .iter()
                .find(|id| *id == &l.child || *id == &l.parent)
        });
        let current = match owner {
            Some(id) => {
                let session: i64 = self.tx.query_row(
                    "SELECT pk FROM session WHERE id=?1",
                    [id.to_string()],
                    |row| row.get(0),
                )?;
                self.snapshot_current(session)?
            }
            None => self.snapshot.is_none(),
        };
        self.tx.execute("INSERT INTO link_source(link_pk,input_pk,chain,source_snapshot,current,facts,parser_version) VALUES (?1,?2,?3,?4,?6,?5,?7) ON CONFLICT(link_pk,input_pk,chain,source_snapshot) DO UPDATE SET current=excluded.current,facts=excluded.facts,parser_version=excluded.parser_version", params![pk,self.input_pk,self.chain,self.snapshot.map_or(-1,|snapshot|snapshot.start as i64),crate::encoding::pack_json(l)?,i64::from(current),self.parser_version])?;
        self.project_link(pk)?;
        Ok(())
    }

    /// Parse position and accounting for the chain, committed with the rows
    /// they cover. A step from offset 0 replaces the chain's counts; a
    /// continuing step adds to them.
    fn write_parse_state(&self, b: &Batch, p: &ParseStep) -> Result<()> {
        let a = &p.accounting;
        let used: u64 = a.materialized.values().sum();
        let unknown: u64 = a.unknown.values().sum();
        let refused: u64 = a.refused.values().sum();
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
            "INSERT INTO parse_state(input_pk, chain, parser_version, status, parsed_upto, framed,
                 used, unrecognized_n, undecodable, detail, source_eof, source_length, retired_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?3)
             ON CONFLICT(input_pk, chain) DO UPDATE SET parser_version = excluded.parser_version,
                 status = excluded.status, parsed_upto = excluded.parsed_upto, detail = excluded.detail,
                 source_eof = excluded.source_eof, source_length = excluded.source_length,
                 framed = excluded.framed, used = excluded.used,
                 unrecognized_n = excluded.unrecognized_n, undecodable = excluded.undecodable"
        } else {
            "INSERT INTO parse_state(input_pk, chain, parser_version, status, parsed_upto, framed,
                 used, unrecognized_n, undecodable, detail, source_eof, source_length, retired_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?3)
             ON CONFLICT(input_pk, chain) DO UPDATE SET parser_version = excluded.parser_version,
                 status = excluded.status, parsed_upto = excluded.parsed_upto, detail = excluded.detail,
                 source_eof = excluded.source_eof, source_length = excluded.source_length,
                 framed = framed + excluded.framed, used = used + excluded.used,
                 unrecognized_n = unrecognized_n + excluded.unrecognized_n,
                 undecodable = undecodable + excluded.undecodable"
        };
        self.tx.prepare_cached(sql)?.execute(params![
            self.input_pk,
            self.chain,
            self.parser_version,
            status,
            p.parsed_upto as i64,
            a.framed as i64,
            used as i64,
            unknown as i64,
            refused as i64,
            detail,
            p.at_end,
            b.length as i64,
        ])?;
        if whole {
            self.tx
                .prepare_cached("DELETE FROM unrecognized WHERE input_pk = ?1 AND chain = ?2")?
                .execute(params![self.input_pk, self.chain])?;
        }
        let groups = [
            ("record:", &a.unknown, 1),
            ("undecodable:", &a.refused, 0),
            ("field:", &a.unread_fields, 0),
        ];
        for (prefix, map, counted_only) in groups {
            for (what, count) in map {
                self.tx
                    .prepare_cached(
                        "INSERT INTO unrecognized(input_pk, chain, format, what, count, counted_only)
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
    match o {
        ToolOutcome::Ok => ("ok", None, None),
        ToolOutcome::Error { reason } => match reason {
            Some(ErrorReason::ExitCode(code)) => ("error", Some("exit_code".into()), Some(*code)),
            Some(ErrorReason::Denied(label)) => ("error", Some(format!("denied:{label}")), None),
            Some(ErrorReason::Interrupted) => ("error", Some("interrupted".into()), None),
            Some(ErrorReason::TimedOut) => ("error", Some("timed_out".into()), None),
            None => ("error", None, None),
        },
        ToolOutcome::NoResult => ("unknown", Some("no_result".into()), None),
        ToolOutcome::Unknown { reason } => ("unknown", Some(name(reason)), None),
    }
}

/// A provider's reported amount is an observed fact, separate from catalog pricing.
fn reported_nusd(usd: Option<&serde_json::Number>) -> Option<i64> {
    let usd = usd?.as_f64()?;
    usd.is_finite().then(|| (usd * 1e9).round() as i64)
}

pub(super) fn require_parser(tx: &Transaction<'_>, format: &str, version: u32) -> Result<()> {
    tx.prepare_cached("INSERT INTO required_parser(format,version) VALUES (?1,?2) ON CONFLICT(format) DO UPDATE SET version=excluded.version WHERE version<excluded.version")?
        .execute(params![format,version])?;
    Ok(())
}
