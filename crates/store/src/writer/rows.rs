//! `index.db` rows for parsed records: sessions, messages and their
//! sources, blocks, tool calls, events, links and parse positions.

use super::*;
use crate::change_summary::ChangeSummary;
use crate::key_maps::KeyMaps;
use polylogue_model::{Completion, Route};
use std::{cell::Cell, collections::BTreeMap};

mod checkpoint;
mod events_links;
mod retract;
pub(super) use retract::prune_sessions;

/// Writes a parse step's rows; returns the number of message rows written.
pub(super) fn write_rows(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    mut context: ReadingWrite<'_>,
) -> Result<u64> {
    context.retraction = false;
    context.replacement = false;
    apply_rows(tx, b, p, context)
}

/// A complete replacement uses this entry point for its first chunk. Later chunks use the continuation path.
pub(super) fn write_replacement_rows(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    mut context: ReadingWrite<'_>,
) -> Result<u64> {
    context.retraction = false;
    context.replacement = true;
    apply_rows(tx, b, p, context)
}

/// Retract a source's contribution and show the surviving readings.
pub(super) fn retract_source(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    mut context: ReadingWrite<'_>,
) -> Result<u64> {
    context.retraction = true;
    context.replacement = false;
    apply_rows(tx, b, p, context)
}

pub(super) struct ReadingWrite<'a> {
    pub(super) texts: &'a crate::text_rows::TextIds,
    pub(super) resolution: &'a resolve::Resolution,
    pub(super) counter: i64,
    pub(super) first_build: Option<&'a KeyMaps>,
    pub(super) retraction: bool,
    pub(super) replacement: bool,
    pub(super) source_commit: Option<&'a crate::wire_shapes::ReadingCommit>,
    pub(super) changes: &'a ChangeSummary,
}

fn apply_rows(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    context: ReadingWrite<'_>,
) -> Result<u64> {
    let ReadingWrite {
        resolution,
        texts,
        counter,
        first_build,
        retraction,
        replacement,
        source_commit,
        changes,
    } = context;
    p.counts.check()?;
    p.prepared.validate(&p.output)?;
    let chain = resolution.chain;
    let keys = &*resolution.keys;
    let choices = &resolution.choices;
    let input_pk = keys.get(resolve::Kind::Input, b.input.id.as_bytes())?.pk;
    validate_range(b, p)?;
    if !crate::wire_shapes::admit_export(tx, &b.input.id, chain, b.length, p)? {
        return Ok(0);
    }
    let previous_checkpoint = checkpoint::read(tx, input_pk, chain)?;
    if p.delivery.is_none() && !retraction && !accept_step(previous_checkpoint.as_ref(), p)? {
        return Ok(0);
    }
    tx.prepare_cached("INSERT INTO input_ref(pk,id,format) VALUES (?1,?2,?3) ON CONFLICT(pk) DO UPDATE SET format=excluded.format WHERE format IS NOT excluded.format")?
        .execute(params![input_pk,b.input.id,b.input.format])?;
    if let Some(update) = &p.comparison {
        update.write(tx, input_pk)?;
    }
    let mut rows = RowWriter {
        tx,
        checkpoint: checkpoint::Update::new(
            b,
            p,
            previous_checkpoint.as_ref(),
            counter,
            retraction,
            source_commit,
        )?,
        source_committed: if retraction {
            None
        } else {
            source_commit
                .or(p.source_commit.as_ref())
                .map(|r| r.sequence)
                .filter(|n| *n > 0)
        },
        input_pk,
        input_id: &b.input.id,
        chain,
        route: b.input.route,
        counter,
        first_build,
        keys,
        choices,
        changes,
        texts,
        versions: HashMap::new(),
        messages: choices
            .messages
            .iter()
            .map(|(&pk, choice)| {
                let value = choice
                    .as_ref()
                    .map(|choice| choices.selected_message(choice))
                    .transpose()?
                    .flatten();
                Ok((
                    pk,
                    MessageWrite {
                        choice: choice.as_ref(),
                        value,
                        created: Cell::new(None),
                    },
                ))
            })
            .collect::<Result<_>>()?,
    };
    if p.reset {
        rows.retract()?;
    }
    let source_sequence = if retraction {
        None
    } else {
        source_commit
            .map(|source_commit| {
                u64::try_from(source_commit.sequence)
                    .map_err(|_| crate::StoreError::stored(0, "negative source commit sequence"))
            })
            .transpose()?
    };
    let membership = crate::text_rows::NewMembership {
        bits: &rows.current_texts(),
        bulk: first_build.is_some(),
    };
    for version in choices.message_versions.values() {
        let registered = crate::versions::store_message(
            tx,
            &version.value.message,
            &version.value.prepared,
            crate::versions::TextRows {
                packed: &p.prepared.texts,
                texts,
                membership: Some(&membership),
            },
            version.replace,
            source_sequence,
        )?;
        rows.versions
            .insert(version.value.prepared.hash, registered);
    }
    let mut sessions: HashMap<String, i64> = HashMap::new();
    for s in &p.output.sessions {
        let (pk, _) = rows.upsert_session(s, rows.source_committed)?;
        sessions.insert(s.id.to_string(), pk);
    }
    let source = crate::documents::Source {
        input_pk,
        chain,
        route: b.input.route,
        counter,
        packed: &p.prepared.texts,
        first_build,
        texts,
        keys,
    };
    for (i, row) in p.output.documents.iter().enumerate() {
        if !choices.keep_documents[i] {
            continue;
        }
        let prepared = &p.prepared.documents[i];
        crate::documents::write(tx, &source, row, prepared, rows.source_committed)?;
    }
    p.prepared
        .fragments
        .apply(tx, input_pk, chain, p.parsed_upto, |_| {
            Ok(rows.source_committed)
        })?;
    let mut written = 0;
    // Versions a source row of this step now refers to. Removing or changing
    // that reference later records the version through its trigger.
    let mut chosen = std::collections::HashSet::new();
    let mut activity: HashMap<i64, i64> = HashMap::new();
    // Messages one record holds share its offset; their order in the
    // record breaks the tie.
    let mut in_record: HashMap<u64, i64> = HashMap::new();
    for (i, row) in p.output.messages.iter().enumerate() {
        if !choices.keep_messages[i] {
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
        let order_time = choices.message_order_times[i];
        let version_pk = rows.version_pk(&prepared.hash)?;
        let new = rows.write_message(
            row,
            session_pk,
            prepared,
            (position, order_time),
            main,
            version_pk,
            &mut sessions,
        )?;
        chosen.insert(version_pk);
        super::hook_fields::write(
            tx,
            super::hook_fields::Source {
                input: &b.input.id,
                input_pk,
                chain,
            },
            keys,
            p,
            i,
            version_pk,
            new,
        )?;
        if let Some(at) = ms(row.message.at) {
            let last = activity.entry(session_pk).or_insert(at);
            *last = (*last).max(at);
        }
    }
    // A stored version no source row chose is deleted before the commit.
    for version in rows.versions.values() {
        if !chosen.contains(&version.pk()) {
            crate::versions::unreferenced_message(tx, version.pk())?;
        }
    }
    // A message read again with another time changes the order times of the
    // chain's readings after it (membership `recompute_later`).
    for (message_pk, time) in &choices.recomputed_order_times {
        tx.prepare_cached(
            "UPDATE message_source SET order_time=?4 WHERE message_pk=?1 AND input_pk=?2 AND chain=?3",
        )?
        .execute(params![message_pk, input_pk, chain, time])?;
    }
    if b.input.route.lists_every_conversation(&b.input.format) {
        let mut latest = HashMap::<&str, i64>::new();
        for row in &p.output.messages {
            if let Some(at) = ms(row.message.at) {
                let origin = latest
                    .entry(row.message.session.origin().as_str())
                    .or_insert(at);
                *origin = (*origin).max(at);
            }
        }
        for (origin, at) in latest {
            tx.prepare_cached(
                "INSERT INTO export_reading(input_pk,chain,origin,latest_at) VALUES (?1,?2,?3,?4)
                 ON CONFLICT(input_pk,chain,origin) DO UPDATE SET latest_at=max(latest_at,excluded.latest_at)",
            )?
            .execute(params![input_pk, chain, origin, at])?;
        }
    }
    for (session_pk, at) in activity {
        tx.prepare_cached(
            "UPDATE session SET last_activity_at = MAX(COALESCE(last_activity_at, ?2), ?2)
             WHERE pk = ?1",
        )?
        .execute(params![session_pk, at])?;
    }
    for (i, e) in p.output.events.iter().enumerate() {
        if let Some(id) = &e.session {
            rows.session_pk(&mut sessions, id)?;
        }
        rows.write_event(i)?;
    }
    for prefix in &p.output.prefix_copies {
        let parent = prefix.parent.to_string();
        let child = prefix.child.to_string();
        let ids = &rows.keys.identities;
        let (parent, child) = (ids.shown_session(&parent), ids.shown_session(&child));
        let committed = rows.source_committed;
        rows.write_prefix_copy(parent, child, prefix.explicit_members, committed)?;
    }
    for member in &p.output.prefix_members {
        let parent = member.parent.to_string();
        let child = member.child.to_string();
        let message = member.message.to_string();
        let ids = &rows.keys.identities;
        let (parent, child) = (ids.shown_session(&parent), ids.shown_session(&child));
        let message = ids.shown_message(&message);
        let source = member
            .source
            .clone()
            .unwrap_or_else(|| polylogue_model::PrefixSource {
                input: b.input.id.clone(),
                chain,
                reading_version: p.reading_version.clone(),
            });
        let committed = rows.source_committed;
        tx.prepare_cached("INSERT INTO prefix_member(input_pk,chain,parent_id,child_id,message_id,source_input,source_chain,source_version,sequence,position,committed) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11) ON CONFLICT DO UPDATE SET committed=coalesce(max(prefix_member.committed,excluded.committed),prefix_member.committed,excluded.committed)")?
            .execute(params![input_pk,chain,parent,child,message,source.input,source.chain,source.reading_version,member.position[0].to_string(),member.position[1].to_string(),committed])?;
        rows.write_prefix_copy(parent, child, true, committed)?;
    }
    for (i, l) in p.output.links.iter().enumerate() {
        let committed = rows.source_committed;
        rows.write_link(l, i, committed)?;
    }
    for l in &p.output.repo_lookups {
        crate::repo_lookup::upsert_lookup(tx, l, input_pk, chain)?;
    }
    {
        for (from, to, _) in &choices.absorbed_sessions {
            rows.absorb_session(*from, *to)?;
        }
        for (source, target) in &choices.moved_messages {
            // The session's own reading of the message may come in a later
            // step; the copy's readings make its row now.
            if !tx
                .prepare_cached("SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1)")?
                .query_row([target], |r| r.get::<_, bool>(0))?
            {
                written += u64::from(rows.refresh_message(*target)?);
            }
            tx.prepare_cached("UPDATE message_source SET message_pk=?2 WHERE message_pk=?1")?
                .execute(params![source, target])?;
        }
        for (source, target, id) in &choices.moved_events {
            // The session's own reading of the event may come in a later
            // step; the copy's readings make its row now.
            if !tx
                .prepare_cached("SELECT EXISTS(SELECT 1 FROM event WHERE pk=?1)")?
                .query_row([target], |r| r.get::<_, bool>(0))?
            {
                changes.event(tx, *target)?;
                tx.prepare_cached(
                    "INSERT INTO event(pk,id,session_pk,seq,after_message_id,seq_from_message,after_message_pk,kind,at,input_pk,source_chain,source_start,source_end,fields,data)
                     SELECT ?2,?3,session_pk,seq,after_message_id,seq_from_message,after_message_pk,kind,at,input_pk,source_chain,source_start,source_end,fields,data
                     FROM event WHERE pk=?1",
                )?
                .execute(params![source, target, id])?;
                if let Some(first) = first_build {
                    first.add_event(id, *target);
                }
            }
            tx.prepare_cached("UPDATE event_source SET event_pk=?2 WHERE event_pk=?1")?
                .execute(params![source, target])?;
        }
        for resolve::membership::ReleasedSession {
            from,
            to,
            copy,
            chains,
        } in &choices.released_sessions
        {
            rows.release_session(*from, *to, copy, chains)?;
            for (source, target) in &choices.released_messages {
                if !tx
                    .prepare_cached("SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1)")?
                    .query_row([target], |r| r.get::<_, bool>(0))?
                {
                    written += u64::from(rows.refresh_message(*target)?);
                }
                for (input, chain) in chains {
                    tx.prepare_cached(
                        "UPDATE message_source SET message_pk=?2, session_pk=?5
                         WHERE message_pk=?1 AND input_pk=?3 AND chain=?4",
                    )?
                    .execute(params![source, target, input, chain, to])?;
                }
            }
            for (source, target, id) in &choices.released_events {
                if !tx
                    .prepare_cached("SELECT EXISTS(SELECT 1 FROM event WHERE pk=?1)")?
                    .query_row([target], |r| r.get::<_, bool>(0))?
                {
                    changes.event(tx, *target)?;
                    tx.prepare_cached(
                        "INSERT INTO event(pk,id,session_pk,seq,after_message_id,seq_from_message,after_message_pk,kind,at,input_pk,source_chain,source_start,source_end,fields,data)
                         SELECT ?2,?3,?4,seq,after_message_id,seq_from_message,after_message_pk,kind,at,input_pk,source_chain,source_start,source_end,fields,data
                         FROM event WHERE pk=?1",
                    )?
                    .execute(params![source, target, id, to])?;
                    if let Some(first) = first_build {
                        first.add_event(id, *target);
                    }
                }
                for (input, chain) in chains {
                    tx.prepare_cached(
                        "UPDATE event_source SET event_pk=?2, session_pk=?5
                         WHERE event_pk=?1 AND input_pk=?3 AND chain=?4",
                    )?
                    .execute(params![source, target, input, chain, to])?;
                }
            }
        }
        for (message_pk, source_input, source_chain) in &choices.dropped_message_sources {
            tx.prepare_cached(
                "DELETE FROM message_source WHERE message_pk=?1 AND input_pk=?2 AND chain=?3",
            )?
            .execute(params![message_pk, source_input, source_chain])?;
        }
        for pk in choices.messages.keys() {
            written += u64::from(rows.refresh_message(*pk)?);
        }
        rows.refresh_event_placement()?;
        for (pk, choice) in &choices.documents {
            crate::documents::apply_choice(
                tx,
                *pk,
                choice.as_ref(),
                counter,
                first_build,
                texts,
                keys,
            )?;
        }
        for pk in choices.events.keys() {
            rows.refresh_event(*pk)?;
        }
        for pk in choices.links.keys() {
            rows.refresh_link(*pk)?;
        }
        for (from, _, copy) in &choices.absorbed_sessions {
            changes.removed_session(*from);
            crate::session_row::remove(tx, *from, copy, counter)?;
        }
    }
    // Every session these records touched is derived again.
    for pk in sessions.values() {
        changes.session(tx, *pk)?;
    }
    // Hooks name the selected tool-call rows, which message block_selection just wrote.
    for (i, e) in p.output.events.iter().enumerate() {
        let session_pk = e
            .session
            .as_ref()
            .map(|id| rows.session_pk(&mut sessions, id))
            .transpose()?;
        if let Some(session_pk) =
            session_pk.filter(|_| polylogue_model::live::is_live_event(&e.kind))
        {
            // Prepared events release their JSON payload before writing. Live
            // events must read the same payload as the stored event.
            let event = &choices.live_events[i];
            crate::live::apply(
                tx,
                session_pk,
                &event.kind,
                event.at.map(|t| t.unix_ms()),
                &event.data,
                counter,
            )?;
        }
    }
    for u in &p.output.background {
        if let Some(call_id) = u.call_id.as_deref()
            && let Some(session) = tx
                .prepare_cached("SELECT pk FROM session WHERE id=?1")?
                .query_row([u.session.to_string()], |row| row.get::<_, i64>(0))
                .optional()?
        {
            changes.call_result(tx, session, call_id)?;
            changes.session(tx, session)?;
        }
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
            u.reported_duration_ms.and_then(|ms| i64::try_from(ms).ok())
        ])?;
        // The call's outcome changed outside its messages.
        tx.prepare_cached(
            "UPDATE session_span SET prefix = NULL
             WHERE session_pk = (SELECT pk FROM session WHERE id = ?1)",
        )?
        .execute([u.session.to_string()])?;
    }
    for pk in sessions.values() {
        tx.prepare_cached(
            "INSERT OR IGNORE INTO parsed(input_pk, chain, session_pk) VALUES (?1, ?2, ?3)",
        )?
        .execute(params![input_pk, chain, pk])?;
    }
    let source_commit = source_commit.or(p.source_commit.as_ref());
    if retraction && source_commit.is_none() {
        crate::wire_shapes::unobserved(tx, input_pk, chain)?;
        tx.execute(
            "DELETE FROM parse_state WHERE input_pk=?1 AND chain=?2",
            params![input_pk, chain],
        )?;
        return Ok(written);
    }
    let source_commit = source_commit.or_invalid("an indexed reading has no source commit")?;
    let apply_wire = if replacement {
        crate::wire_shapes::replace
    } else {
        crate::wire_shapes::apply
    };
    apply_wire(
        tx,
        crate::wire_shapes::Source {
            input_pk,
            input: &b.input.id,
            format: &b.input.format,
            chain,
            length: b.length,
            commit_sequence: source_commit.sequence,
            commit_id: source_commit.id.as_deref(),
            observation: source_commit.observation.as_deref(),
            observed_at_ms: source_commit.observed_at_ms,
        },
        p,
    )?;
    if !crate::parser_context::apply(tx, input_pk, chain, p, &rows.versions)? {
        rows.checkpoint.forget_context();
    }
    rows.write_parse_state(b, p)?;
    Ok(written)
}

/// Ends a transaction that applied rows: brings the session item index up
/// to date from what the transaction changed and deletes the version bodies,
/// values and texts it left without references. Returns the item units read.
pub(crate) fn finish(
    tx: &Transaction<'_>,
    changes: &ChangeSummary,
    first_build: Option<&KeyMaps>,
    texts: &crate::text_rows::TextIds,
    counter: i64,
) -> Result<u64> {
    changes.finish_summaries(tx, Some(counter))?;
    changes.finish_links(tx)?;
    let read = crate::items::refresh(tx, changes)?;
    crate::session_values::refresh(tx, changes)?;
    // A first build derives every session once, when it ends.
    if first_build.is_none() {
        let sessions = changes.sessions(tx)?;
        crate::derive::links::add_finished_links(tx, &sessions)?;
        crate::measure::time::forget_settled(tx, changes, &sessions)?;
    }
    crate::versions::remove_unreferenced(tx, first_build, texts)?;
    Ok(read)
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
fn accept_step(previous: Option<&checkpoint::Stored>, step: &ParseStep) -> Result<bool> {
    if step.reset {
        return Ok(true);
    }
    let cursor = previous.map_or(0, checkpoint::Stored::cursor);
    if step.start < cursor
        && step.parsed_upto <= cursor
        && previous.is_some_and(|prior| prior.version() == step.reading_version)
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
    messages: BTreeMap<i64, MessageWrite<'a>>,
    /// Results of registering this transaction's message payloads.
    versions: HashMap<[u8; 32], crate::versions::RegisteredMessage<'a>>,
    checkpoint: checkpoint::Update,
    source_committed: Option<i64>,
    tx: &'a Transaction<'t>,
    input_pk: i64,
    input_id: &'a str,
    chain: i64,
    /// The input's route; a message row read from page text has its own.
    route: Route,
    counter: i64,
    /// Set during a first build ([`Writer::start`]): search entries,
    /// trigram entries and secondary indexes are built once, in bulk, when
    /// it ends ([`Inner::build_deferred`]), and rows are found by natural
    /// key through it ([`KeyMaps`]).
    first_build: Option<&'a KeyMaps>,
    keys: &'a resolve::ResolvedKeys,
    choices: &'a resolve::membership::Choices,
    changes: &'a ChangeSummary,
    texts: &'a crate::text_rows::TextIds,
}

/// One selected message's application in this step. The creation result is
/// consumed by its first body installation, including early copy moves.
struct MessageWrite<'a> {
    choice: Option<&'a resolve::membership::MessageChoice>,
    value: Option<&'a resolve::membership::MessageValue>,
    created: Cell<Option<CreatedMessage>>,
}

struct CreatedMessage;

/// The main route's reading of a shown message.
pub(super) struct Version {
    time: Vec<u8>,
    input_pk: i64,
    chain: i64,
    start: i64,
    end: i64,
    in_progress: bool,
}

fn selected_version(
    choice: &resolve::membership::MessageChoice,
    value: Option<&resolve::membership::MessageValue>,
) -> Version {
    Version {
        time: value
            .map(|value| value.prepared.time.clone())
            .unwrap_or_default(),
        input_pk: choice.input,
        chain: choice.chain,
        start: choice.start,
        end: choice.end,
        in_progress: choice.in_progress,
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
    fn write_prefix_copy(
        &self,
        parent: &str,
        child: &str,
        explicit: bool,
        committed: Option<i64>,
    ) -> Result<()> {
        self.changes.link_child(self.tx, child)?;
        self.tx.prepare_cached("INSERT INTO prefix_copy(input_pk,chain,parent_id,child_id,committed,explicit_members) VALUES (?1,?2,?3,?4,?5,?6) ON CONFLICT(input_pk,chain,parent_id,child_id) DO UPDATE SET committed=coalesce(max(prefix_copy.committed,excluded.committed),prefix_copy.committed,excluded.committed),explicit_members=max(prefix_copy.explicit_members,excluded.explicit_members)")?
            .execute(params![self.input_pk,self.chain,parent,child,committed,explicit])?;
        let sessions = self
            .tx
            .prepare_cached("SELECT pk FROM session WHERE id IN (?1,?2)")?
            .query_map(params![parent, child], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for session in sessions {
            self.changes.session(self.tx, session)?;
        }
        Ok(())
    }

    /// The message with natural ID `id`.
    /// Moves what a copy's session holds to the session its name is an alias
    /// of (docs/capture.md, AI Studio through Drive): each reading's
    /// statements and events, and links naming it. Its messages move with
    /// their readings (`Choices::moved_messages`).
    fn absorb_session(&self, from: i64, to: i64) -> Result<()> {
        for table in [
            "session_record",
            "parsed",
            "message_source",
            "event",
            "event_source",
        ] {
            self.tx
                .prepare_cached(&format!(
                    "UPDATE {table} SET session_pk=?2 WHERE session_pk=?1"
                ))?
                .execute(params![from, to])?;
        }
        let (copy, session): (String, String) = self.tx.query_row(
            "SELECT (SELECT id FROM session WHERE pk=?1),(SELECT id FROM session WHERE pk=?2)",
            params![from, to],
            |r| <(_, _)>::try_from(r),
        )?;
        // A link naming the copy names the session; where the session already
        // has that link, the copy's link sources join it.
        let links = self
            .tx
            .prepare_cached(
                "SELECT pk,parent_id,child_id,kind,how,branch_point,spawning_call FROM link WHERE parent_id=?1 OR child_id=?1",
            )?
            .query_map([&copy], |r| <(i64, String, String, String, Option<String>, Option<String>, Option<String>)>::try_from(r))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for (pk, parent, child, kind, how, at, call) in links {
            let name = |id: String| if id == copy { session.clone() } else { id };
            let (parent, child) = (name(parent), name(child));
            self.changes.link(self.tx, pk)?;
            let existing: Option<i64> = self
                .tx
                .scalar_optional("SELECT pk FROM link WHERE parent_id=?1 AND child_id=?2 AND kind=?3 AND how IS ?4 AND branch_point IS ?5 AND spawning_call IS ?6", params![parent, child, kind, how, at, call])?;
            match existing {
                Some(existing) => {
                    self.tx.execute(
                        "UPDATE link_source SET link_pk=?2 WHERE link_pk=?1",
                        params![pk, existing],
                    )?;
                    self.tx.execute("DELETE FROM link WHERE pk=?1", [pk])?;
                }
                None => {
                    self.tx.execute(
                        "UPDATE link SET parent_id=?2,child_id=?3,child_pk=(SELECT pk FROM session WHERE id=?3) WHERE pk=?1",
                        params![pk, parent, child],
                    )?;
                }
            }
        }
        self.changes.whole_session(to);

        Ok(())
    }

    /// Returns a released copy's session statements to its own session,
    /// whose row comes back (docs/capture.md, AI Studio through Drive). Its
    /// messages and events move with their readings
    /// (`Choices::released_messages`, `Choices::released_events`).
    fn release_session(&self, from: i64, to: i64, copy: &str, chains: &[(i64, i64)]) -> Result<()> {
        let id = SessionId::parse(copy).map_err(crate::StoreError::invalid)?;
        self.session_row(&id, false)?;
        self.tx
            .prepare_cached("DELETE FROM removed_session WHERE session_id=?1")?
            .execute([copy])?;
        for (input, chain) in chains {
            for table in ["session_record", "parsed"] {
                self.tx
                    .prepare_cached(&format!(
                        "UPDATE {table} SET session_pk=?2 WHERE session_pk=?1 AND input_pk=?3 AND chain=?4"
                    ))?
                    .execute(params![from, to, input, chain])?;
            }
        }
        for pk in [from, to] {
            self.changes.whole_session(pk);
        }
        Ok(())
    }

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
    fn upsert_session(&self, s: &Session, committed: Option<i64>) -> Result<(i64, bool)> {
        let pk = self.session_row(&s.id, false)?;
        let merged = self
            .choices
            .sessions
            .get(&pk)
            .or_invalid("unresolved session facts")?;
        self.tx
            .prepare_cached(
                "INSERT INTO session_record(session_pk, input_pk, chain, facts, committed) VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(session_pk, input_pk, chain) DO UPDATE SET facts = excluded.facts, committed=coalesce(max(session_record.committed,excluded.committed),session_record.committed,excluded.committed)",
            )?
            .execute(params![pk, self.input_pk, self.chain, serde_json::to_string(&merged)?, committed])?;
        self.tx
            .prepare_cached(
                "DELETE FROM session_alias WHERE session_pk=?1 AND input_pk=?2 AND chain=?3",
            )?
            .execute(params![pk, self.input_pk, self.chain])?;
        for alias in &merged.aliases {
            self.tx
                .prepare_cached(
                    "INSERT INTO session_alias(alias,session_pk,input_pk,chain) VALUES (?1,?2,?3,?4)",
                )?
                .execute(params![alias.to_string(), pk, self.input_pk, self.chain])?;
        }
        let main = merged.kind == polylogue_model::SessionKind::Main;
        self.tx.prepare_cached("UPDATE message_source SET main=?4 WHERE session_pk=?1 AND input_pk=?2 AND chain=?3 AND main<>?4")?
            .execute(params![pk,self.input_pk,self.chain,main])?;
        self.changes.session(self.tx, pk)?;
        self.tx
            .prepare_cached("UPDATE session SET stub=0 WHERE pk=?1 AND stub IS NOT 0")?
            .execute([pk])?;
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
        self.tx.prepare_cached("INSERT INTO session(pk,id,origin,native_id,kind,cwd_json,changed,harness,stub,
            message_count,human_prompt_count,human_command_count,word_count,human_word_count,model_word_count,
            thinking_count,compaction_count,tool_call_count,error_count)
            VALUES (?1,?2,?3,?4,'main','[]',?5,?3,?6,0,0,0,0,0,0,0,0,0,0)")?.execute(params![pk,key,id.origin().as_str(),id.native().as_str(),self.counter,stub])?;
        self.tx
            .prepare_cached(
                "UPDATE link SET child_pk = ?1 WHERE child_id = ?2 AND child_pk IS NULL",
            )?
            .execute(params![pk, key])?;
        Ok(pk)
    }

    /// Stores a chain's reading, as version `version_pk`, beneath the resolved
    /// selected value; returns whether the message row was created here.
    #[allow(clippy::too_many_arguments)]
    fn write_message(
        &self,
        row: &MessageRow,
        session_pk: i64,
        prepared: &PreparedMessage,
        (position, order_time): ((i64, i64), Option<i64>),
        main: bool,
        version_pk: i64,
        sessions: &mut HashMap<String, i64>,
    ) -> Result<bool> {
        let m = &row.message;
        let id = m.id.to_string();
        let pk = self.message_pk(&id)?;
        let new = pk.is_none();
        let route = row.route(self.route);
        let completion = Completion::of(m, route);
        let at = ms(row.message.at);
        let v = Version {
            time: prepared.time.clone(),
            input_pk: self.input_pk,
            chain: self.chain,
            start: row.offset as i64,
            end: row.end.max(row.offset) as i64,
            in_progress: completion == Completion::InProgress,
        };
        let pk = match pk {
            Some(pk) => pk,
            None => {
                self.changes
                    .new_message(self.keys.get(resolve::Kind::Message, id.as_bytes())?.pk);
                let write = self
                    .messages
                    .get(&self.keys.get(resolve::Kind::Message, id.as_bytes())?.pk)
                    .or_invalid("unresolved message version")?;
                let pk = if let Some(choice) = write.choice.filter(|choice| choice.changed) {
                    let selected = write.value.map(|value| &value.message).ok_or_else(|| {
                        StoreError::Invalid("a changed version has no prepared message".into())
                    })?;
                    self.session_pk(sessions, &selected.session)?;
                    let pk = self.insert_message(
                        selected,
                        choice.session,
                        &selected_version(choice, write.value),
                        &choice.order,
                    )?;
                    write.created.set(Some(CreatedMessage));
                    pk
                } else {
                    let order = crate::encoding::session_order(
                        order_time,
                        self.input_id,
                        position.0,
                        position.1,
                    );
                    self.insert_message(m, session_pk, &v, &order)?
                };
                if let Some(first) = self.first_build {
                    first.add_message(&self.keys.identities.shown_message(&id), pk);
                }
                pk
            }
        };
        let id_kind = row.id_kind.as_str();
        let (route, completion) = (route.as_str(), completion.as_str());
        if new {
            self.tx.prepare_cached(
                "INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,id_kind,completion,version_pk,session_pk,main,route,seq,record_position,order_time,at)
                 VALUES (?1,?2,?3,?4,?5,?13,?6,?7,?8,?9,?10,?11,?12,?14,?15)"
            )?.execute(params![pk,v.input_pk,v.chain,v.start,v.end,completion,version_pk,session_pk,main,route,position.0,position.1,id_kind,order_time,at])?;
            return Ok(true);
        }
        self.tx.prepare_cached(
            "INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,id_kind,completion,version_pk,session_pk,main,route,seq,record_position,order_time,at)
             VALUES (?1,?2,?3,?4,?5,?13,?6,?7,?8,?9,?10,?11,?12,?14,?15)
             ON CONFLICT(message_pk,input_pk,chain) DO UPDATE SET source_start=excluded.source_start,
             source_end=excluded.source_end,id_kind=excluded.id_kind,completion=excluded.completion,version_pk=excluded.version_pk,
             session_pk=excluded.session_pk,main=excluded.main,route=excluded.route,
             seq=excluded.seq,record_position=excluded.record_position,order_time=excluded.order_time,at=excluded.at"
        )?.execute(params![pk,v.input_pk,v.chain,v.start,v.end,completion,version_pk,session_pk,main,route,position.0,position.1,id_kind,order_time,at])?;
        Ok(false)
    }

    fn refresh_message(&self, pk: i64) -> Result<bool> {
        let write = self
            .messages
            .get(&pk)
            .or_invalid("unresolved message version")?;
        let choice = write.choice;
        let value = write.value;
        let created = write.created.take();
        let winner = choice.map(|choice| (selected_version(choice, value), choice.session));
        let Some((v, session_pk)) = winner else {
            if !self
                .tx
                .prepare_cached("SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1)")?
                .query_row([pk], |r| r.get::<_, bool>(0))?
            {
                return Ok(false);
            }
            self.changes.message(self.tx, pk)?;
            self.clear_blocks(pk)?;
            self.detach_event_placement(pk)?;
            self.tx
                .execute("DELETE FROM message_route WHERE message_pk=?1", [pk])?;
            self.changes.replaced_message(pk)?;
            self.tx.execute("DELETE FROM message WHERE pk=?1", [pk])?;
            return Ok(true);
        };
        let choice = choice.expect("selected message");
        let order = &choice.order;
        if !choice.changed {
            let (unchanged, items_unchanged) = self
                .tx
                .prepare_cached(
                    "SELECT input_pk IS ?2 AND source_chain IS ?3 AND source_start IS ?4
                 AND source_end IS ?5 AND session_order IS ?6 AND in_progress IS ?7,
                 session_order IS ?6 AND in_progress IS ?7
                 FROM message WHERE pk=?1",
                )?
                .query_row(
                    params![
                        pk,
                        v.input_pk,
                        v.chain,
                        v.start,
                        v.end,
                        order,
                        v.in_progress
                    ],
                    |row| <(bool, bool)>::try_from(row),
                )?;
            if !unchanged {
                // Items and completion use the selected body and placement;
                // record citations are read from the current message row.
                if !items_unchanged {
                    self.changes.message_metadata(self.tx, pk)?;
                }
                self.tx
                    .prepare_cached(
                        "UPDATE message SET input_pk=?2,source_chain=?3,source_start=?4,source_end=?5,
                     session_order=?6,in_progress=?7 WHERE pk=?1",
                    )?
                    .execute(params![
                        pk,
                        v.input_pk,
                        v.chain,
                        v.start,
                        v.end,
                        order,
                        v.in_progress
                    ],
                )?;
            }
            // Adding or removing a stated source can change session
            // derivations even when the displayed message stays equal.
            self.changes.session(self.tx, session_pk)?;
            return Ok(false);
        }
        self.changes.message(self.tx, pk)?;
        let value = value.ok_or_else(|| {
            StoreError::Invalid("a changed version has no prepared message".into())
        })?;
        let m = &value.message;
        let prepared = &value.prepared;
        self.changes.prepare_items(pk, prepared.items.clone());
        self.changes.session(self.tx, session_pk)?;
        if created.is_none() {
            if self
                .tx
                .prepare_cached("SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1)")?
                .query_row([pk], |r| r.get::<_, bool>(0))?
            {
                self.clear_blocks(pk)?;
                self.update_message(pk, m, session_pk, &v, order)?;
                self.tx
                    .prepare_cached("DELETE FROM message_route WHERE message_pk=?1")?
                    .execute([pk])?;
            } else {
                // A copy joining its session can first materialize here.
                self.insert_message(m, session_pk, &v, order)?;
            }
        }
        self.insert_blocks(pk, session_pk, m, prepared)?;
        for (group, input, chain) in &choice.routes {
            self.tx
                .prepare_cached(
                    "INSERT INTO message_route(message_pk,field_group,input_pk,chain) VALUES (?1,?2,?3,?4)",
                )?
                .execute(params![pk, group.as_str(), input, chain])?;
        }
        Ok(true)
    }

    fn insert_message(
        &self,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: &str,
    ) -> Result<i64> {
        self.set_message(None, m, session_pk, v, order)
    }

    fn update_message(
        &self,
        pk: i64,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: &str,
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
        order: &str,
    ) -> Result<i64> {
        self.changes.session(self.tx, session_pk)?;
        let author = polylogue_model::WrittenBy::of(m);
        let written_by = author.as_str();
        // A copy's message is written with its ID in the session.
        let id = self
            .keys
            .identities
            .shown_message(&m.id.to_string())
            .into_owned();
        let parent = m.parent.as_ref().map(|p| {
            self.keys
                .identities
                .shown_message(&p.to_string())
                .into_owned()
        });
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
            "UPDATE message SET session_pk = ?2, session_order = ?3,
                 role = ?4, written_by = ?5, prompt_kind = ?6, model = ?7, at = ?8,
                 parent_id = ?9, input_pk = ?10, source_chain = ?11, source_start = ?12,
                 source_end = ?24, in_progress = ?13, input_tokens = ?14, output_tokens = ?15,
                 cache_read_tokens = ?16, cache_write_tokens = ?17, reasoning_tokens = ?18,
                 service_tier = ?19, stop_reason = ?20, duration_ms = ?21, attrs_json = ?22,
                 words = ?23, cost_nusd = ?29, provider_cost_nusd = ?25,
                 cache_write_5m_tokens = ?26, cache_write_1h_tokens = ?27, speed = ?28,
                 at_time = ?31
             WHERE pk = ?1 AND pk = ?30"
        } else {
            "INSERT INTO message(pk, id, session_pk, session_order, role, written_by,
                 prompt_kind, model, at, parent_id, input_pk, source_chain, source_start,
                 source_end, in_progress, input_tokens, output_tokens, cache_read_tokens,
                 cache_write_tokens, reasoning_tokens, service_tier, stop_reason, duration_ms,
                 attrs_json, words, provider_cost_nusd,
                 cache_write_5m_tokens, cache_write_1h_tokens, speed,
                 cost_nusd, at_time)
             VALUES (?30, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?24, ?13, ?14,
                 ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?25, ?26, ?27, ?28, ?29, ?31)"
        };
        let first: rusqlite::types::Value = match pk {
            Some(pk) => pk.into(),
            None => id.clone().into(),
        };
        self.tx.prepare_cached(sql)?.execute(params![
            first,
            session_pk,
            order,
            name(&m.role),
            written_by,
            m.prompt_kind.map(|k| name(&k)),
            m.model,
            ms(m.at),
            parent.as_deref(),
            v.input_pk,
            v.chain,
            v.start,
            i64::from(v.in_progress),
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
            // The cost derivation's value, written with the row so a first
            // build's derive does not rewrite every priced message.
            polylogue_model::Catalog::builtin().cost_nusd(m.model.as_deref(), u),
            // An update names its row; an insert takes the reserved key.
            match pk {
                Some(pk) => pk,
                None => {
                    self.keys.get(resolve::Kind::Message, id.as_bytes())?.pk
                }
            },
            v.time,
        ])?;
        let created = pk.is_none();
        let pk = pk.unwrap_or_else(|| self.tx.last_insert_rowid());
        if created {
            self.changes
                .installed_message(pk, session_pk, parent, m, words, author)?;
        } else {
            self.changes.replaced_message(pk)?;
        }
        Ok(pk)
    }

    /// Sources and selected rows now agree. Events need only their final
    /// anchors, including when an unchanged message's selected source moved.
    fn refresh_event_placement(&self) -> Result<()> {
        if self.choices.messages.is_empty()
            || !self
                .tx
                .prepare_cached(
                    "SELECT EXISTS(SELECT 1 FROM event WHERE after_message_id IS NOT NULL)",
                )?
                .query_row([], |row| row.get::<_, bool>(0))?
        {
            return Ok(());
        }
        let messages = serde_json::to_string(&self.choices.messages.keys().collect::<Vec<_>>())?;
        let events = self
            .tx
            .prepare_cached(
                "UPDATE event SET after_message_pk=m.pk,
                 seq=CASE WHEN seq_from_message=1 THEN coalesce(s.seq,0) ELSE event.seq END
             FROM json_each(?1) changed JOIN message m ON m.pk=changed.value
             LEFT JOIN message_source s ON s.message_pk=m.pk
                 AND s.input_pk=m.input_pk AND s.chain=m.source_chain
             WHERE event.after_message_id=m.id AND (event.after_message_pk IS NOT m.pk
                 OR (event.seq_from_message=1 AND event.seq IS NOT coalesce(s.seq,0)))
             RETURNING pk,session_pk",
            )?
            .query_map([messages], |row| <(_, _)>::try_from(row))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        self.apply_event_moves(events)
    }

    /// A referenced message cannot be deleted before its events are detached.
    /// Their natural anchor remains available for a later message reading.
    fn detach_event_placement(&self, message_pk: i64) -> Result<()> {
        let id: String = self
            .tx
            .prepare_cached("SELECT id FROM message WHERE pk=?1")?
            .query_row([message_pk], |r| r.get(0))?;
        let events = self
            .tx
            .prepare_cached(
                "UPDATE event SET after_message_pk=NULL,
             seq=CASE WHEN seq_from_message=1 THEN 0 ELSE seq END
             WHERE after_message_id=?1 AND (after_message_pk IS NOT NULL
                OR (seq_from_message=1 AND seq IS NOT 0)) RETURNING pk, session_pk",
            )?
            .query_map([id], |r| <(i64, Option<i64>)>::try_from(r))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        self.apply_event_moves(events)
    }

    fn apply_event_moves(&self, events: Vec<(i64, Option<i64>)>) -> Result<()> {
        for (event, session) in events {
            // Only its place moved; its session is the one it had.
            self.changes.event(self.tx, event)?;
            let Some(session) = session else { continue };
            self.changes.session(self.tx, session)?;
        }
        Ok(())
    }

    /// Removes a message's blocks, keeping search entries and tool-call
    /// pairs in step, before another version replaces them. It runs before
    /// the message row changes: a version from another session moves the
    /// message, and its old blocks must not outlive the move.
    fn clear_blocks(&self, message_pk: i64) -> Result<()> {
        self.changes.remove_results(self.tx, message_pk)?;
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
            .query_map([message_pk], |r| <(_, _, _)>::try_from(r))?
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
            crate::text_rows::unindex_unused(
                self.tx,
                self.texts,
                text,
                self.first_build.is_some(),
            )?;
        }
        Ok(())
    }

    /// The search indexes the texts of the blocks this step installs belong
    /// to, by text hash: every changed selected message's blocks are written
    /// ([`Self::refresh_message`]).
    fn current_texts(&self) -> HashMap<[u8; 32], i64> {
        let mut bits = HashMap::new();
        for write in self.messages.values() {
            let (Some(choice), Some(value)) = (write.choice, write.value) else {
                continue;
            };
            if !choice.changed {
                continue;
            }
            for (block, text) in value.message.blocks.iter().zip(&value.prepared.texts) {
                if let Some(hash) = text {
                    *bits.entry(*hash).or_default() |= search_bit(block);
                }
            }
        }
        bits
    }

    fn version_pk(&self, hash: &[u8; 32]) -> Result<i64> {
        match self.versions.get(hash) {
            Some(version) => Ok(version.pk()),
            None => crate::versions::message_key(self.tx, hash),
        }
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
        prepared.block_selection.validate(m.blocks.len())?;
        let stored;
        let stored_texts = match self.versions.get(&prepared.hash).and_then(|v| v.text_ids()) {
            Some(ids) => ids,
            None => {
                stored =
                    crate::versions::message_text_ids(self.tx, self.version_pk(&prepared.hash)?)?;
                &stored
            }
        };
        for (idx, ((block, body), data)) in m
            .blocks
            .iter()
            .zip(&prepared.blocks)
            .zip(&prepared.data)
            .enumerate()
        {
            let text_id = prepared.texts[idx]
                .as_ref()
                .map(|_| {
                    stored_texts.get(&idx).copied().ok_or_else(|| {
                        StoreError::Invalid("stored message text reference is missing".into())
                    })
                })
                .transpose()?;
            let stored = StoredBlock {
                body,
                data: data.as_deref(),
                text_id,
            };
            self.insert_block(
                message_pk,
                session_pk,
                m.session.origin().as_str(),
                (idx as i64, prepared.block_selection.block_ordinals[idx]),
                block,
                stored,
                ms(m.at),
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_block(
        &self,
        message_pk: i64,
        session_pk: i64,
        origin: &str,
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
        let bit = search_bit(block);
        if let Some(id) = text_id {
            crate::text_rows::activate(self.tx, self.texts, id, bit, self.first_build.is_some())?;
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
        // A copy's call IDs name its message; they take the message's ID in
        // the session its copy name is an alias of, as the message does.
        let call_id = call_id.map(|id| self.keys.identities.shown_call(origin, id));
        let call_id = call_id.as_deref();
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
                call_id: Some(_),
                outcome,
                reported_duration_ms,
                ..
            } => {
                let (outcome, reason, exit) = outcome_columns(outcome);
                if let Some(call_id) = call_id {
                    self.changes.call_result(self.tx, session_pk, call_id)?;
                }
                self.tx
                    .prepare_cached(
                        "UPDATE tool_call SET result_block_pk = ?1, outcome = ?2, outcome_reason = ?3,
                             exit_code = ?4, result_at=?7,reported_duration_ms=?8
                         WHERE session_pk = ?5 AND call_id = ?6",
                    )?
                    .execute(params![block_pk, outcome, reason, exit, session_pk, call_id,at,reported_duration_ms.and_then(|ms| i64::try_from(ms).ok())])?;
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
        call_id: Option<&str>,
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
            .query_row(params![session_pk, call_id], |r| <(_, _, _)>::try_from(r))
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
                    reported_duration_ms.and_then(|ms| i64::try_from(ms).ok())
                ])?;
        }
        Ok(())
    }

    fn write_event(&self, index: usize) -> Result<()> {
        let id = &self.choices.event_ids[index];
        let (pk, source) = &self.choices.event_sources[index];
        // The resolver has already distinguished stored events from new identities.
        // Another stated source alone does not change the selected event's items.
        if !self.keys.get(resolve::Kind::Event, id.as_bytes())?.existing {
            self.changes.event(self.tx, *pk)?;
        }
        self.tx.prepare_cached("INSERT OR IGNORE INTO event(pk,id,session_pk,seq,kind,at,input_pk,source_chain,source_start,source_end,fields,data)
            VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)")?
            .execute(params![pk,id,source.session,source.seq.unwrap_or(0),source.kind,source.at,self.input_pk,self.chain,source.start,source.end,source.fields,source.body])?;
        if let Some(first) = self.first_build {
            first.add_event(id, *pk);
        }
        self.store_event_value(index)
    }

    fn write_link(&self, l: &Link, index: usize, committed: Option<i64>) -> Result<()> {
        self.store_link(l, index, committed)
    }

    /// Parse position and record counts for the chain, committed with the rows
    /// they cover. A step from offset 0 replaces the chain's counts; a
    /// continuing step adds to them.
    fn write_parse_state(&self, b: &Batch, p: &ParseStep) -> Result<()> {
        self.checkpoint.write(self.tx, self.input_pk, self.chain)?;
        let a = &p.counts;
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

/// The search index a block's text belongs to: 2 for tool output, 1 for prose.
fn search_bit(block: &Block) -> i64 {
    if matches!(block, Block::ToolResult { .. } | Block::ToolCall { .. }) {
        2
    } else {
        1
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

/// A provider's reported amount is an observed fact, separate from catalog
/// pricing. The number keeps its exact spelling (`arbitrary_precision`), so
/// it converts to nano-USD without passing through a float.
fn reported_nusd(usd: Option<&serde_json::Number>) -> Option<i64> {
    polylogue_model::money::nusd_from_usd(&usd?.to_string())
}
