//! The shown rows (session, message, message_route, block, tool_call,
//! event, link, document) computed from the readings present
//! (docs/architecture.md, composition). A transaction collects what its
//! readings changed ([`Touched`]); before it commits, the shown rows of
//! those things are computed again by the composition functions of
//! [`super::compose`] and written where they differ from what is stored.
//! Nothing is corrected by the difference a step made: each shown row is a
//! function of the readings present, so it does not depend on the order or
//! division of the steps that brought them. Names stay as the source wrote
//! them: a message's parent and a link's ends are joined when read, a tool
//! result is joined to its call by call ID within the session, and an event
//! names the message it follows by ID.
use super::compose;
use super::identity::Identities;
use super::readings::{self, Touched};
use super::*;
use crate::change_summary::ChangeSummary;
use crate::encoding::Columns as _;
use polylogue_model::{Completion, Document, FieldGroup, MessageId, Route};
use rusqlite::types::Value;
use std::collections::{BTreeMap, BTreeSet};

/// What composition writes with.
pub(super) struct Context<'a> {
    pub tx: &'a Transaction<'a>,
    pub counter: i64,
    pub texts: &'a crate::text_rows::TextIds,
    pub at_build_end: bool,
    pub changes: &'a ChangeSummary,
}

/// One chain's reading of a message.
#[derive(Debug, Clone)]
struct MessageReading {
    input_pk: i64,
    input: String,
    chain: i64,
    /// The message and session as the chain names them.
    message: String,
    session: String,
    start: i64,
    end: i64,
    seq: i64,
    position: i64,
    order_time: Option<i64>,
    route: Route,
    completion: Completion,
    content: bool,
    hash: Vec<u8>,
}

/// The value of a reading: its message with every block's values, and its
/// packed rows.
type Read = (Message, PreparedMessage);

/// Writes the shown rows of everything `touched` names; returns the number
/// of message rows written, changed or removed.
pub(super) fn compose(context: &Context<'_>, touched: Touched) -> Result<u64> {
    let mut work = Work {
        context,
        ids: Identities::new(context.tx),
        sessions: BTreeMap::new(),
        calls: BTreeSet::new(),
        events: BTreeSet::new(),
        written: 0,
    };
    work.run(touched)?;
    Ok(work.written)
}

struct Work<'a, 'c> {
    context: &'a Context<'c>,
    ids: Identities<'c>,
    /// Session rows by shown ID, created as needed.
    sessions: BTreeMap<String, i64>,
    /// Calls whose pairing with a result is computed again: (session row,
    /// call ID).
    calls: BTreeSet<(i64, String)>,
    /// Shown events to compose again because the message they follow moved:
    /// (stated session, stated event ID).
    events: BTreeSet<(Option<String>, String)>,
    written: u64,
}

impl<'c> Work<'_, 'c> {
    fn tx(&self) -> &'c Transaction<'c> {
        self.context.tx
    }

    fn run(&mut self, touched: Touched) -> Result<()> {
        let Touched {
            sessions,
            whole,
            messages,
            events,
            links,
            documents,
            calls,
            released_texts,
            live,
        } = touched;
        // Sessions composed whole: every message, event and link their
        // readings state, and every row they show now, are composed again.
        let mut shown_whole = BTreeSet::new();
        for stated in &whole {
            shown_whole.insert(stated.clone());
            shown_whole.insert(self.ids.session(stated)?);
        }
        let mut message_ids = BTreeSet::new();
        let mut event_work = events;
        let mut link_work = links;
        for (_, message) in &messages {
            message_ids.insert(self.ids.message(message)?);
        }
        let mut whole_sessions = Vec::new();
        for session in &shown_whole {
            let stated = self.ids.stated(session)?;
            for id in &stated {
                for message in self
                    .tx()
                    .prepare_cached("SELECT message_id FROM message_parse WHERE session_id=?1")?
                    .query_map([id], |row| row.get::<_, String>(0))?
                {
                    message_ids.insert(self.ids.message(&message?)?);
                }
                for row in self
                    .tx()
                    .prepare_cached(
                        "SELECT session_id,event_id FROM event_parse WHERE session_id=?1",
                    )?
                    .query_map([id], |row| <(Option<String>, String)>::try_from(row))?
                {
                    event_work.insert(row?);
                }
                for row in self
                    .tx()
                    .prepare_cached(
                        "SELECT statement FROM link_parse WHERE parent_id=?1 UNION SELECT statement FROM link_parse WHERE child_id=?1",
                    )?
                    .query_map([id], |row| row.get::<_, Vec<u8>>(0))?
                {
                    link_work.insert(row?);
                }
            }
            for message in self
                .tx()
                .prepare_cached(
                    "SELECT m.id FROM message m JOIN session s ON s.pk=m.session_pk WHERE s.id=?1",
                )?
                .query_map([session], |row| row.get::<_, String>(0))?
            {
                message_ids.insert(message?);
            }
            whole_sessions.push(session.clone());
        }
        let mut session_work: BTreeSet<String> = BTreeSet::new();
        for stated in sessions.iter().chain(&whole) {
            session_work.insert(self.ids.session(stated)?);
            session_work.insert(stated.clone());
        }
        for message in &message_ids {
            if let Some(session) = self.compose_message(message)? {
                session_work.insert(session);
            }
        }
        self.pair_calls(calls)?;
        event_work.append(&mut std::mem::take(&mut self.events));
        let mut shown_events = BTreeMap::<String, BTreeSet<String>>::new();
        for (session, event) in event_work {
            let (shown, stated) = self.shown_event_id(&event)?;
            if let Some(session) = session {
                session_work.insert(self.ids.session(&session)?);
            }
            shown_events.entry(shown).or_default().extend(stated);
        }
        for session in &whole_sessions {
            // An event a session shows that no reading states any more.
            for id in self
                .tx()
                .prepare_cached(
                    "SELECT e.id FROM event e JOIN session s ON s.pk=e.session_pk WHERE s.id=?1",
                )?
                .query_map([session], |row| row.get::<_, String>(0))?
            {
                let id = id?;
                shown_events.entry(id.clone()).or_default().insert(id);
            }
        }
        for (shown, stated) in shown_events {
            self.compose_event(&shown, &stated)?;
        }
        let mut shown_links = BTreeMap::<Vec<u8>, BTreeSet<Vec<u8>>>::new();
        for statement in link_work {
            let shown = self.shown_statement(&statement)?;
            shown_links.entry(shown).or_default().insert(statement);
        }
        for session in &whole_sessions {
            for row in self
                .tx()
                .prepare_cached(
                    "SELECT parent_id,child_id,kind,how,branch_point,spawning_call FROM link WHERE parent_id=?1
                     UNION SELECT parent_id,child_id,kind,how,branch_point,spawning_call FROM link WHERE child_id=?1",
                )?
                .query_map([session], |row| {
                    <(String, String, String, Option<String>, Option<String>, Option<String>)>::try_from(row)
                })?
            {
                let (parent, child, kind, how, at, call) = row?;
                let statement = serde_json::to_vec(&(parent, child, kind, how, at, call))?;
                shown_links.entry(statement.clone()).or_default().insert(statement);
            }
        }
        for (shown, stated) in shown_links {
            self.compose_link(&shown, &stated)?;
        }
        for document in documents {
            self.compose_document(&document)?;
        }
        for session in session_work {
            self.settle_session(&session)?;
        }
        for event in live {
            let Some(session) = &event.session else {
                continue;
            };
            let shown = self.ids.session(&session.to_string())?;
            if let Some(pk) = self.session_pk(&shown)? {
                crate::live::apply(
                    self.tx(),
                    pk,
                    &event.kind,
                    event.at.map(|t| t.unix_ms()),
                    &event.data,
                    self.context.counter,
                )?;
            }
        }
        readings::remove_released_texts(
            self.tx(),
            self.context.texts,
            &released_texts,
            self.context.at_build_end,
        )?;
        Ok(())
    }

    /// The session row of a shown session, when it has one.
    fn session_pk(&mut self, session: &str) -> Result<Option<i64>> {
        if let Some(pk) = self.sessions.get(session) {
            return Ok(Some(*pk));
        }
        let pk: Option<i64> = self
            .tx()
            .scalar_optional("SELECT pk FROM session WHERE id=?1", [session])?;
        if let Some(pk) = pk {
            self.sessions.insert(session.to_owned(), pk);
        }
        Ok(pk)
    }

    /// The session row of a shown session, created on first sight.
    fn session_row(&mut self, session: &str) -> Result<i64> {
        if let Some(pk) = self.session_pk(session)? {
            return Ok(pk);
        }
        let id = SessionId::parse(session).map_err(StoreError::invalid)?;
        self.tx().prepare_cached("INSERT INTO session(id,origin,native_id,kind,cwd_json,changed,harness,stub,
            message_count,human_prompt_count,human_command_count,word_count,human_word_count,model_word_count,
            thinking_count,compaction_count,tool_call_count,error_count)
            VALUES (?1,?2,?3,'main','[]',?4,?2,1,0,0,0,0,0,0,0,0,0,0)")?
            .execute(params![session, id.origin().as_str(), id.native().as_str(), self.context.counter])?;
        let pk = self.tx().last_insert_rowid();
        self.sessions.insert(session.to_owned(), pk);
        Ok(pk)
    }

    /// A session exists while some reading states it, under its own ID or
    /// a copy's; it is a stub while only events name it. Its own columns
    /// are derived from its readings' facts (derive/session.rs).
    fn settle_session(&mut self, session: &str) -> Result<()> {
        let stated = self.ids.stated(session)?;
        // A session is a stub while only events name it: no reading states
        // its facts or a message of it.
        let mut own = false;
        let mut stated_at_all = false;
        for id in &stated {
            own |= self.tx().exists(
                "SELECT EXISTS(SELECT 1 FROM session_record WHERE session_id=?1)
                 OR EXISTS(SELECT 1 FROM message_parse WHERE session_id=?1)",
                [id],
            )?;
            stated_at_all |= own
                || self.tx().exists(
                    "SELECT EXISTS(SELECT 1 FROM event_parse WHERE session_id=?1)",
                    [id],
                )?;
        }
        let held = self.session_pk(session)?;
        match (held, stated_at_all) {
            (None, false) => Ok(()),
            (Some(pk), false) => {
                if self.tx().exists(
                    "SELECT EXISTS(SELECT 1 FROM message WHERE session_pk=?1)
                     OR EXISTS(SELECT 1 FROM event WHERE session_pk=?1)",
                    [pk],
                )? {
                    return Err(StoreError::Invalid(
                        "a session without readings still shows rows".into(),
                    ));
                }
                crate::session_row::remove(self.tx(), pk, session, self.context.counter)?;
                self.sessions.remove(session);
                Ok(())
            }
            (_, true) => {
                let pk = self.session_row(session)?;
                self.context.changes.session(self.tx(), pk)?;
                self.tx()
                    .prepare_cached("UPDATE session SET stub=?2 WHERE pk=?1 AND stub IS NOT ?2")?
                    .execute(params![pk, !own])?;
                Ok(())
            }
        }
    }

    /// Each input's reading of a shown message: that of the newest of its
    /// chains holding it. A content-named message an input's newer chain,
    /// read to its end, no longer holds is gone from that input.
    fn shown_message_parses(&mut self, shown: &str) -> Result<Vec<MessageReading>> {
        let mut all = Vec::new();
        for stated in self.ids.stated_messages(shown)? {
            let rows = self
                .tx()
                .prepare_cached(
                    "SELECT r.input_pk,i.id,r.chain,r.message_id,r.session_id,r.source_start,r.source_end,r.seq,
                         r.record_position,r.order_time,r.route,r.completion,r.id_kind='content',r.hash
                     FROM message_parse r JOIN input_ref i ON i.pk=r.input_pk WHERE r.message_id=?1",
                )?
                .query_map([&stated], |r| {
                    Ok(MessageReading {
                        input_pk: r.get(0)?,
                        input: r.get(1)?,
                        chain: r.get(2)?,
                        message: r.get(3)?,
                        session: r.get(4)?,
                        start: r.get(5)?,
                        end: r.get(6)?,
                        seq: r.get(7)?,
                        position: r.get(8)?,
                        order_time: r.get(9)?,
                        route: r.name(10)?,
                        completion: r.name(11)?,
                        content: r.get(12)?,
                        hash: r.get(13)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            for row in rows {
                if self.ids.message(&row.message)? == shown {
                    all.push(row);
                }
            }
        }
        let mut newest = BTreeMap::<i64, i64>::new();
        for r in &all {
            let chain = newest.entry(r.input_pk).or_insert(r.chain);
            *chain = (*chain).max(r.chain);
        }
        all.retain(|r| newest.get(&r.input_pk) == Some(&r.chain));
        let mut kept = Vec::with_capacity(all.len());
        for r in all {
            if r.content
                && self.tx().exists(
                    "SELECT EXISTS(SELECT 1 FROM parse_state WHERE input_pk=?1 AND chain>?2 AND status IN ('done','waiting'))",
                    params![r.input_pk, r.chain],
                )?
            {
                continue;
            }
            kept.push(r);
        }
        // A stable order, so that ties in composition fall the same way.
        kept.sort_by(|a, b| (&a.input, a.chain).cmp(&(&b.input, b.chain)));
        Ok(kept)
    }

    /// The kind its own chain states for a reading's session.
    fn reading_kind(&self, r: &MessageReading) -> Result<Option<polylogue_model::SessionKind>> {
        let kind: Option<Option<String>> = self
            .tx()
            .prepare_cached(
                "SELECT json_extract(facts,'$.kind') FROM session_record WHERE input_pk=?1 AND chain=?2 AND session_id=?3",
            )?
            .query_row(params![r.input_pk, r.chain, r.session], |row| row.get(0))
            .optional()?;
        kind.flatten()
            .map(|kind| {
                polylogue_model::SessionKind::from_name(&kind)
                    .ok_or_else(|| StoreError::stored(0, "unknown stored session kind"))
            })
            .transpose()
    }

    /// A reading's value; a reading holding some of its version's blocks
    /// shows every block the readings of that version hold.
    fn value(&mut self, r: &MessageReading) -> Result<Read> {
        let (message, selection) =
            readings::read_message(self.tx(), r.input_pk, r.chain, &r.message)?
                .ok_or_else(|| StoreError::Invalid("a message reading has no body".into()))?;
        let (message, selection) = if selection.block_count == message.blocks.len() as u64 {
            (message, selection)
        } else {
            let shown = self.ids.message(&r.message)?;
            let ids = self.ids.stated_messages(&shown)?;
            readings::read_version(self.tx(), &ids, &r.hash)?.unwrap_or((message, selection))
        };
        let prepared = PreparedMessage::projected(&message, &selection)?;
        Ok((message, prepared))
    }

    /// Composes one shown message from its readings and writes its rows;
    /// returns the session it is shown in, if any.
    fn compose_message(&mut self, shown: &str) -> Result<Option<String>> {
        let readings = self.shown_message_parses(shown)?;
        let stored: Option<(i64, i64)> = self
            .tx()
            .prepare_cached("SELECT pk,session_pk FROM message WHERE id=?1")?
            .query_row([shown], |r| <(i64, i64)>::try_from(r))
            .optional()?;
        if readings.is_empty() {
            if let Some((pk, session_pk)) = stored {
                self.remove_message(pk, session_pk, shown)?;
                self.written += 1;
            }
            return Ok(None);
        }
        if readings.len() == 1 && self.unchanged(&readings[0], stored)? {
            return Ok(Some(self.ids.session(&readings[0].session)?));
        }
        let mut values: Vec<Option<Read>> = vec![None; readings.len()];
        let (main, groups, mut message, prepared) = if readings.len() == 1 {
            let value = self.value(&readings[0])?;
            (0, Vec::new(), value.0, value.1)
        } else {
            for (n, r) in readings.iter().enumerate() {
                values[n] = Some(self.value(r)?);
            }
            let mut kinds = Vec::with_capacity(readings.len());
            for r in &readings {
                kinds.push(self.reading_kind(r)?);
            }
            let composed = {
                let inputs: Vec<_> = readings
                    .iter()
                    .zip(&values)
                    .zip(&kinds)
                    .map(|((r, value), kind)| compose::MessageReading {
                        input: &r.input,
                        route: r.route,
                        main: kind.is_none_or(|kind| kind == polylogue_model::SessionKind::Main),
                        completion: r.completion,
                        hash: &r.hash,
                        message: &value.as_ref().expect("loaded").0,
                    })
                    .collect();
                compose::message(&inputs, |n| {
                    Ok(kinds[n]
                        .is_some_and(|kind| kind != polylogue_model::SessionKind::Compaction))
                })?
                .expect("readings are not empty")
            };
            let prepared = values[composed.main].take().expect("loaded").1;
            (composed.main, composed.groups, composed.message, prepared)
        };
        let r = &readings[main];
        let session = self.ids.session(&r.session)?;
        let session_pk = self.session_row(&session)?;
        // A copy's message is shown in its session, under its name there.
        message.id = MessageId::parse(shown).map_err(StoreError::invalid)?;
        message.session = SessionId::parse(&session).map_err(StoreError::invalid)?;
        if let Some(parent) = &message.parent {
            let renamed = self.ids.message(&parent.to_string())?;
            message.parent = Some(MessageId::parse(&renamed).map_err(StoreError::invalid)?);
        }
        let in_progress = r.completion == Completion::InProgress
            && !groups.iter().any(|(group, _)| *group == FieldGroup::Stop);
        // The earliest place among the readings that know an order time, so
        // the message stays before what follows it in each input; a reading
        // that knows none never pulls it to the session's start.
        let place = readings
            .iter()
            .min_by_key(|m| {
                (
                    m.order_time.is_none(),
                    m.order_time,
                    &m.input,
                    m.seq,
                    m.position,
                )
            })
            .expect("readings are not empty");
        let order = crate::encoding::session_order(
            place.order_time,
            &place.input,
            place.seq,
            place.position,
        );
        let mut routes: Vec<(FieldGroup, i64, i64)> = groups
            .iter()
            .map(|(group, n)| (*group, readings[*n].input_pk, readings[*n].chain))
            .collect();
        routes.sort();
        let layout = layout_column(&prepared.block_selection)?;
        let row = message_columns(
            &message,
            MainReading {
                session_pk,
                order: &order,
                input_pk: r.input_pk,
                chain: r.chain,
                start: r.start,
                end: r.end,
                in_progress,
                time: crate::encoding::time_bytes(message.at, message.time_accuracy())?,
                version: &r.hash,
                layout: layout.as_deref(),
            },
        )?;
        let stored_routes = match stored {
            Some((pk, _)) => self
                .tx()
                .prepare_cached("SELECT field_group,input_pk,chain FROM message_route WHERE message_pk=?1 ORDER BY field_group")?
                .query_map([pk], |row| <(String, i64, i64)>::try_from(row))?
                .collect::<rusqlite::Result<Vec<_>>>()?,
            None => Vec::new(),
        };
        let mut new_routes: Vec<(String, i64, i64)> = routes
            .iter()
            .map(|(group, input, chain)| (group.as_str().to_owned(), *input, *chain))
            .collect();
        new_routes.sort();
        let stored_row = match stored {
            Some((pk, _)) => Some(self.stored_message_columns(pk)?),
            None => None,
        };
        let pk = match (stored, &stored_row) {
            (Some((pk, before_session)), Some(before)) => {
                let content_same = before[..CONTENT_COLUMNS] == row[..CONTENT_COLUMNS]
                    && stored_routes == new_routes;
                if content_same {
                    if before[CONTENT_COLUMNS..] != row[CONTENT_COLUMNS..] {
                        // Where the main reading is changes no item; its
                        // place and completion do.
                        if before[CONTENT_COLUMNS] != row[CONTENT_COLUMNS]
                            || before[IN_PROGRESS] != row[IN_PROGRESS]
                        {
                            self.context.changes.message_metadata(self.tx(), pk)?;
                        }
                        self.update_message(pk, &row)?;
                        self.events_after(shown, false)?;
                    }
                    self.context.changes.session(self.tx(), session_pk)?;
                    return Ok(Some(session));
                }
                self.context.changes.message(self.tx(), pk)?;
                self.clear_blocks(pk)?;
                self.update_message(pk, &row)?;
                self.tx()
                    .prepare_cached("DELETE FROM message_route WHERE message_pk=?1")?
                    .execute([pk])?;
                if before_session != session_pk {
                    self.context.changes.session(self.tx(), before_session)?;
                }
                self.events_after(shown, true)?;
                pk
            }
            _ => {
                self.tx()
                    .prepare_cached(&format!(
                        "INSERT INTO message(id,{}) VALUES (?1,{})",
                        MESSAGE_COLUMNS.join(","),
                        (2..=MESSAGE_COLUMNS.len() + 1)
                            .map(|n| format!("?{n}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    ))?
                    .execute(rusqlite::params_from_iter(
                        std::iter::once(Value::Text(shown.to_owned())).chain(row.iter().cloned()),
                    ))?;
                let pk = self.tx().last_insert_rowid();
                self.context.changes.new_message(pk);
                self.events_after(shown, true)?;
                pk
            }
        };
        self.context.changes.session(self.tx(), session_pk)?;
        let mut items = prepared.items.clone();
        polylogue_model::read::update_message_entries(
            &mut items,
            &message,
            polylogue_model::read::MessagePlace::default(),
        );
        self.context.changes.prepare_items(pk, items);
        self.insert_blocks(pk, session_pk, &message, &prepared)?;
        for (group, input, chain) in &routes {
            self.tx()
                .prepare_cached(
                    "INSERT INTO message_route(message_pk,field_group,input_pk,chain) VALUES (?1,?2,?3,?4)",
                )?
                .execute(params![pk, group.as_str(), input, chain])?;
        }
        self.written += 1;
        Ok(Some(session))
    }

    /// A message shown whole from one reading is unchanged when its row
    /// already shows that reading, in its session and place: nothing is
    /// read or written for it.
    fn unchanged(&mut self, r: &MessageReading, stored: Option<(i64, i64)>) -> Result<bool> {
        let Some((pk, session_pk)) = stored else {
            return Ok(false);
        };
        let session = self.ids.session(&r.session)?;
        if self.session_pk(&session)? != Some(session_pk) {
            return Ok(false);
        }
        let layout: Vec<u8> = self
            .tx()
            .prepare_cached(
                "SELECT layout FROM message_parse WHERE input_pk=?1 AND chain=?2 AND message_id=?3",
            )?
            .query_row(params![r.input_pk, r.chain, r.message], |row| row.get(0))?;
        let selection = crate::encoding::message_block_selection(&r.hash, Some(&layout), 0)?;
        let layout = layout_column(&selection)?;
        let order = crate::encoding::session_order(r.order_time, &r.input, r.seq, r.position);
        let in_progress = r.completion == Completion::InProgress;
        let same = self.tx().exists(
            "SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1 AND version_hash IS ?2 AND layout IS ?3
                 AND input_pk=?4 AND source_chain=?5 AND source_start=?6 AND source_end=?7
                 AND session_order=?8 AND in_progress=?9)
             AND NOT EXISTS(SELECT 1 FROM message_route WHERE message_pk=?1)",
            params![pk, r.hash, layout, r.input_pk, r.chain, r.start, r.end, order, in_progress],
        )?;
        if same {
            self.context.changes.session(self.tx(), session_pk)?;
        }
        Ok(same)
    }

    fn stored_message_columns(&self, pk: i64) -> Result<Vec<Value>> {
        let mut statement = self.tx().prepare_cached(&format!(
            "SELECT {} FROM message WHERE pk=?1",
            MESSAGE_COLUMNS.join(",")
        ))?;
        Ok(statement.query_row([pk], |row| {
            (0..MESSAGE_COLUMNS.len())
                .map(|n| row.get::<_, Value>(n))
                .collect::<rusqlite::Result<Vec<_>>>()
        })?)
    }

    fn update_message(&self, pk: i64, row: &[Value]) -> Result<()> {
        self.tx()
            .prepare_cached(&format!(
                "UPDATE message SET {} WHERE pk=?1",
                MESSAGE_COLUMNS
                    .iter()
                    .enumerate()
                    .map(|(n, column)| format!("{column}=?{}", n + 2))
                    .collect::<Vec<_>>()
                    .join(",")
            ))?
            .execute(rusqlite::params_from_iter(
                std::iter::once(Value::Integer(pk)).chain(row.iter().cloned()),
            ))?;
        Ok(())
    }

    /// The events placed after a message move with it: their place in the
    /// item index changes, and those that take their place in the input
    /// from the message's reading are composed again.
    fn events_after(&mut self, message: &str, placement: bool) -> Result<()> {
        let rows = self
            .tx()
            .prepare_cached("SELECT pk,id FROM event WHERE after_message_id=?1")?
            .query_map([message], |row| <(i64, String)>::try_from(row))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for (pk, id) in rows {
            if placement {
                self.context.changes.event(self.tx(), pk)?;
            }
            self.events.insert((None, id));
        }
        Ok(())
    }

    fn remove_message(&mut self, pk: i64, session_pk: i64, id: &str) -> Result<()> {
        self.context.changes.message(self.tx(), pk)?;
        self.clear_blocks(pk)?;
        self.tx()
            .execute("DELETE FROM message_route WHERE message_pk=?1", [pk])?;
        self.events_after(id, true)?;
        self.tx().execute("DELETE FROM message WHERE pk=?1", [pk])?;
        self.context.changes.session(self.tx(), session_pk)?;
        Ok(())
    }

    /// Removes a message's blocks and calls; the calls its blocks named and
    /// answered are paired again.
    fn clear_blocks(&mut self, message_pk: i64) -> Result<()> {
        let blocks = self
            .tx()
            .prepare_cached("SELECT session_pk,call_id,text_id FROM block WHERE message_pk=?1")?
            .query_map([message_pk], |r| {
                <(i64, Option<String>, Option<i64>)>::try_from(r)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let old_calls: Vec<(i64, Option<String>, Option<String>)> = self
            .tx()
            .prepare_cached(
                "SELECT t.pk, t.command, t.path FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
                 WHERE b.message_pk = ?1",
            )?
            .query_map([message_pk], |r| <(_, _, _)>::try_from(r))?
            .collect::<rusqlite::Result<_>>()?;
        for (pk, command, path) in old_calls {
            if !self.context.at_build_end {
                self.tx()
                    .prepare_cached(
                        "INSERT INTO tool_call_trigram(tool_call_trigram, rowid, command, path)
                         VALUES ('delete', ?1, ?2, ?3)",
                    )?
                    .execute(params![pk, command, path])?;
            }
            self.tx()
                .prepare_cached("DELETE FROM tool_call WHERE pk = ?1")?
                .execute([pk])?;
        }
        self.tx()
            .prepare_cached("DELETE FROM block WHERE message_pk = ?1")?
            .execute([message_pk])?;
        let mut texts = BTreeSet::new();
        for (session_pk, call_id, text_id) in blocks {
            if let Some(call_id) = call_id {
                self.calls.insert((session_pk, call_id));
            }
            if let Some(text) = text_id {
                texts.insert(text);
            }
        }
        for text in texts {
            crate::text_rows::unindex_unused(
                self.tx(),
                self.context.texts,
                text,
                self.context.at_build_end,
            )?;
        }
        Ok(())
    }

    fn insert_blocks(
        &mut self,
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
        let origin = m.session.origin().as_str().to_owned();
        let at = m.at.map(Timestamp::unix_ms);
        for (idx, ((block, body), data)) in m
            .blocks
            .iter()
            .zip(&prepared.blocks)
            .zip(&prepared.data)
            .enumerate()
        {
            let text_id = prepared.texts[idx]
                .as_ref()
                .map(|hash| self.context.texts.resolve(self.tx(), hash, None, None))
                .transpose()?;
            let bit = search_bit(block);
            if let Some(id) = text_id {
                crate::text_rows::join_search_index(
                    self.tx(),
                    self.context.texts,
                    id,
                    bit,
                    self.context.at_build_end,
                )?;
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
            let call_id = call_id.map(|id| self.ids.call(&origin, id)).transpose()?;
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
            self.tx()
                .prepare_cached(
                    "INSERT INTO block(message_pk, session_pk, idx, kind, text_id, body, call_id,
                         blob_hash, media_type, file_name, size, url, direction, signature, encrypted,
                         other_type, data, ordinal)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
                )?
                .execute(params![
                    message_pk,
                    session_pk,
                    idx as i64,
                    kind,
                    text_id,
                    body,
                    call_id,
                    attachment.and_then(|a| a.blob).map(|h| h.0.to_vec()),
                    attachment.and_then(|a| a.media_type.as_deref()),
                    attachment.and_then(|a| a.name.as_deref()),
                    attachment.and_then(|a| a.size).and_then(|s| i64::try_from(s).ok()),
                    attachment.and_then(|a| a.url.as_deref()),
                    attachment.map(|a| name(&a.direction)),
                    signature,
                    i64::from(encrypted),
                    other_type,
                    data.as_deref(),
                    prepared.block_selection.block_ordinals[idx].to_string(),
                ])?;
            let block_pk = self.tx().last_insert_rowid();
            if let Some(call_id) = &call_id {
                self.calls.insert((session_pk, call_id.clone()));
            }
            if let Block::ToolCall {
                tool,
                action,
                command,
                path,
                run_by_script,
                ..
            } = block
            {
                let tool = tool.as_deref().unwrap_or("unknown");
                self.tx()
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
                        (*action == ToolAction::Shell)
                            .then(|| command.as_deref().and_then(polylogue_model::command::normalize))
                            .flatten()
                    ])?;
                let pk = self.tx().last_insert_rowid();
                if !self.context.at_build_end {
                    self.tx()
                        .prepare_cached(
                            "INSERT INTO tool_call_trigram(rowid, command, path) VALUES (?1, ?2, ?3)",
                        )?
                        .execute(params![pk, command, path])?;
                }
            }
        }
        Ok(())
    }

    /// Pairs each touched call with its result: the first tool_result block
    /// with its call ID in session order. An outcome learned after the
    /// result, for a call run in the background, is the call's outcome.
    fn pair_calls(&mut self, stated: BTreeSet<(String, String)>) -> Result<()> {
        let mut calls = std::mem::take(&mut self.calls);
        for (session, call) in stated {
            let shown = self.ids.session(&session)?;
            if let Some(pk) = self.session_pk(&shown)? {
                calls.insert((pk, call));
            }
        }
        for (session_pk, call_id) in calls {
            let session: Option<String> = self
                .tx()
                .scalar_optional("SELECT id FROM session WHERE pk=?1", [session_pk])?;
            let Some(session) = session else { continue };
            if !self.tx().exists(
                "SELECT EXISTS(SELECT 1 FROM tool_call WHERE session_pk=?1 AND call_id=?2)",
                params![session_pk, call_id],
            )? {
                continue;
            }
            let result: Option<(String, i64, Vec<u8>, Option<i64>)> = self
                .tx()
                .prepare_cached(
                    "SELECT m.id,b.idx,b.body,m.at FROM block b JOIN message m ON m.pk=b.message_pk
                     WHERE b.session_pk=?1 AND b.call_id=?2 AND b.kind='tool_result'
                     ORDER BY m.session_order,m.id,b.idx LIMIT 1",
                )?
                .query_row(params![session_pk, call_id], |r| {
                    <(_, _, _, _)>::try_from(r)
                })
                .optional()?;
            let mut columns = Pairing {
                outcome: "unknown".into(),
                reason: Some("no_result".into()),
                ..Pairing::default()
            };
            if let Some((message, idx, body, at)) = result {
                columns.message = Some(message);
                columns.idx = Some(idx);
                if let Ok(Block::ToolResult {
                    outcome,
                    reported_duration_ms,
                    ..
                }) = unpack_block(&body)
                {
                    let (outcome, reason, exit) = outcome_columns(&outcome);
                    columns.outcome = outcome.to_owned();
                    columns.reason = reason;
                    columns.exit = exit;
                    columns.result_at = at;
                    columns.duration = reported_duration_ms.and_then(|ms| i64::try_from(ms).ok());
                }
            }
            let background = self.background_outcome(&session, &call_id)?;
            if let Some(update) = &background {
                let (outcome, reason, exit) = outcome_columns(&update.outcome);
                columns.outcome = outcome.to_owned();
                columns.reason = reason;
                columns.exit = exit;
                columns.result_at = update.result_at.map(Timestamp::unix_ms);
                columns.duration = update
                    .reported_duration_ms
                    .and_then(|ms| i64::try_from(ms).ok());
            }
            let same = self.tx().exists(
                "SELECT NOT EXISTS(SELECT 1 FROM tool_call WHERE session_pk=?1 AND call_id=?2 AND NOT (
                     result_message_id IS ?3 AND result_idx IS ?4 AND outcome IS ?5 AND outcome_reason IS ?6
                     AND exit_code IS ?7 AND result_at IS ?8 AND reported_duration_ms IS ?9))",
                params![session_pk, call_id, columns.message, columns.idx, columns.outcome, columns.reason, columns.exit, columns.result_at, columns.duration],
            )?;
            if same {
                continue;
            }
            self.context
                .changes
                .call_result(self.tx(), session_pk, &call_id)?;
            self.context.changes.session(self.tx(), session_pk)?;
            self.tx()
                .prepare_cached(
                    "UPDATE tool_call SET result_message_id=?3,result_idx=?4,outcome=?5,outcome_reason=?6,
                         exit_code=?7,result_at=?8,reported_duration_ms=?9
                     WHERE session_pk=?1 AND call_id=?2",
                )?
                .execute(params![session_pk, call_id, columns.message, columns.idx, columns.outcome, columns.reason, columns.exit, columns.result_at, columns.duration])?;
            if background.is_some() {
                // The call's outcome changed outside its messages.
                self.tx()
                    .prepare_cached("UPDATE session_span SET prefix = NULL WHERE session_pk = ?1")?
                    .execute([session_pk])?;
            }
        }
        Ok(())
    }

    /// The background outcome the readings of a session state for a call:
    /// each input's newest chain's last statement, then route precedence,
    /// then the first input ID.
    fn background_outcome(
        &mut self,
        session: &str,
        call: &str,
    ) -> Result<Option<polylogue_model::BackgroundUpdate>> {
        let mut found: Vec<(i64, i64, i64, String, Route, String)> = Vec::new();
        for stated in self.ids.stated(session)? {
            let rows = self
                .tx()
                .prepare_cached(
                    "SELECT o.input_pk,o.chain,o.position,i.id,i.route,o.body FROM call_outcome o
                     JOIN input_ref i ON i.pk=o.input_pk WHERE o.session_id=?1 AND o.call_id=?2",
                )?
                .query_map(params![stated, call], |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.name(4)?,
                        r.get(5)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            found.extend(rows);
        }
        let mut newest = BTreeMap::<i64, i64>::new();
        for (input, chain, ..) in &found {
            let n = newest.entry(*input).or_insert(*chain);
            *n = (*n).max(*chain);
        }
        let best = found
            .into_iter()
            .filter(|(input, chain, ..)| newest.get(input) == Some(chain))
            .min_by(|a, b| {
                (FieldGroup::Content.rank(a.4), &a.3, std::cmp::Reverse(a.2)).cmp(&(
                    FieldGroup::Content.rank(b.4),
                    &b.3,
                    std::cmp::Reverse(b.2),
                ))
            });
        best.map(|(.., body)| Ok(serde_json::from_str(&body)?))
            .transpose()
    }

    /// The ID an event is shown with, and the stated IDs whose readings may
    /// show it.
    fn shown_event_id(&mut self, stated: &str) -> Result<(String, BTreeSet<String>)> {
        let mut ids = BTreeSet::from([stated.to_owned()]);
        let row: Option<(Vec<u8>, Vec<u8>)> = self
            .tx()
            .prepare_cached("SELECT fields,data FROM event_parse WHERE event_id=?1 LIMIT 1")?
            .query_row([stated], |r| <(_, _)>::try_from(r))
            .optional()?;
        let Some((fields, data)) = row else {
            return Ok((stated.to_owned(), ids));
        };
        let mut event: Event = crate::encoding::unpack_json(&fields)?;
        event.data = crate::encoding::unpack_json(&data)?;
        let (_, shown) = self.ids.event(&event, stated)?;
        ids.insert(shown.clone());
        Ok((shown, ids))
    }

    /// Composes one shown event: by route precedence, then value hash,
    /// between each input's newest chain's reading.
    fn compose_event(&mut self, shown: &str, stated: &BTreeSet<String>) -> Result<()> {
        struct Reading {
            input: String,
            chain: i64,
            input_pk: i64,
            route: Route,
            hash: Vec<u8>,
            event: Event,
            seq: i64,
            start: Option<i64>,
            end: Option<i64>,
            fields: Vec<u8>,
            data: Vec<u8>,
        }
        let mut found = Vec::new();
        for id in stated {
            let rows = self
                .tx()
                .prepare_cached(
                    "SELECT i.id,r.chain,r.input_pk,r.route,r.hash,r.fields,r.data,r.seq,r.source_start,r.source_end
                     FROM event_parse r JOIN input_ref i ON i.pk=r.input_pk WHERE r.event_id=?1",
                )?
                .query_map([id], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.name::<Route>(3)?,
                        r.get::<_, Vec<u8>>(4)?,
                        r.get::<_, Vec<u8>>(5)?,
                        r.get::<_, Vec<u8>>(6)?,
                        r.get::<_, i64>(7)?,
                        r.get::<_, Option<i64>>(8)?,
                        r.get::<_, Option<i64>>(9)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            for (input, chain, input_pk, route, hash, fields, data, seq, start, end) in rows {
                let mut event: Event = crate::encoding::unpack_json(&fields)?;
                event.data = crate::encoding::unpack_json(&data)?;
                let (moved, id_shown) = self.ids.event(&event, id)?;
                if id_shown != shown {
                    continue;
                }
                let fields = match &moved {
                    Some(moved) => {
                        let mut fields = moved.clone();
                        fields.data = Json::Null;
                        crate::encoding::pack_json(&fields)?
                    }
                    None => fields,
                };
                found.push(Reading {
                    input,
                    chain,
                    input_pk,
                    route,
                    hash,
                    event: moved.unwrap_or(event),
                    seq,
                    start,
                    end,
                    fields,
                    data,
                });
            }
        }
        let mut newest = BTreeMap::<i64, i64>::new();
        for r in &found {
            let n = newest.entry(r.input_pk).or_insert(r.chain);
            *n = (*n).max(r.chain);
        }
        found.retain(|r| newest.get(&r.input_pk) == Some(&r.chain));
        found.sort_by(|a, b| (&a.input, a.chain).cmp(&(&b.input, b.chain)));
        let stored: Option<(i64, Option<i64>)> = self
            .tx()
            .prepare_cached("SELECT pk,session_pk FROM event WHERE id=?1")?
            .query_row([shown], |r| <(i64, Option<i64>)>::try_from(r))
            .optional()?;
        let winner = compose::value(
            &found
                .iter()
                .map(|r| (r.route, r.hash.as_slice(), r.input.as_str()))
                .collect::<Vec<_>>(),
        );
        let Some(winner) = winner.map(|n| &found[n]) else {
            if let Some((pk, _)) = stored {
                self.context.changes.event(self.tx(), pk)?;
                self.tx().execute("DELETE FROM event WHERE pk=?1", [pk])?;
            }
            return Ok(());
        };
        let session_pk = match &winner.event.session {
            Some(session) => Some(self.session_row(&session.to_string())?),
            None => None,
        };
        let after = winner.event.after.as_ref().map(ToString::to_string);
        let seq = if winner.seq >= 0 {
            winner.seq
        } else {
            match &after {
                Some(after) => self.message_seq(after)?.unwrap_or(0),
                None => 0,
            }
        };
        let values = params![
            shown,
            session_pk,
            seq,
            after,
            winner.event.kind,
            winner.event.at.map(Timestamp::unix_ms),
            winner.input_pk,
            winner.chain,
            winner.start,
            winner.end,
            winner.fields,
            winner.data
        ];
        match stored {
            Some((pk, before_session)) => {
                let same = self.tx().exists(
                    "SELECT EXISTS(SELECT 1 FROM event WHERE id=?1 AND session_pk IS ?2 AND seq IS ?3 AND after_message_id IS ?4
                     AND kind IS ?5 AND at IS ?6 AND input_pk IS ?7 AND source_chain IS ?8 AND source_start IS ?9
                     AND source_end IS ?10 AND fields IS ?11 AND data IS ?12)",
                    values,
                )?;
                if same {
                    return Ok(());
                }
                self.context.changes.event(self.tx(), pk)?;
                if let Some(session) = before_session.filter(|s| Some(*s) != session_pk) {
                    self.context.changes.session(self.tx(), session)?;
                }
                self.tx().prepare_cached(
                    "UPDATE event SET session_pk=?2,seq=?3,after_message_id=?4,kind=?5,at=?6,input_pk=?7,
                         source_chain=?8,source_start=?9,source_end=?10,fields=?11,data=?12 WHERE id=?1",
                )?
                .execute(values)?;
            }
            None => {
                self.tx().prepare_cached(
                    "INSERT INTO event(id,session_pk,seq,after_message_id,kind,at,input_pk,source_chain,source_start,source_end,fields,data)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
                )?
                .execute(values)?;
                let pk = self.tx().last_insert_rowid();
                self.context.changes.new_event(pk);
            }
        }
        if let Some(session) = session_pk {
            self.context.changes.session(self.tx(), session)?;
        }
        Ok(())
    }

    /// The place in its input of the reading a shown message shows.
    fn message_seq(&mut self, message: &str) -> Result<Option<i64>> {
        let main: Option<(i64, i64)> = self
            .tx()
            .prepare_cached("SELECT input_pk,source_chain FROM message WHERE id=?1")?
            .query_row([message], |r| <(i64, i64)>::try_from(r))
            .optional()?;
        let Some((input, chain)) = main else {
            return Ok(None);
        };
        for stated in self.ids.stated_messages(message)? {
            let seq: Option<i64> = self.tx().scalar_optional(
                "SELECT seq FROM message_parse WHERE input_pk=?1 AND chain=?2 AND message_id=?3",
                params![input, chain, stated],
            )?;
            if seq.is_some() {
                return Ok(seq);
            }
        }
        Ok(None)
    }

    /// A stated link's coordinates as shown: its ends and branch point named
    /// as their sessions and messages are shown.
    fn shown_link(&mut self, mut link: Link) -> Result<Link> {
        link.parent = SessionId::parse(&self.ids.session(&link.parent.to_string())?)
            .map_err(StoreError::invalid)?;
        link.child = SessionId::parse(&self.ids.session(&link.child.to_string())?)
            .map_err(StoreError::invalid)?;
        link.branch_point = link
            .branch_point
            .map(|at| {
                MessageId::parse(&self.ids.message(&at.to_string())?).map_err(StoreError::invalid)
            })
            .transpose()?;
        Ok(link)
    }

    fn shown_statement(&mut self, statement: &[u8]) -> Result<Vec<u8>> {
        let body: Option<Vec<u8>> = self.tx().scalar_optional(
            "SELECT body FROM link_parse WHERE statement=?1 LIMIT 1",
            [statement],
        )?;
        match body {
            Some(body) => {
                let link = self.shown_link(serde_json::from_slice(&body)?)?;
                readings::link_statement(&link)
            }
            None => Ok(statement.to_vec()),
        }
    }

    /// Composes one shown link: the preferred reading's input, with the
    /// evidence and copied-prefix observation merged over every reading.
    fn compose_link(&mut self, shown: &[u8], stated: &BTreeSet<Vec<u8>>) -> Result<()> {
        let mut found: Vec<(String, i64, i64, Route, Vec<u8>, Link)> = Vec::new();
        let mut statements = stated.clone();
        statements.insert(shown.to_vec());
        for statement in &statements {
            let rows = self
                .tx()
                .prepare_cached(
                    "SELECT i.id,r.input_pk,r.chain,r.route,r.hash,r.body FROM link_parse r
                     JOIN input_ref i ON i.pk=r.input_pk WHERE r.statement=?1",
                )?
                .query_map([statement], |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.name(3)?,
                        r.get(4)?,
                        r.get::<_, Vec<u8>>(5)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            for (input, input_pk, chain, route, hash, body) in rows {
                let link = self.shown_link(serde_json::from_slice(&body)?)?;
                if readings::link_statement(&link)? == shown {
                    found.push((input, input_pk, chain, route, hash, link));
                }
            }
        }
        let mut newest = BTreeMap::<i64, i64>::new();
        for (_, input, chain, ..) in &found {
            let n = newest.entry(*input).or_insert(*chain);
            *n = (*n).max(*chain);
        }
        found.retain(|(_, input, chain, ..)| newest.get(input) == Some(chain));
        found.sort_by(|a, b| (&a.0, a.2).cmp(&(&b.0, b.2)));
        let (parent, child, kind, how, at, call): (
            String,
            String,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
        ) = serde_json::from_slice(shown)?;
        let stored: Option<i64> = self.tx().scalar_optional(
            "SELECT pk FROM link WHERE parent_id=?1 AND child_id=?2 AND kind=?3 AND how IS ?4 AND branch_point IS ?5 AND spawning_call IS ?6",
            params![parent, child, kind, how, at, call],
        )?;
        let winner = compose::value(
            &found
                .iter()
                .map(|r| (r.3, r.4.as_slice(), r.0.as_str()))
                .collect::<Vec<_>>(),
        );
        let Some(winner) = winner else {
            if let Some(pk) = stored {
                self.context.changes.link(self.tx(), pk)?;
                self.mark_link_sessions(&parent, &child)?;
                self.tx().execute("DELETE FROM link WHERE pk=?1", [pk])?;
            }
            return Ok(());
        };
        let mut link = found[winner].5.clone();
        for other in &found {
            link = readings::merge_link(link, other.5.clone());
        }
        let (input_pk, chain) = (found[winner].1, found[winner].2);
        match stored {
            Some(pk) => {
                let same = self.tx().exists(
                    "SELECT EXISTS(SELECT 1 FROM link WHERE pk=?1 AND evidence IS ?2 AND input_pk IS ?3 AND source_chain IS ?4 AND copies_prefix IS ?5)",
                    params![pk, link.evidence, input_pk, chain, link.copies_prefix],
                )?;
                if same {
                    return Ok(());
                }
                self.context.changes.link(self.tx(), pk)?;
                self.tx()
                    .prepare_cached("UPDATE link SET evidence=?2,input_pk=?3,source_chain=?4,copies_prefix=?5 WHERE pk=?1")?
                    .execute(params![pk, link.evidence, input_pk, chain, link.copies_prefix])?;
            }
            None => {
                self.context.changes.link_child(self.tx(), &child)?;
                self.tx().prepare_cached(
                    "INSERT INTO link(parent_id,child_id,kind,how,evidence,input_pk,source_chain,branch_point,spawning_call,copies_prefix)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                )?
                .execute(params![parent, child, kind, how, link.evidence, input_pk, chain, at, call, link.copies_prefix])?;
            }
        }
        self.mark_link_sessions(&parent, &child)
    }

    fn mark_link_sessions(&mut self, parent: &str, child: &str) -> Result<()> {
        for session in [parent, child] {
            if let Some(pk) = self.session_pk(session)? {
                self.context.changes.session(self.tx(), pk)?;
            }
        }
        Ok(())
    }

    /// Composes one shown document: the newest source time, then route
    /// precedence, then the content hash.
    fn compose_document(&mut self, id: &str) -> Result<()> {
        let mut found = self
            .tx()
            .prepare_cached(
                "SELECT i.id,r.input_pk,r.chain,r.route,r.source_time,r.hash,r.source_start,r.source_end
                 FROM document_parse r JOIN input_ref i ON i.pk=r.input_pk WHERE r.document_id=?1",
            )?
            .query_map([id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.name::<Route>(3)?,
                    r.get::<_, Option<i64>>(4)?,
                    r.get::<_, Vec<u8>>(5)?,
                    r.get::<_, i64>(6)?,
                    r.get::<_, i64>(7)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut newest = BTreeMap::<i64, i64>::new();
        for r in &found {
            let n = newest.entry(r.1).or_insert(r.2);
            *n = (*n).max(r.2);
        }
        found.retain(|r| newest.get(&r.1) == Some(&r.2));
        let stored: Option<(i64, Vec<u8>)> = self
            .tx()
            .prepare_cached("SELECT pk,version_hash FROM document WHERE id=?1")?
            .query_row([id], |r| <(i64, Vec<u8>)>::try_from(r))
            .optional()?;
        let winner = compose::document(
            &found
                .iter()
                .map(|r| compose::DocumentReading {
                    input: &r.0,
                    time: r.4.and_then(Timestamp::from_unix_ms),
                    route: r.3,
                    hash: &r.5,
                })
                .collect::<Vec<_>>(),
        );
        let Some(winner) = winner.map(|n| &found[n]) else {
            if let Some((pk, _)) = stored {
                let old = self.document_texts(pk)?;
                self.tx()
                    .execute("DELETE FROM document_block WHERE document_pk=?1", [pk])?;
                self.tx()
                    .execute("DELETE FROM document WHERE pk=?1", [pk])?;
                for text in old {
                    crate::text_rows::unindex_unused(
                        self.tx(),
                        self.context.texts,
                        text,
                        self.context.at_build_end,
                    )?;
                }
            }
            return Ok(());
        };
        let (input_pk, chain, start, end) = (winner.1, winner.2, winner.6, winner.7);
        if let Some((pk, hash)) = &stored
            && *hash == winner.5
        {
            self.tx()
                .prepare_cached(
                    "UPDATE document SET input_pk=?2,source_chain=?3,source_start=?4,source_end=?5,changed=?6
                     WHERE pk=?1 AND NOT (input_pk IS ?2 AND source_chain IS ?3 AND source_start IS ?4 AND source_end IS ?5)",
                )?
                .execute(params![pk, input_pk, chain, start, end, self.context.counter])?;
            return Ok(());
        }
        let document: Document = readings::read_document(self.tx(), input_pk, chain, id)?
            .ok_or_else(|| StoreError::Invalid("a document reading has no body".into()))?;
        let prepared = PreparedDocument::new(&document)?;
        let pk = match stored {
            Some((pk, _)) => {
                self.tx().execute(
                    "UPDATE document SET purpose=?2,title=?3,source_time=?4,authorship=?5,version_hash=?6,input_pk=?7,
                         source_chain=?8,source_start=?9,source_end=?10,changed=?11,fields=?12 WHERE pk=?1",
                    params![pk, document.purpose.as_str(), document.title, document.source_time.map(Timestamp::unix_ms),
                        document.authorship.as_str(), &prepared.hash[..], input_pk, chain, start, end,
                        self.context.counter, prepared.fields],
                )?;
                pk
            }
            None => {
                self.tx().execute(
                    "INSERT INTO document(id,origin,native_id,purpose,title,source_time,authorship,version_hash,input_pk,
                         source_chain,source_start,source_end,fields,changed)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
                    params![id, document.id.origin().as_str(), document.id.native().as_str(), document.purpose.as_str(),
                        document.title, document.source_time.map(Timestamp::unix_ms), document.authorship.as_str(),
                        &prepared.hash[..], input_pk, chain, start, end, prepared.fields, self.context.counter],
                )?;
                self.tx().last_insert_rowid()
            }
        };
        let old = self.document_texts(pk)?;
        self.tx()
            .execute("DELETE FROM document_block WHERE document_pk=?1", [pk])?;
        let text_ids = readings::text_ids(self.tx(), input_pk, chain, true, id)?;
        for (idx, ((block, body), data)) in document
            .blocks
            .iter()
            .zip(&prepared.blocks)
            .zip(&prepared.data)
            .enumerate()
        {
            let text = prepared.texts[idx]
                .as_ref()
                .map(|_| {
                    text_ids.get(&idx).copied().ok_or_else(|| {
                        StoreError::Invalid("a document reading's text reference is missing".into())
                    })
                })
                .transpose()?;
            if let Some(text) = text {
                crate::text_rows::join_search_index(
                    self.tx(),
                    self.context.texts,
                    text,
                    1,
                    self.context.at_build_end,
                )?;
            }
            let blob = match block {
                Block::Image(attachment) | Block::File(attachment) => {
                    attachment.blob.map(|h| h.0.to_vec())
                }
                _ => None,
            };
            self.tx().execute(
                "INSERT INTO document_block(document_pk,idx,kind,body,text_id,blob_hash,data) VALUES (?1,?2,?3,?4,?5,?6,?7)",
                params![pk, idx as i64, block.kind().as_str(), body, text, blob, data],
            )?;
        }
        for text in old {
            crate::text_rows::unindex_unused(
                self.tx(),
                self.context.texts,
                text,
                self.context.at_build_end,
            )?;
        }
        Ok(())
    }

    fn document_texts(&self, pk: i64) -> Result<Vec<i64>> {
        Ok(self
            .tx()
            .prepare_cached("SELECT DISTINCT text_id FROM document_block WHERE document_pk=?1 AND text_id IS NOT NULL")?
            .query_map([pk], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }
}

/// What a call's pairing with its result sets on its row.
#[derive(Default)]
struct Pairing {
    message: Option<String>,
    idx: Option<i64>,
    outcome: String,
    reason: Option<String>,
    exit: Option<i64>,
    result_at: Option<i64>,
    duration: Option<i64>,
}

/// The main reading of a shown message and what it is shown with.
struct MainReading<'a> {
    session_pk: i64,
    order: &'a str,
    input_pk: i64,
    chain: i64,
    start: i64,
    end: i64,
    in_progress: bool,
    time: Vec<u8>,
    version: &'a [u8],
    layout: Option<&'a [u8]>,
}

/// The message columns composition writes. The first [`CONTENT_COLUMNS`]
/// are its content and session; the rest are where its main reading is.
const MESSAGE_COLUMNS: &[&str] = &[
    "session_pk",
    "version_hash",
    "layout",
    "role",
    "written_by",
    "prompt_kind",
    "model",
    "at",
    "at_time",
    "parent_id",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cache_write_5m_tokens",
    "cache_write_1h_tokens",
    "reasoning_tokens",
    "speed",
    "service_tier",
    "provider_cost_nusd",
    "cost_nusd",
    "stop_reason",
    "duration_ms",
    "words",
    "attrs_json",
    "session_order",
    "input_pk",
    "source_chain",
    "source_start",
    "source_end",
    "in_progress",
];
const CONTENT_COLUMNS: usize = 25;
const IN_PROGRESS: usize = 30;

/// A block layout worth storing: `None` for a message holding every block.
fn layout_column(selection: &polylogue_model::bundle::ShownBlocks) -> Result<Option<Vec<u8>>> {
    let dense = selection.block_count == selection.block_ordinals.len() as u64
        && selection
            .block_ordinals
            .iter()
            .enumerate()
            .all(|(n, ordinal)| *ordinal == n as u64);
    if dense {
        Ok(None)
    } else {
        Ok(Some(crate::encoding::message_layout(selection)?))
    }
}

fn message_columns(m: &Message, main: MainReading<'_>) -> Result<Vec<Value>> {
    let author = polylogue_model::WrittenBy::of(m);
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
    // A count beyond SQLite's integer range is unknown, never wrapped.
    let n = |x: Option<u64>| x.and_then(|x| i64::try_from(x).ok());
    let int = |x: Option<i64>| x.map_or(Value::Null, Value::Integer);
    let text = |x: Option<String>| x.map_or(Value::Null, Value::Text);
    Ok(vec![
        Value::Integer(main.session_pk),
        Value::Blob(main.version.to_vec()),
        main.layout.map_or(Value::Null, |l| Value::Blob(l.to_vec())),
        Value::Text(name(&m.role)),
        Value::Text(author.as_str().to_owned()),
        text(m.prompt_kind.map(|k| name(&k))),
        text(m.model.clone()),
        int(m.at.map(Timestamp::unix_ms)),
        Value::Blob(main.time),
        text(m.parent.as_ref().map(ToString::to_string)),
        int(n(u.input)),
        int(n(u.output)),
        int(n(u.cached_input)),
        int(n(u.cache_write)),
        // The five-minute part is known only when the one-hour part is.
        int(u
            .cache_write_1h
            .and(u.cache_write)
            .and_then(|w| i64::try_from(w.saturating_sub(u.cache_write_1h.unwrap_or(0))).ok())),
        int(n(u.cache_write_1h)),
        int(n(u.reasoning)),
        text(u.speed.clone()),
        text(u.service_tier.clone()),
        int(reported_nusd(m.reported_cost_usd.as_ref())),
        // The cost derivation's value, written with the row so a first
        // build's derive does not rewrite every priced message.
        int(polylogue_model::Catalog::builtin().cost_nusd(m.model.as_deref(), u)),
        text(m.stop_reason.clone()),
        int(n(m.duration_ms)),
        Value::Integer(words),
        text(attrs),
        Value::Text(main.order.to_owned()),
        Value::Integer(main.input_pk),
        Value::Integer(main.chain),
        Value::Integer(main.start),
        Value::Integer(main.end),
        Value::Integer(i64::from(main.in_progress)),
    ])
}

/// The search index a block's text belongs to: 2 for tool results and tool
/// calls (the `output:` field, which includes written file bodies), 1 for prose.
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

pub(super) fn name<T: serde::Serialize>(v: &T) -> String {
    match serde_json::to_value(v) {
        Ok(serde_json::Value::String(s)) => s,
        _ => String::new(),
    }
}

/// A provider's reported amount is an observed fact, separate from catalog
/// pricing. The number keeps its exact spelling (`arbitrary_precision`), so
/// it converts to nano-USD without passing through a float.
fn reported_nusd(usd: Option<&serde_json::Number>) -> Option<i64> {
    polylogue_model::money::nusd_from_usd(&usd?.to_string())
}
