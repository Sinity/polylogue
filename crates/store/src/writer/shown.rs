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
use rusqlite::types::{ToSqlOutput, Value, ValueRef};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// What composition writes with.
pub(super) struct Context<'a> {
    pub tx: &'a Transaction<'a>,
    pub counter: i64,
    pub changes: &'a ChangeSummary,
}

/// One chain's reading of a message.
#[derive(Debug)]
pub(super) struct MessageReading {
    pub input_pk: i64,
    pub input: String,
    pub chain: i64,
    /// The message and session as the chain names them.
    pub message: String,
    pub session: String,
    pub start: i64,
    pub end: i64,
    pub seq: i64,
    pub position: i64,
    pub order_time: Option<i64>,
    pub route: Route,
    pub completion: Completion,
    pub content: bool,
    pub hash: Vec<u8>,
    /// Where the reading's content is stored.
    pub stored: crate::content::Stretch,
    /// What the reading brings with it and its session's kind, when they
    /// come with it (a build's stream); otherwise they are read from the
    /// index.
    pub given: Option<Given>,
}

/// What a reading supplied with it brings: the kind its chain states for
/// its session, and what composition takes from its value besides its
/// stored content.
#[derive(Debug)]
pub(super) struct Given {
    pub kind: Option<polylogue_model::SessionKind>,
    /// A newer chain of its input was read to its end or waits for more.
    pub superseded: bool,
    pub words: i64,
    /// Each block's text hash and text row.
    pub texts: Vec<Option<[u8; 32]>>,
    pub text_ids: Vec<Option<i64>>,
}

/// What composition takes from a reading's value: its message and where its
/// blocks' pieces are stored. The message's texts are needed only for its
/// word count, so a value given it may carry the message without them.
#[derive(Debug)]
pub(super) struct ReadingValue {
    pub message: Message,
    pub words: i64,
    pub selection: polylogue_model::bundle::ShownBlocks,
    /// Each block's piece, in the content of a reading that holds it.
    pub pieces: Vec<crate::content::Piece>,
    /// Each block's text hash, for a block that keeps its text in `text`.
    pub texts: Vec<Option<[u8; 32]>>,
    /// Each block's text row, when the reading came with them; otherwise
    /// each text is found by its hash.
    pub text_ids: Option<Vec<Option<i64>>>,
}

impl ReadingValue {
    /// The value of a message as its readings hold it, with its texts.
    pub(super) fn of(stored: readings::StoredReading) -> Result<Self> {
        let readings::StoredReading {
            message,
            selection,
            pieces,
        } = stored;
        selection.validate(message.blocks.len())?;
        Ok(Self {
            words: text_words(&message),
            texts: crate::encoding::text_hashes(&message.blocks),
            message,
            selection,
            pieces,
            text_ids: None,
        })
    }
}

/// The words of a message's text blocks.
pub(super) fn text_words(message: &Message) -> i64 {
    message
        .blocks
        .iter()
        .filter_map(Block::text)
        .map(|t| t.split_whitespace().count() as i64)
        .sum()
}

/// Writes the shown rows of everything `touched` names; returns the number
/// of message rows written, changed or removed.
pub(super) fn compose(context: &Context<'_>, touched: Touched) -> Result<u64> {
    let mut work = Work::new(context, Identities::new(context.tx), None)?;
    work.run(touched)?;
    Ok(work.written)
}

/// Each input's reading of a shown message: that of the newest of its
/// chains holding it, without a content-named message an input's newer
/// chain, read to its end, no longer holds (`gone` says whether it does).
fn latest_readings(
    mut all: Vec<MessageReading>,
    mut gone: impl FnMut(&MessageReading) -> Result<bool>,
) -> Result<Vec<MessageReading>> {
    let mut newest = BTreeMap::<i64, i64>::new();
    for r in &all {
        let chain = newest.entry(r.input_pk).or_insert(r.chain);
        *chain = (*chain).max(r.chain);
    }
    all.retain(|r| newest.get(&r.input_pk) == Some(&r.chain));
    let mut kept = Vec::with_capacity(all.len());
    for r in all {
        if r.content && gone(&r)? {
            continue;
        }
        kept.push(r);
    }
    // A stable order, so that ties in composition fall the same way.
    kept.sort_by(|a, b| (&a.input, a.chain).cmp(&(&b.input, b.chain)));
    Ok(kept)
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
    reader: crate::content::Reader,
    /// Set when the readings come grouped by session from a build's stream.
    stream: Option<Stream>,
}

/// Sessions composed from readings supplied grouped by session, as a build
/// supplies them (docs/architecture.md, one path). Every reading of what a
/// session shows comes with the session, and nothing is stored for it
/// before: no stored row is read or compared, and calls are paired from the
/// session's own blocks before their rows are written.
#[derive(Default)]
struct Stream {
    /// The readings of the message being composed.
    pending: Vec<MessageReading>,
    /// The session being composed: whether its readings state its facts or
    /// a message, and its event readings and background outcomes.
    own: bool,
    events: Vec<EventRow>,
    /// Where each event ID's readings are in `events`, so that composing an
    /// event finds its readings without passing over the session's others.
    event_readings: HashMap<String, Vec<usize>>,
    outcomes: Vec<OutcomeRow>,
    /// Each composed message's place in its main reading's input, for the
    /// events that follow it.
    seqs: HashMap<String, i64>,
    /// The session's tool calls, written once paired.
    calls: Vec<Call>,
    /// The first result of each call ID in session order.
    results: HashMap<String, CallResult>,
    /// Events whose place comes from a message another session shows, and
    /// events several sessions state: composed from the index once every
    /// session is written.
    deferred: Vec<(Option<String>, String)>,
    /// The event IDs several sessions state.
    shared_events: Arc<std::collections::HashSet<String>>,
}

/// A tool call row waiting for its pairing.
struct Call {
    session_pk: i64,
    block_pk: i64,
    call_id: Option<String>,
    at: Option<i64>,
    tool: String,
    action: &'static str,
    command: Option<String>,
    path: Option<String>,
    run_by_script: bool,
    command_norm: Option<String>,
}

/// A tool result block: (session order, message ID, block index) places it.
struct CallResult {
    place: (Vec<u8>, String, i64),
    outcome: Option<ResultOutcome>,
    at: Option<i64>,
}

/// One chain's reading of an event.
#[derive(Debug, Clone)]
pub(super) struct EventRow {
    pub input: String,
    pub input_pk: i64,
    pub chain: i64,
    pub event_id: String,
    pub route: Route,
    pub hash: Vec<u8>,
    /// Its stored content: its fields, then its data.
    pub piece: crate::content::Piece,
    pub seq: i64,
    pub start: Option<i64>,
    pub end: Option<i64>,
}

/// One chain's statement of a call's background outcome.
#[derive(Debug)]
pub(super) struct OutcomeRow {
    pub input_pk: i64,
    pub chain: i64,
    pub position: i64,
    pub input: String,
    pub route: Route,
    pub call_id: String,
    pub body: String,
}

/// Composes sessions whose readings a build supplies grouped by session.
/// A session is begun, given its readings (facts, then each message's
/// readings, events and outcomes), and ended; the transaction's sessions are
/// then finished by [`super::rows::finish_summaries`].
pub(super) struct Composer<'a, 'c>(Work<'a, 'c>);

impl<'a, 'c> Composer<'a, 'c> {
    pub(super) fn new(
        context: &'a Context<'c>,
        aliases: Arc<super::identity::Aliases>,
        shared_events: Arc<std::collections::HashSet<String>>,
    ) -> Result<Self> {
        Ok(Self(Work::new(
            context,
            Identities::loaded(aliases),
            Some(Stream {
                shared_events,
                ..Stream::default()
            }),
        )?))
    }

    fn stream(&mut self) -> &mut Stream {
        self.0.stream.as_mut().expect("a composer reads a stream")
    }

    /// Starts a session.
    pub(super) fn begin(&mut self) {
        let stream = self.stream();
        stream.own = false;
        stream.events.clear();
        stream.event_readings.clear();
        stream.outcomes.clear();
        stream.seqs.clear();
        stream.calls.clear();
        stream.results.clear();
    }

    /// The session's readings state its facts.
    pub(super) fn facts(&mut self) {
        self.stream().own = true;
    }

    /// Composes the message the session's readings name `id`.
    pub(super) fn message(&mut self, id: &str, readings: Vec<MessageReading>) -> Result<()> {
        let stream = self.stream();
        stream.own = true;
        stream.pending = readings;
        self.0.compose_message(id)?;
        Ok(())
    }

    pub(super) fn event(&mut self, event: EventRow) {
        self.stream().events.push(event);
    }

    pub(super) fn outcome(&mut self, outcome: OutcomeRow) {
        self.stream().outcomes.push(outcome);
    }

    /// Ends the session `session`: pairs its calls, composes its events and
    /// settles its row.
    pub(super) fn end(&mut self, session: &str) -> Result<()> {
        let work = &mut self.0;
        work.pair_stream_calls(session)?;
        let events = std::mem::take(&mut work.stream().events);
        let mut shown_events = BTreeMap::<String, BTreeSet<String>>::new();
        for event in &events {
            let stream = work.stream();
            if stream.shared_events.contains(&event.event_id) {
                // This session holds only some of its readings.
                stream
                    .deferred
                    .push((Some(session.to_owned()), event.event_id.clone()));
                continue;
            }
            let (shown, stated) = work.shown_event_id(&event.event_id)?;
            shown_events.entry(shown).or_default().extend(stated);
        }
        let stream = work.stream();
        stream.event_readings.clear();
        for (n, event) in events.iter().enumerate() {
            stream
                .event_readings
                .entry(event.event_id.clone())
                .or_default()
                .push(n);
        }
        stream.events = events;
        for (shown, stated) in shown_events {
            work.compose_event(&shown, &stated)?;
        }
        work.settle_session(session)?;
        Ok(())
    }

    /// The messages written, and the events left to compose from the
    /// index.
    pub(super) fn finish(mut self) -> (u64, Vec<(Option<String>, String)>) {
        let deferred = std::mem::take(&mut self.stream().deferred);
        (self.0.written, deferred)
    }
}

impl<'a, 'c> Work<'a, 'c> {
    fn new(context: &'a Context<'c>, ids: Identities<'c>, stream: Option<Stream>) -> Result<Self> {
        Ok(Self {
            context,
            ids,
            sessions: BTreeMap::new(),
            calls: BTreeSet::new(),
            events: BTreeSet::new(),
            written: 0,
            reader: crate::content::Reader::default(),
            stream,
        })
    }

    fn stream(&mut self) -> &mut Stream {
        self.stream.as_mut().expect("a composer reads a stream")
    }
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
            released_groups,
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
                    "SELECT parent_id,child_id,kind,reason,branch_point,spawning_call FROM link_statement WHERE parent_id=?1
                     UNION SELECT parent_id,child_id,kind,reason,branch_point,spawning_call FROM link_statement WHERE child_id=?1",
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
        let mut relations = BTreeSet::new();
        for (shown, stated) in shown_links {
            self.compose_link(&shown, &stated)?;
            let (parent, child, kind, how, ..): (
                String,
                String,
                String,
                Option<String>,
                Option<String>,
                Option<String>,
            ) = serde_json::from_slice(&shown)?;
            relations.insert((parent, child, kind, how));
        }
        for relation in relations {
            self.resolve_links(relation)?;
        }
        for document in documents {
            self.compose_document(&document)?;
        }
        for session in session_work {
            self.settle_session(&session)?;
        }
        readings::remove_released_texts(self.tx(), &released_texts)?;
        readings::remove_released_groups(self.tx(), &released_groups)?;
        Ok(())
    }

    /// The session row of a shown session, when it has one.
    fn session_pk(&mut self, session: &str) -> Result<Option<i64>> {
        if let Some(pk) = self.sessions.get(session) {
            return Ok(Some(*pk));
        }
        // A session composed from a stream has a row once this work made it.
        if self.stream.is_some() {
            return Ok(None);
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
        self.tx().prepare_cached("INSERT INTO session(id,origin,native_id,kind,cwd_json,changed,harness,has_transcript,
            message_count,human_prompt_count,human_command_count,word_count,human_word_count,model_word_count,
            thinking_count,compaction_count,tool_call_count,error_count)
            VALUES (?1,?2,?3,'main','[]',?4,?2,0,0,0,0,0,0,0,0,0,0,0)")?
            .execute(params![session, id.origin().as_str(), id.native().as_str(), self.context.counter])?;
        let pk = self.tx().last_insert_rowid();
        self.sessions.insert(session.to_owned(), pk);
        crate::lineage::mark_due(self.tx(), session)?;
        Ok(pk)
    }

    /// A session exists while some reading states it, under its own ID or
    /// a copy's; it is a stub while only events name it. Its own columns
    /// are derived from its readings' facts (derive/session.rs).
    fn settle_session(&mut self, session: &str) -> Result<()> {
        if let Some(stream) = &self.stream {
            let own = stream.own;
            if !own && stream.events.is_empty() {
                return Ok(());
            }
            let pk = self.session_row(session)?;
            self.context.changes.whole_session(pk);
            self.context.changes.session(pk);
            self.set_has_transcript(pk, session, own)?;
            return Ok(());
        }
        let stated = self.ids.stated(session)?;
        // A session is a stub while only events name it: no reading states
        // its facts or a message of it.
        let mut own = false;
        let mut stated_at_all = false;
        for id in &stated {
            own |= self.tx().exists(
                "SELECT EXISTS(SELECT 1 FROM session_parse WHERE session_id=?1)
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
                crate::lineage::mark_due(self.tx(), session)
            }
            (_, true) => {
                let pk = self.session_row(session)?;
                self.context.changes.session(pk);
                self.set_has_transcript(pk, session, own)
            }
        }
    }

    /// Records whether session `pk` has a transcript (it is a stub while only
    /// events name it); a change marks its lineage due.
    fn set_has_transcript(&mut self, pk: i64, session: &str, has_transcript: bool) -> Result<()> {
        let changed = self
            .tx()
            .prepare_cached(
                "UPDATE session SET has_transcript=?2 WHERE pk=?1 AND has_transcript IS NOT ?2",
            )?
            .execute(params![pk, has_transcript])?;
        if changed > 0 {
            crate::lineage::mark_due(self.tx(), session)?;
        }
        Ok(())
    }

    /// Each input's reading of a shown message (see [`latest_readings`]).
    fn shown_message_parses(&mut self, shown: &str) -> Result<Vec<MessageReading>> {
        if let Some(stream) = &mut self.stream {
            return latest_readings(std::mem::take(&mut stream.pending), |r| {
                Ok(r.given.as_ref().is_some_and(|given| given.superseded))
            });
        }
        let mut all = Vec::new();
        for stated in self.ids.stated_messages(shown)? {
            let rows = self
                .tx()
                .prepare_cached(
                    "SELECT r.input_pk,i.id,r.chain,r.message_id,r.session_id,r.source_start,r.source_start+r.source_length,r.position,
                         r.record_position,r.order_time,r.route,r.state,r.id_kind='content',r.hash,
                         r.content_group,r.content_start,r.content_length
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
                        stored: crate::content::Stretch {
                            group: r.get(14)?,
                            start: r.get(15)?,
                            length: r.get(16)?,
                        },
                        given: None,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            for row in rows {
                if self.ids.message(&row.message)? == shown {
                    all.push(row);
                }
            }
        }
        let tx = self.tx();
        latest_readings(all, |r| {
            tx.exists(
                "SELECT EXISTS(SELECT 1 FROM parse_state WHERE input_pk=?1 AND chain>?2 AND state='parsed')",
                params![r.input_pk, r.chain],
            )
        })
    }

    /// The kind its own chain states for a reading's session.
    fn reading_kind(&self, r: &MessageReading) -> Result<Option<polylogue_model::SessionKind>> {
        if let Some(given) = &r.given {
            return Ok(given.kind);
        }
        let kind: Option<Option<String>> = self
            .tx()
            .prepare_cached(
                "SELECT json_extract(facts,'$.kind') FROM session_parse WHERE input_pk=?1 AND chain=?2 AND session_id=?3",
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
    fn value(&mut self, r: &mut MessageReading) -> Result<ReadingValue> {
        if let Some(given) = r.given.take() {
            // The content without texts or structured values: composition
            // writes neither, and the words came with the reading.
            let (message, pieces) = crate::encoding::read_message_content(
                self.reader.bytes(self.context.tx, r.stored)?,
                false,
            )?;
            let selection =
                crate::encoding::message_block_selection(&r.hash, None, message.blocks.len())?;
            if given.texts.len() != message.blocks.len()
                || given
                    .text_ids
                    .iter()
                    .zip(&given.texts)
                    .any(|(id, text)| id.is_some() != text.is_some())
            {
                return Err(StoreError::Invalid(
                    "a message reading's text rows differ from its blocks".into(),
                ));
            }
            return Ok(ReadingValue {
                message,
                words: given.words,
                selection,
                pieces: readings::placed(r.stored, &pieces),
                texts: given.texts,
                text_ids: Some(given.text_ids),
            });
        }
        let stored = readings::read_reading(
            self.context.tx,
            &mut self.reader,
            r.input_pk,
            r.chain,
            &r.message,
        )?
        .ok_or_else(|| StoreError::Invalid("a message reading has no body".into()))?;
        let stored = if stored.selection.block_count == stored.message.blocks.len() as u64 {
            stored
        } else {
            let shown = self.ids.message(&r.message)?;
            let ids = self.ids.stated_messages(&shown)?;
            match readings::read_version_reading(self.context.tx, &mut self.reader, &ids, &r.hash)?
            {
                Some(merged) => merged,
                None => stored,
            }
        };
        ReadingValue::of(stored)
    }

    /// Composes one shown message from its readings and writes its rows;
    /// returns the session it is shown in, if any.
    fn compose_message(&mut self, shown: &str) -> Result<Option<String>> {
        let mut readings = self.shown_message_parses(shown)?;
        let stored: Option<(i64, i64)> = if self.stream.is_some() {
            None
        } else {
            self.tx()
                .prepare_cached("SELECT pk,session_pk FROM message WHERE id=?1")?
                .query_row([shown], |r| <(i64, i64)>::try_from(r))
                .optional()?
        };
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
        let (main, groups, mut value) = if readings.len() == 1 {
            (0, Vec::new(), self.value(&mut readings[0])?)
        } else {
            let mut kinds = Vec::with_capacity(readings.len());
            for r in &readings {
                kinds.push(self.reading_kind(r)?);
            }
            let mut values = Vec::with_capacity(readings.len());
            for r in &mut readings {
                values.push(Some(self.value(r)?));
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
                        message: &value.as_ref().expect("loaded").message,
                    })
                    .collect();
                compose::message(&inputs, |n| {
                    Ok(kinds[n]
                        .is_some_and(|kind| kind != polylogue_model::SessionKind::Compaction))
                })?
                .expect("readings are not empty")
            };
            // The content and its word count are the main reading's.
            let mut value = values[composed.main].take().expect("loaded");
            value.message = composed.message;
            (composed.main, composed.groups, value)
        };
        let r = &readings[main];
        let session = self.ids.session(&r.session)?;
        let session_pk = self.session_row(&session)?;
        // A copy's message is shown in its session, under its name there.
        value.message.id = MessageId::parse(shown).map_err(StoreError::invalid)?;
        value.message.session = SessionId::parse(&session).map_err(StoreError::invalid)?;
        if let Some(parent) = &value.message.parent {
            let renamed = self.ids.message(&parent.to_string())?;
            value.message.parent = Some(MessageId::parse(&renamed).map_err(StoreError::invalid)?);
        }
        let in_progress = r.completion == Completion::InProgress
            && !groups.iter().any(|(group, _)| *group == FieldGroup::Stop);
        // The earliest place among the readings that know an order time, so
        // the message stays before what follows it in each input; a reading
        // that knows none never pulls it to the session's start.
        let order = readings
            .iter()
            .map(|m| {
                (
                    m.order_time.is_none(),
                    crate::encoding::session_order(m.order_time, &m.input, m.seq, m.position),
                )
            })
            .min()
            .map(|(_, order)| order)
            .expect("readings are not empty");
        let mut routes: Vec<(FieldGroup, i64, i64)> = groups
            .iter()
            .map(|(group, n)| (*group, readings[*n].input_pk, readings[*n].chain))
            .collect();
        routes.sort();
        let layout = layout_column(&value.selection)?;
        let message = &value.message;
        let row = message_columns(
            message,
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
                words: value.words,
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
                    // The same content may be stored again elsewhere (a
                    // reading written anew): its blocks name it there.
                    self.move_blocks(pk, &value.pieces)?;
                    self.context.changes.message(self.tx(), pk)?;
                    if before[CONTENT_COLUMNS..] != row[CONTENT_COLUMNS..] {
                        self.update_message(pk, &row)?;
                        self.events_after(shown, false)?;
                    }
                    self.context.changes.session(session_pk);
                    return Ok(Some(session));
                }
                self.context.changes.message(self.tx(), pk)?;
                self.clear_blocks(pk)?;
                self.update_message(pk, &row)?;
                self.tx()
                    .prepare_cached("DELETE FROM message_route WHERE message_pk=?1")?
                    .execute([pk])?;
                if before_session != session_pk {
                    self.context.changes.session(before_session);
                }
                self.events_after(shown, true)?;
                pk
            }
            _ => {
                self.tx()
                    .prepare_cached(&INSERT_MESSAGE)?
                    .execute(rusqlite::params_from_iter(
                        std::iter::once(ToSqlOutput::Borrowed(ValueRef::Text(shown.as_bytes())))
                            .chain(row.iter().map(|v| ToSqlOutput::Borrowed(ValueRef::from(v)))),
                    ))?;
                let pk = self.tx().last_insert_rowid();
                self.context.changes.new_message(pk);
                if self.stream.is_none() {
                    self.events_after(shown, true)?;
                }
                pk
            }
        };
        self.context.changes.session(session_pk);
        self.insert_blocks(pk, session_pk, &order, &value)?;
        for (group, input, chain) in &routes {
            self.tx()
                .prepare_cached(
                    "INSERT INTO message_route(message_pk,field_group,input_pk,chain) VALUES (?1,?2,?3,?4)",
                )?
                .execute(params![pk, group.as_str(), input, chain])?;
        }
        if let Some(stream) = &mut self.stream {
            stream.seqs.insert(shown.to_owned(), r.seq);
        }
        self.written += 1;
        Ok(Some(session))
    }

    /// A message shown whole from one reading is unchanged when its row
    /// already shows that reading, in its session and place, and its blocks
    /// name that reading's stored content: nothing is read or written for it.
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
        // A reading holding some blocks shows blocks other readings hold.
        if selection.block_count != selection.block_ordinals.len() as u64 {
            return Ok(false);
        }
        let layout = layout_column(&selection)?;
        let order = crate::encoding::session_order(r.order_time, &r.input, r.seq, r.position);
        let in_progress = r.completion == Completion::InProgress;
        let same = self.tx().exists(
            "SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1 AND version_hash IS ?2 AND layout IS ?3
                 AND input_pk=?4 AND source_chain=?5 AND source_start=?6 AND source_end=?7
                 AND session_order=?8 AND in_progress=?9)
             AND NOT EXISTS(SELECT 1 FROM message_route WHERE message_pk=?1)
             AND NOT EXISTS(SELECT 1 FROM block WHERE message_pk=?1 AND (content_group IS NOT ?10
                 OR content_start < ?11 OR content_start >= ?11 + ?12))",
            params![
                pk,
                r.hash,
                layout,
                r.input_pk,
                r.chain,
                r.start,
                r.end,
                order,
                in_progress,
                r.stored.group,
                r.stored.start,
                r.stored.length
            ],
        )?;
        if same {
            self.context.changes.message(self.tx(), pk)?;
            self.context.changes.session(session_pk);
        }
        Ok(same)
    }

    /// Points a message's blocks at the pieces given, where they differ.
    fn move_blocks(&self, message_pk: i64, pieces: &[crate::content::Piece]) -> Result<()> {
        let stored = self
            .tx()
            .prepare_cached(
                "SELECT position,content_group,content_start,content_length,data_length FROM block
                 WHERE message_pk=?1 ORDER BY position",
            )?
            .query_map([message_pk], |r| {
                Ok((r.get::<_, i64>(0)?, crate::content::Piece::of(r, 1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for (idx, before) in stored {
            let piece = usize::try_from(idx)
                .ok()
                .and_then(|idx| pieces.get(idx))
                .ok_or_else(|| StoreError::Invalid("a shown block has no piece".into()))?;
            if *piece != before {
                self.tx()
                    .prepare_cached(
                        "UPDATE block SET content_group=?3,content_start=?4,content_length=?5,data_length=?6
                         WHERE message_pk=?1 AND position=?2",
                    )?
                    .execute(params![
                        message_pk,
                        idx,
                        piece.group,
                        piece.start,
                        piece.length,
                        piece.data_length
                    ])?;
            }
        }
        Ok(())
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
    /// transcript changes, and those that take their place in the input
    /// from the message's reading are composed again.
    fn events_after(&mut self, message: &str, placement: bool) -> Result<()> {
        // Nothing stored follows a message composed from a stream.
        if self.stream.is_some() {
            return Ok(());
        }
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
        self.context.changes.session(session_pk);
        Ok(())
    }

    /// Removes a message's blocks and calls; the calls its blocks named and
    /// answered are paired again.
    fn clear_blocks(&mut self, message_pk: i64) -> Result<()> {
        let calls = self
            .tx()
            .prepare_cached(
                "SELECT session_pk,call_id FROM block WHERE message_pk=?1 AND call_id IS NOT NULL",
            )?
            .query_map([message_pk], |r| <(i64, String)>::try_from(r))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        self.tx()
            .prepare_cached(
                "DELETE FROM tool_call WHERE call_block_pk IN
                 (SELECT pk FROM block WHERE message_pk = ?1)",
            )?
            .execute([message_pk])?;
        self.tx()
            .prepare_cached("DELETE FROM block WHERE message_pk = ?1")?
            .execute([message_pk])?;
        self.calls.extend(calls);
        Ok(())
    }

    fn insert_blocks(
        &mut self,
        message_pk: i64,
        session_pk: i64,
        order: &[u8],
        value: &ReadingValue,
    ) -> Result<()> {
        let m = &value.message;
        if value.pieces.len() != m.blocks.len() || value.texts.len() != m.blocks.len() {
            return Err(StoreError::Invalid(
                "prepared message differs from its parsed blocks".into(),
            ));
        }
        value.selection.validate(m.blocks.len())?;
        let origin = m.session.origin().as_str().to_owned();
        let at = m.at.map(Timestamp::unix_ms);
        let mut insert = self.tx().prepare_cached(
            "INSERT INTO block(message_pk, session_pk, position, kind, text_id, content_group,
                 content_start, content_length, data_length, call_id, blob_hash, media_type,
                 file_name, size, url, direction, encrypted, other_type, ordinal)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
        )?;
        for (idx, (block, piece)) in m.blocks.iter().zip(&value.pieces).enumerate() {
            let text_id = match (&value.texts[idx], &value.text_ids) {
                (None, _) => None,
                (Some(_), Some(ids)) => Some(ids.get(idx).copied().flatten().ok_or_else(|| {
                    StoreError::Invalid("a message reading's text reference is missing".into())
                })?),
                (Some(hash), None) => Some(crate::text_rows::resolve(self.tx(), hash, None)?),
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
            let call_id = call_id.map(|id| self.ids.call(&origin, id)).transpose()?;
            let encrypted = matches!(block, Block::Thinking { redacted: true, .. });
            let other_type = match block {
                Block::Other { type_name, .. } => Some(type_name.as_str()),
                _ => None,
            };
            insert.execute(params![
                message_pk,
                session_pk,
                idx as i64,
                kind,
                text_id,
                piece.group,
                piece.start,
                piece.length,
                piece.data_length,
                call_id,
                attachment.and_then(|a| a.blob).map(|h| h.0.to_vec()),
                attachment.and_then(|a| a.media_type.as_deref()),
                attachment.and_then(|a| a.name.as_deref()),
                attachment
                    .and_then(|a| a.size)
                    .and_then(|s| i64::try_from(s).ok()),
                attachment.and_then(|a| a.url.as_deref()),
                attachment.map(|a| name(&a.direction)),
                i64::from(encrypted),
                other_type,
                value.selection.block_ordinals[idx].to_string(),
            ])?;
            let block_pk = self.tx().last_insert_rowid();
            if let Some(stream) = &mut self.stream {
                // A session from a stream pairs its calls from its own
                // blocks once all are written.
                if let (Block::ToolResult { .. }, Some(call)) = (block, &call_id) {
                    let place = (order.to_owned(), m.id.to_string(), idx as i64);
                    if stream
                        .results
                        .get(call)
                        .is_none_or(|first| place < first.place)
                    {
                        stream.results.insert(
                            call.clone(),
                            CallResult {
                                place,
                                outcome: ResultOutcome::of(block),
                                at,
                            },
                        );
                    }
                }
            } else if let Some(call_id) = &call_id {
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
                let call = Call {
                    session_pk,
                    block_pk,
                    call_id: call_id.clone(),
                    at,
                    tool: tool.as_deref().unwrap_or("unknown").to_owned(),
                    action: action.as_str(),
                    command: command.clone(),
                    path: path.clone(),
                    run_by_script: *run_by_script,
                    // The tool-call derivation's value, written with the row so a
                    // first build's derive does not rewrite every shell call.
                    command_norm: (*action == ToolAction::Shell)
                        .then(|| {
                            command
                                .as_deref()
                                .and_then(polylogue_model::command::normalize)
                        })
                        .flatten(),
                };
                match &mut self.stream {
                    Some(stream) => stream.calls.push(call),
                    None => self.insert_call(&call, &Pairing::unpaired())?,
                }
            }
        }
        Ok(())
    }

    /// Writes a tool call's row with its pairing.
    fn insert_call(&self, call: &Call, pairing: &Pairing) -> Result<()> {
        self.tx()
            .prepare_cached(
                "INSERT INTO tool_call(session_pk, call_block_pk, call_id, at, tool, action,
                     command, path, outcome, outcome_reason, run_by_script, command_norm,
                     result_message_id, result_position, exit_code, result_at, reported_duration_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            )?
            .execute(params![
                call.session_pk,
                call.block_pk,
                call.call_id,
                call.at,
                call.tool,
                call.action,
                call.command,
                call.path,
                pairing.outcome,
                pairing.reason,
                call.run_by_script,
                call.command_norm,
                pairing.message,
                pairing.idx,
                pairing.exit,
                pairing.result_at,
                pairing.duration,
            ])?;
        Ok(())
    }

    /// Pairs the calls of a session composed from a stream with their
    /// results and writes their rows.
    fn pair_stream_calls(&mut self, session: &str) -> Result<()> {
        let stream = self.stream();
        let calls = std::mem::take(&mut stream.calls);
        let results = std::mem::take(&mut stream.results);
        let mut paired = HashMap::<String, Pairing>::new();
        for call in &calls {
            let Some(call_id) = &call.call_id else {
                self.insert_call(call, &Pairing::unpaired())?;
                continue;
            };
            if !paired.contains_key(call_id) {
                let result = results
                    .get(call_id)
                    .map(|r| (r.place.1.clone(), r.place.2, r.outcome.clone(), r.at));
                let background = self.background_outcome(session, call_id)?;
                paired.insert(call_id.clone(), Pairing::of(result, background.as_ref()));
            }
            self.insert_call(call, &paired[call_id])?;
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
            let result: Option<(String, i64, crate::content::Piece, Option<i64>)> = self
                .tx()
                .prepare_cached(
                    "SELECT m.id,b.position,b.content_group,b.content_start,b.content_length,b.data_length,m.at
                     FROM block b JOIN message m ON m.pk=b.message_pk
                     WHERE b.session_pk=?1 AND b.call_id=?2 AND b.kind='tool_result'
                     ORDER BY m.session_order,m.id,b.position LIMIT 1",
                )?
                .query_row(params![session_pk, call_id], |r| {
                    Ok((r.get(0)?, r.get(1)?, crate::content::Piece::of(r, 2)?, r.get(6)?))
                })
                .optional()?;
            let result = match result {
                Some((message, idx, piece, at)) => {
                    let block: Block = self.reader.json(self.context.tx, piece.body())?;
                    Some((message, idx, ResultOutcome::of(&block), at))
                }
                None => None,
            };
            let background = self.background_outcome(&session, &call_id)?;
            let columns = Pairing::of(result, background.as_ref());
            let same = self.tx().exists(
                "SELECT NOT EXISTS(SELECT 1 FROM tool_call WHERE session_pk=?1 AND call_id=?2 AND NOT (
                     result_message_id IS ?3 AND result_position IS ?4 AND outcome IS ?5 AND outcome_reason IS ?6
                     AND exit_code IS ?7 AND result_at IS ?8 AND reported_duration_ms IS ?9))",
                params![session_pk, call_id, columns.message, columns.idx, columns.outcome, columns.reason, columns.exit, columns.result_at, columns.duration],
            )?;
            if same {
                continue;
            }
            self.context
                .changes
                .call_result(self.tx(), session_pk, &call_id)?;
            self.context.changes.session(session_pk);
            self.tx()
                .prepare_cached(
                    "UPDATE tool_call SET result_message_id=?3,result_position=?4,outcome=?5,outcome_reason=?6,
                         exit_code=?7,result_at=?8,reported_duration_ms=?9
                     WHERE session_pk=?1 AND call_id=?2",
                )?
                .execute(params![session_pk, call_id, columns.message, columns.idx, columns.outcome, columns.reason, columns.exit, columns.result_at, columns.duration])?;
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
        if let Some(stream) = &self.stream {
            found.extend(
                stream
                    .outcomes
                    .iter()
                    .filter(|o| o.call_id == call)
                    .map(|o| {
                        (
                            o.input_pk,
                            o.chain,
                            o.position,
                            o.input.clone(),
                            o.route,
                            o.body.clone(),
                        )
                    }),
            );
        } else {
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
        let piece = match &self.stream {
            Some(stream) => stream
                .events
                .iter()
                .find(|r| r.event_id == stated)
                .map(|r| r.piece),
            None => self
                .tx()
                .prepare_cached(
                    "SELECT content_group,content_start,content_length,data_length FROM event_parse
                     WHERE event_id=?1 LIMIT 1",
                )?
                .query_row([stated], |r| crate::content::Piece::of(r, 0))
                .optional()?,
        };
        let event = piece
            .map(|piece| readings::read_event(self.tx(), &mut self.reader, piece))
            .transpose()?;
        let Some(event) = event else {
            return Ok((stated.to_owned(), ids));
        };
        let (_, shown) = self.ids.event(&event, stated)?;
        ids.insert(shown.clone());
        Ok((shown, ids))
    }

    /// The readings stating an event under `id`.
    fn event_rows(&mut self, id: &str) -> Result<Vec<EventRow>> {
        if let Some(stream) = &self.stream {
            return Ok(stream
                .event_readings
                .get(id)
                .into_iter()
                .flatten()
                .map(|&n| stream.events[n].clone())
                .collect());
        }
        Ok(self
            .tx()
            .prepare_cached(
                "SELECT i.id,r.chain,r.input_pk,r.route,r.hash,r.position,r.source_start,r.source_start+r.source_length,
                     r.content_group,r.content_start,r.content_length,r.data_length
                 FROM event_parse r JOIN input_ref i ON i.pk=r.input_pk WHERE r.event_id=?1",
            )?
            .query_map([id], |r| {
                Ok(EventRow {
                    input: r.get(0)?,
                    chain: r.get(1)?,
                    input_pk: r.get(2)?,
                    route: r.name(3)?,
                    hash: r.get(4)?,
                    seq: r.get(5)?,
                    start: r.get(6)?,
                    end: r.get(7)?,
                    piece: crate::content::Piece::of(r, 8)?,
                    event_id: id.to_owned(),
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?)
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
            piece: crate::content::Piece,
        }
        let mut found = Vec::new();
        for id in stated {
            for row in self.event_rows(id)? {
                let event = readings::read_event(self.tx(), &mut self.reader, row.piece)?;
                let (moved, id_shown) = self.ids.event(&event, id)?;
                if id_shown != shown {
                    continue;
                }
                found.push(Reading {
                    input: row.input,
                    chain: row.chain,
                    input_pk: row.input_pk,
                    route: row.route,
                    hash: row.hash,
                    event: moved.unwrap_or(event),
                    seq: row.seq,
                    start: row.start,
                    end: row.end,
                    piece: row.piece,
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
        let stored: Option<(i64, Option<i64>, crate::content::Piece)> = if self.stream.is_some() {
            None
        } else {
            self.tx()
                .prepare_cached(
                    "SELECT pk,session_pk,content_group,content_start,content_length,data_length
                     FROM event WHERE id=?1",
                )?
                .query_row([shown], |r| {
                    Ok((r.get(0)?, r.get(1)?, crate::content::Piece::of(r, 2)?))
                })
                .optional()?
        };
        let winner = compose::value(
            &found
                .iter()
                .map(|r| (r.route, r.hash.as_slice(), r.input.as_str()))
                .collect::<Vec<_>>(),
        );
        let Some(winner) = winner.map(|n| &found[n]) else {
            if let Some((pk, ..)) = stored {
                self.context.changes.event(self.tx(), pk)?;
                self.tx().execute("DELETE FROM event WHERE pk=?1", [pk])?;
            }
            return Ok(());
        };
        let after = winner.event.after.as_ref().map(ToString::to_string);
        let seq = if winner.seq >= 0 {
            winner.seq
        } else {
            match &after {
                Some(after) => match &mut self.stream {
                    Some(stream) => match stream.seqs.get(after) {
                        Some(seq) => *seq,
                        // The message it follows is another session's: the
                        // event is composed once every session is written.
                        None => {
                            let session = winner.event.session.as_ref().map(ToString::to_string);
                            for id in stated {
                                stream.deferred.push((session.clone(), id.clone()));
                            }
                            return Ok(());
                        }
                    },
                    None => self.message_seq(after)?.unwrap_or(0),
                },
                None => 0,
            }
        };
        let session_pk = match &winner.event.session {
            Some(session) => Some(self.session_row(&session.to_string())?),
            None => None,
        };
        let piece = winner.piece;
        let order = session_pk.map(|_| {
            crate::encoding::session_order(
                winner.event.at.map(Timestamp::unix_ms),
                &winner.input,
                seq,
                winner.start.unwrap_or(0),
            )
        });
        let time = crate::encoding::TimeColumns::of(winner.event.at, winner.event.time_accuracy());
        let values = params![
            shown,
            session_pk,
            seq,
            after,
            winner.event.kind,
            time.at,
            winner.input_pk,
            winner.chain,
            winner.start,
            winner.end,
            piece.group,
            piece.start,
            piece.length,
            piece.data_length,
            order,
            time.until,
            time.precision_name(),
        ];
        match stored {
            Some((pk, before_session, before)) => {
                let same = self.tx().exists(
                    "SELECT EXISTS(SELECT 1 FROM event WHERE id=?1 AND session_pk IS ?2 AND position IS ?3 AND after_message_id IS ?4
                     AND kind IS ?5 AND at IS ?6 AND input_pk IS ?7 AND source_chain IS ?8 AND source_start IS ?9
                     AND source_end IS ?10)",
                    params![shown, session_pk, seq, after, winner.event.kind,
                        winner.event.at.map(Timestamp::unix_ms), winner.input_pk, winner.chain,
                        winner.start, winner.end],
                )?;
                if same && before == piece {
                    return Ok(());
                }
                // The same reading's content stored again elsewhere moves the
                // row without changing what it shows.
                let moved_only = same && self.same_bytes(before, piece)?;
                if !moved_only {
                    self.context.changes.event(self.tx(), pk)?;
                    if let Some(session) = before_session.filter(|s| Some(*s) != session_pk) {
                        self.context.changes.session(session);
                    }
                }
                self.tx().prepare_cached(
                    "UPDATE event SET session_pk=?2,position=?3,after_message_id=?4,kind=?5,at=?6,input_pk=?7,
                         source_chain=?8,source_start=?9,source_end=?10,content_group=?11,content_start=?12,
                         content_length=?13,data_length=?14,session_order=?15,at_until=?16,at_precision=?17
                     WHERE id=?1",
                )?
                .execute(values)?;
                if moved_only {
                    return Ok(());
                }
            }
            None => {
                self.tx().prepare_cached(
                    "INSERT INTO event(id,session_pk,position,after_message_id,kind,at,input_pk,source_chain,source_start,
                         source_end,content_group,content_start,content_length,data_length,session_order,
                         at_until,at_precision)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)",
                )?
                .execute(values)?;
                let pk = self.tx().last_insert_rowid();
                self.context.changes.new_event(pk);
            }
        }
        if let Some(session) = session_pk {
            self.context.changes.session(session);
        }
        Ok(())
    }

    /// Whether two stored pieces hold the same bytes, data included.
    fn same_bytes(&mut self, a: crate::content::Piece, b: crate::content::Piece) -> Result<bool> {
        if (a.length, a.data_length) != (b.length, b.data_length) {
            return Ok(false);
        }
        let whole = |p: crate::content::Piece| crate::content::Stretch {
            group: p.group,
            start: p.start,
            length: p.length + p.data_length.unwrap_or(0),
        };
        let first = self.reader.bytes(self.tx(), whole(a))?.to_vec();
        Ok(first == self.reader.bytes(self.tx(), whole(b))?)
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
                "SELECT position FROM message_parse WHERE input_pk=?1 AND chain=?2 AND message_id=?3",
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
            "SELECT pk FROM link_statement WHERE parent_id=?1 AND child_id=?2 AND kind=?3 AND reason IS ?4 AND branch_point IS ?5 AND spawning_call IS ?6",
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
                self.tx()
                    .execute("DELETE FROM link_statement WHERE pk=?1", [pk])?;
                if crate::lineage::is_lineage(&kind) {
                    crate::lineage::mark_due(self.tx(), &child)?;
                }
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
                    "SELECT EXISTS(SELECT 1 FROM link_statement WHERE pk=?1 AND evidence IS ?2 AND input_pk IS ?3 AND source_chain IS ?4 AND copies_prefix IS ?5)",
                    params![pk, link.evidence, input_pk, chain, link.copies_prefix],
                )?;
                if same {
                    return Ok(());
                }
                self.context.changes.link(self.tx(), pk)?;
                self.tx()
                    .prepare_cached("UPDATE link_statement SET evidence=?2,input_pk=?3,source_chain=?4,copies_prefix=?5 WHERE pk=?1")?
                    .execute(params![pk, link.evidence, input_pk, chain, link.copies_prefix])?;
            }
            None => {
                self.context.changes.link_child(self.tx(), &child)?;
                self.tx().prepare_cached(
                    "INSERT INTO link_statement(parent_id,child_id,kind,reason,evidence,input_pk,source_chain,branch_point,spawning_call,copies_prefix)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                )?
                .execute(params![parent, child, kind, how, link.evidence, input_pk, chain, at, call, link.copies_prefix])?;
                if crate::lineage::is_lineage(&kind) {
                    crate::lineage::mark_due(self.tx(), &child)?;
                }
            }
        }
        self.mark_link_sessions(&parent, &child)
    }

    /// Writes the relations between `parent` and `child` of one kind and
    /// how from all their statements (`session_link`), in one pass over the
    /// statements, so a reader reads the relations as rows.
    fn resolve_links(
        &mut self,
        (parent, child, kind, how): (String, String, String, Option<String>),
    ) -> Result<()> {
        let statements: Vec<crate::lineage::LinkStatement> = self
            .tx()
            .prepare_cached(
                "SELECT pk, branch_point, spawning_call, copies_prefix, evidence, source_chain, input_pk
                 FROM link_statement WHERE parent_id=?1 AND child_id=?2 AND kind=?3 AND reason IS ?4
                 ORDER BY pk",
            )?
            .query_map(params![parent, child, kind, how], |r| {
                Ok(crate::lineage::LinkStatement {
                    branch_point: r.get(1)?,
                    spawning_call: r.get(2)?,
                    copies_prefix: r.get(3)?,
                    evidence: r.get(4)?,
                    source_chain: r.get(5)?,
                    input_pk: r.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        let resolved = crate::lineage::relations(&statements);
        let stored: Vec<crate::lineage::Relation> = self
            .tx()
            .prepare_cached(
                "SELECT branch_point, spawning_call, copies_prefix, evidence, source_chain, input_pk
                 FROM session_link WHERE parent_id=?1 AND child_id=?2 AND kind=?3 AND reason IS ?4
                 ORDER BY pk",
            )?
            .query_map(params![parent, child, kind, how], |r| {
                Ok(crate::lineage::Relation {
                    branch_point: r.get(0)?,
                    spawning_call: r.get(1)?,
                    copies_prefix: r.get(2)?,
                    evidence: r.get(3)?,
                    source_chain: r.get(4)?,
                    input_pk: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        if stored == resolved {
            return Ok(());
        }
        self.context.changes.link_child(self.tx(), &child)?;
        self.tx()
            .prepare_cached(
                "DELETE FROM session_link WHERE parent_id=?1 AND child_id=?2 AND kind=?3 AND reason IS ?4",
            )?
            .execute(params![parent, child, kind, how])?;
        for relation in &resolved {
            self.tx()
                .prepare_cached(
                    "INSERT INTO session_link(parent_id,child_id,kind,reason,copies_prefix,branch_point,
                         spawning_call,evidence,source_chain,input_pk)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                )?
                .execute(params![
                    parent,
                    child,
                    kind,
                    how,
                    relation.copies_prefix,
                    relation.branch_point,
                    relation.spawning_call,
                    relation.evidence,
                    relation.source_chain,
                    relation.input_pk
                ])?;
        }
        self.mark_link_sessions(&parent, &child)
    }

    fn mark_link_sessions(&mut self, parent: &str, child: &str) -> Result<()> {
        for session in [parent, child] {
            if let Some(pk) = self.session_pk(session)? {
                self.context.changes.session(pk);
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
                "SELECT i.id,r.input_pk,r.chain,r.route,r.at,r.hash,r.source_start,r.source_start+r.source_length
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
                self.tx()
                    .execute("DELETE FROM document_block WHERE document_pk=?1", [pk])?;
                self.tx()
                    .execute("DELETE FROM document WHERE pk=?1", [pk])?;
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
        let time = crate::encoding::TimeColumns::of(document.source_time, document.time_accuracy());
        let pk = match stored {
            Some((pk, _)) => {
                self.tx().execute(
                    "UPDATE document SET purpose=?2,title=?3,at=?4,authorship=?5,version_hash=?6,input_pk=?7,
                         source_chain=?8,source_start=?9,source_end=?10,changed=?11,fields=?12,
                         at_until=?13,at_precision=?14 WHERE pk=?1",
                    params![pk, document.purpose.as_str(), document.title, time.at,
                        document.authorship.as_str(), &prepared.hash[..], input_pk, chain, start, end,
                        self.context.counter, prepared.fields, time.until, time.precision_name()],
                )?;
                pk
            }
            None => {
                self.tx().execute(
                    "INSERT INTO document(id,origin,native_id,purpose,title,at,authorship,version_hash,input_pk,
                         source_chain,source_start,source_end,fields,changed,at_until,at_precision)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
                    params![id, document.id.origin().as_str(), document.id.native().as_str(), document.purpose.as_str(),
                        document.title, time.at, document.authorship.as_str(),
                        &prepared.hash[..], input_pk, chain, start, end, prepared.fields, self.context.counter,
                        time.until, time.precision_name()],
                )?;
                self.tx().last_insert_rowid()
            }
        };
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
            let blob = match block {
                Block::Image(attachment) | Block::File(attachment) => {
                    attachment.blob.map(|h| h.0.to_vec())
                }
                _ => None,
            };
            self.tx().execute(
                "INSERT INTO document_block(document_pk,position,kind,body,text_id,blob_hash,data) VALUES (?1,?2,?3,?4,?5,?6,?7)",
                params![pk, idx as i64, block.kind().as_str(), body, text, blob, data],
            )?;
        }
        Ok(())
    }
}

/// What a call's pairing with its result sets on its row.
struct Pairing {
    message: Option<String>,
    idx: Option<i64>,
    outcome: String,
    reason: Option<String>,
    exit: Option<i64>,
    result_at: Option<i64>,
    duration: Option<i64>,
}

impl Pairing {
    /// A call no result answers.
    fn unpaired() -> Self {
        Self {
            message: None,
            idx: None,
            outcome: "unknown".into(),
            reason: Some("no_result".into()),
            exit: None,
            result_at: None,
            duration: None,
        }
    }

    /// The pairing of a call with its first result, a (message ID, block
    /// index, what the result block says, message time), and with the
    /// outcome learned after it for a call run in the background, which wins.
    fn of(
        result: Option<(String, i64, Option<ResultOutcome>, Option<i64>)>,
        background: Option<&polylogue_model::BackgroundUpdate>,
    ) -> Self {
        let mut columns = Self::unpaired();
        if let Some((message, idx, outcome, at)) = result {
            columns.message = Some(message);
            columns.idx = Some(idx);
            if let Some(result) = outcome {
                columns.outcome = result.outcome.to_owned();
                columns.reason = result.reason;
                columns.exit = result.exit;
                columns.result_at = at;
                columns.duration = result.duration;
            }
        }
        if let Some(update) = background {
            let (outcome, reason, exit) = outcome_columns(&update.outcome);
            columns.outcome = outcome.to_owned();
            columns.reason = reason;
            columns.exit = exit;
            columns.result_at = update.result_at.map(Timestamp::unix_ms);
            columns.duration = update
                .reported_duration_ms
                .and_then(|ms| i64::try_from(ms).ok());
        }
        columns
    }
}

/// What a tool result block states of its call's outcome.
#[derive(Clone)]
struct ResultOutcome {
    outcome: &'static str,
    reason: Option<String>,
    exit: Option<i64>,
    duration: Option<i64>,
}

impl ResultOutcome {
    fn of(block: &Block) -> Option<Self> {
        let Block::ToolResult {
            outcome,
            reported_duration_ms,
            ..
        } = block
        else {
            return None;
        };
        let (outcome, reason, exit) = outcome_columns(outcome);
        Some(Self {
            outcome,
            reason,
            exit,
            duration: reported_duration_ms.and_then(|ms| i64::try_from(ms).ok()),
        })
    }
}

/// The main reading of a shown message and what it is shown with.
struct MainReading<'a> {
    session_pk: i64,
    /// The words of the main reading's text blocks.
    words: i64,
    order: &'a [u8],
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
    "at_until",
    "at_precision",
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
const CONTENT_COLUMNS: usize = 27;

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
    let time = crate::encoding::TimeColumns::of(m.at, m.time_accuracy());
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
        int(time.at),
        int(time.until),
        text(time.precision_name().map(str::to_owned)),
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
        Value::Integer(main.words),
        text(attrs),
        Value::Blob(main.order.to_owned()),
        Value::Integer(main.input_pk),
        Value::Integer(main.chain),
        Value::Integer(main.start),
        Value::Integer(main.end),
        Value::Integer(i64::from(main.in_progress)),
    ])
}

static INSERT_MESSAGE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!(
        "INSERT INTO message(id,{}) VALUES (?1,{})",
        MESSAGE_COLUMNS.join(","),
        (2..=MESSAGE_COLUMNS.len() + 1)
            .map(|n| format!("?{n}"))
            .collect::<Vec<_>>()
            .join(",")
    )
});

/// Marks each text the shown blocks and document blocks hold as a member of
/// the search indexes their kinds name (search_triggers.sql), for a build,
/// whose composition runs before those triggers exist.
pub(super) fn mark_searched_texts(tx: &Transaction<'_>) -> Result<()> {
    tx.execute(
        "UPDATE text SET indexed = b.bits
         FROM (SELECT text_id, max(bit) | min(bit) AS bits FROM (
                   SELECT text_id, iif(kind IN ('tool_result','tool_call'), 2, 1) AS bit
                   FROM block WHERE text_id IS NOT NULL
                   UNION ALL
                   SELECT text_id, 1 FROM document_block WHERE text_id IS NOT NULL)
               GROUP BY text_id) b
         WHERE text.id = b.text_id AND text.indexed <> b.bits",
        [],
    )?;
    Ok(())
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
