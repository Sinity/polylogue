//! Names that belong to one chain, settled when its reading is formed.
//!
//! A parser writes names as the source writes them. Three kinds of name only
//! make sense inside one chain, and are settled here from the chain's rows
//! alone, so a chain read in one step and the same chain read in many give
//! the same reading:
//!
//! - **Responses.** A response written as several records (Claude Code
//!   writes one record per content block, sharing `message.id`) yields one
//!   row per record carrying the response key. Rows of one chain with the
//!   same session and key are one message, named by its first record, with
//!   the blocks of every record in byte order; each later record's own name
//!   becomes an alias of the message.
//! - **Content names.** A message without a native ID is written as its
//!   content name and its record's position ([`crate::content_ids`]); it is
//!   numbered by how many earlier messages of the chain have the same
//!   content name. A reference written the same way names the same message.
//! - **Aliases.** A parent, or the message an event follows, named through
//!   alias rows is the nearest message the aliases lead to; an alias row's
//!   target is stored that way too, so one lookup resolves any name.
//!
//! What a step needs from earlier steps of its chain it reads through
//! [`EarlierRows`]: the store's rows of the chain, or [`Formed`], the rows
//! settled earlier in the same run.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

parser_modules! {}

use crate::content_ids;
use polylogue_model::{Alias, Json, MessageId, MessageRow, Output, SessionId};
use std::collections::{BTreeMap, HashMap};

/// The rows a chain's reading already holds before the step being settled.
pub trait EarlierRows {
    /// Why the earlier rows could not be read.
    type Error;
    /// The message an earlier row of `session` with this response key
    /// formed, with its byte range.
    fn response(
        &mut self,
        session: &SessionId,
        key: &str,
    ) -> Result<Option<MessageRow>, Self::Error>;
    /// What `name` stands for, when an earlier alias row names it:
    /// `Some(None)` stands for no message.
    fn alias(&mut self, name: &MessageId) -> Result<Option<Option<MessageId>>, Self::Error>;
    /// How many earlier messages have this content name.
    fn content_count(&mut self, name: &str) -> Result<u64, Self::Error>;
    /// How many earlier messages have this content name and start before
    /// `position`: the number of the message written as `name@position`.
    fn content_before(&mut self, name: &str, position: u64) -> Result<u64, Self::Error>;
    /// The message of `session` whose first record is the chain's latest.
    fn latest_opened(&mut self, session: &SessionId) -> Result<Option<MessageId>, Self::Error>;
    /// The session of a message of the chain.
    fn session_of(&mut self, message: &MessageId) -> Result<Option<SessionId>, Self::Error>;
}

/// Why a step's names could not be settled.
#[derive(Debug, PartialEq, Eq)]
pub enum SettleError<E> {
    /// The step's own rows cannot be named: the chain's bytes are at fault.
    Content(&'static str),
    /// The chain's earlier rows could not be read.
    Lookup(E),
}

impl<E> From<&'static str> for SettleError<E> {
    fn from(problem: &'static str) -> Self {
        Self::Content(problem)
    }
}

impl<E: std::fmt::Display> std::fmt::Display for SettleError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Content(problem) => f.write_str(problem),
            Self::Lookup(error) => write!(f, "reading the chain's earlier rows: {error}"),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for SettleError<E> {}

impl SettleError<std::convert::Infallible> {
    /// The content problem of a settle that has no earlier rows to read.
    pub fn problem(self) -> &'static str {
        match self {
            Self::Content(problem) => problem,
            Self::Lookup(never) => match never {},
        }
    }
}

/// A step that starts a chain's reading, or restates whole conversations
/// (a capture frame), has no earlier rows to name things by.
pub struct NoEarlierRows;

impl EarlierRows for NoEarlierRows {
    type Error = std::convert::Infallible;
    fn response(&mut self, _: &SessionId, _: &str) -> Result<Option<MessageRow>, Self::Error> {
        Ok(None)
    }
    fn alias(&mut self, _: &MessageId) -> Result<Option<Option<MessageId>>, Self::Error> {
        Ok(None)
    }
    fn content_count(&mut self, _: &str) -> Result<u64, Self::Error> {
        Ok(0)
    }
    fn content_before(&mut self, _: &str, _: u64) -> Result<u64, Self::Error> {
        Ok(0)
    }
    fn latest_opened(&mut self, _: &SessionId) -> Result<Option<MessageId>, Self::Error> {
        Ok(None)
    }
    fn session_of(&mut self, _: &MessageId) -> Result<Option<SessionId>, Self::Error> {
        Ok(None)
    }
}

/// The rows settled earlier in one run over a chain, in front of the rows
/// the chain held before the run (`before`).
pub struct Formed<B> {
    pub before: B,
    responses: HashMap<(SessionId, String), MessageRow>,
    aliases: HashMap<MessageId, Option<MessageId>>,
    /// Start of each message of a content name, in reading order.
    content: HashMap<String, Vec<u64>>,
    /// A content name and position as written, to the number it was given.
    written: HashMap<(String, u64), u64>,
    /// Each session's message whose first record is the latest, with its start.
    opened: HashMap<SessionId, (u64, MessageId)>,
    /// Each message's session.
    sessions: HashMap<MessageId, SessionId>,
}

impl<B: EarlierRows> Formed<B> {
    pub fn new(before: B) -> Self {
        Self {
            before,
            responses: HashMap::new(),
            aliases: HashMap::new(),
            content: HashMap::new(),
            written: HashMap::new(),
            opened: HashMap::new(),
            sessions: HashMap::new(),
        }
    }

    /// Settles a step and keeps what later steps of the run name it by.
    pub fn settle(&mut self, out: &mut Output) -> Result<(), SettleError<B::Error>> {
        let written = settle(out, self)?;
        for row in &out.messages {
            let opened = self
                .opened
                .entry(row.message.session.clone())
                .or_insert((row.offset, row.message.id.clone()));
            if row.offset >= opened.0 {
                *opened = (row.offset, row.message.id.clone());
            }
            self.sessions
                .insert(row.message.id.clone(), row.message.session.clone());
            if let Some(key) = &row.response {
                self.responses
                    .insert((row.message.session.clone(), key.clone()), row.clone());
            }
            if row.id_kind == polylogue_model::IdKind::Content
                && let Some(name) = content_ids::settled_name(row.message.id.native().as_str())
            {
                self.content
                    .entry(name.to_owned())
                    .or_default()
                    .push(row.offset);
            }
        }
        for alias in &out.aliases {
            self.aliases
                .insert(alias.name.clone(), alias.target.clone());
        }
        self.written.extend(written);
        Ok(())
    }
}

impl<B: EarlierRows> EarlierRows for Formed<B> {
    type Error = B::Error;
    fn response(
        &mut self,
        session: &SessionId,
        key: &str,
    ) -> Result<Option<MessageRow>, Self::Error> {
        match self.responses.get(&(session.clone(), key.to_owned())) {
            Some(row) => Ok(Some(row.clone())),
            None => self.before.response(session, key),
        }
    }
    fn alias(&mut self, name: &MessageId) -> Result<Option<Option<MessageId>>, Self::Error> {
        match self.aliases.get(name) {
            Some(target) => Ok(Some(target.clone())),
            None => self.before.alias(name),
        }
    }
    fn content_count(&mut self, name: &str) -> Result<u64, Self::Error> {
        let here = self.content.get(name).map_or(0, Vec::len) as u64;
        Ok(self.before.content_count(name)? + here)
    }
    fn content_before(&mut self, name: &str, position: u64) -> Result<u64, Self::Error> {
        if let Some(number) = self.written.get(&(name.to_owned(), position)) {
            return Ok(*number);
        }
        let here = self.content.get(name).map_or(0, |starts| {
            starts.iter().filter(|start| **start < position).count()
        }) as u64;
        Ok(self.before.content_before(name, position)? + here)
    }
    fn latest_opened(&mut self, session: &SessionId) -> Result<Option<MessageId>, Self::Error> {
        match self.opened.get(session) {
            Some((_, id)) => Ok(Some(id.clone())),
            None => self.before.latest_opened(session),
        }
    }
    fn session_of(&mut self, message: &MessageId) -> Result<Option<SessionId>, Self::Error> {
        match self.sessions.get(message) {
            Some(session) => Ok(Some(session.clone())),
            None => self.before.session_of(message),
        }
    }
}

/// Each content name and position as written, with the number it was given.
pub type Numbers = Vec<((String, u64), u64)>;

/// Settles the chain-local names of one step's rows against the chain's
/// earlier rows. Returns each content name and position as written with the
/// number it was given.
pub fn settle<E>(
    out: &mut Output,
    earlier: &mut dyn EarlierRows<Error = E>,
) -> Result<Numbers, SettleError<E>> {
    let written = number_content(out, earlier)?;
    join(out, earlier)?;
    let mut seen = std::collections::HashSet::new();
    out.links.retain(|link| seen.insert(link.clone()));
    dedup(&mut out.prefix_copies);
    Ok(written)
}

/// Gives each content-named message its number and rewrites every
/// reference written the same way.
fn number_content<E>(
    out: &mut Output,
    earlier: &mut dyn EarlierRows<Error = E>,
) -> Result<Numbers, SettleError<E>> {
    let mut next: HashMap<String, u64> = HashMap::new();
    let mut given: HashMap<MessageId, MessageId> = HashMap::new();
    let mut written = Vec::new();
    for row in &mut out.messages {
        let Some((name, position)) = content_ids::as_written(&row.message.id) else {
            continue;
        };
        let name = name.to_owned();
        let number = match next.get_mut(&name) {
            Some(number) => number,
            None => {
                let first = earlier.content_count(&name).map_err(SettleError::Lookup)?;
                next.entry(name.clone()).or_insert(first)
            }
        };
        let id = content_ids::numbered(row.message.id.origin().as_str(), &name, *number)
            .ok_or("a content name does not form an ID")?;
        written.push(((name, position), *number));
        *number += 1;
        given.insert(std::mem::replace(&mut row.message.id, id.clone()), id);
    }
    let mut rename = |id: &mut MessageId| -> Result<(), SettleError<E>> {
        let Some((name, position)) = content_ids::as_written(id) else {
            return Ok(());
        };
        let settled = match given.get(id) {
            Some(settled) => settled.clone(),
            None => content_ids::numbered(
                id.origin().as_str(),
                name,
                earlier
                    .content_before(name, position)
                    .map_err(SettleError::Lookup)?,
            )
            .ok_or("a content name does not form an ID")?,
        };
        *id = settled;
        Ok(())
    };
    for row in &mut out.messages {
        if let Some(parent) = &mut row.message.parent {
            rename(parent)?;
        }
        // A tool call a source names after its message (`<message>:<n>`)
        // follows the message's settled name.
        let origin = row.message.id.origin().as_str().to_owned();
        for block in &mut row.message.blocks {
            if let polylogue_model::Block::ToolCall {
                call_id: Some(call),
                ..
            }
            | polylogue_model::Block::ToolResult {
                call_id: Some(call),
                ..
            } = block
                && let Some((head, rest)) = call.rsplit_once(':')
                && let Some(mut id) = crate::document::message_id(&origin, head)
                && content_ids::as_written(&id).is_some()
            {
                rename(&mut id)?;
                *call = format!("{}:{rest}", id.native().as_str());
            }
        }
    }
    for event in &mut out.events {
        if let Some(after) = &mut event.after {
            rename(after)?;
        }
        rename_in(&mut event.data, &mut rename)?;
    }
    for link in &mut out.links {
        if let Some(at) = &mut link.branch_point {
            rename(at)?;
        }
    }
    for member in &mut out.prefix_members {
        rename(&mut member.message)?;
    }
    for alias in &mut out.aliases {
        rename(&mut alias.name)?;
        if let Some(target) = &mut alias.target {
            rename(target)?;
        }
    }
    Ok(written)
}

/// Message IDs inside an event's typed data (a compaction boundary's
/// summary and kept messages) are written as strings.
fn rename_in<E>(
    value: &mut Json,
    rename: &mut dyn FnMut(&mut MessageId) -> Result<(), SettleError<E>>,
) -> Result<(), SettleError<E>> {
    match value {
        Json::String(text) if text.contains('@') => {
            if let Ok(mut id) = MessageId::parse(text)
                && content_ids::as_written(&id).is_some()
            {
                rename(&mut id)?;
                *text = id.to_string();
            }
        }
        Json::Array(items) => {
            for item in items {
                rename_in(item, rename)?;
            }
        }
        Json::Object(fields) => {
            for item in fields.values_mut() {
                rename_in(item, rename)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// One thing a step states, at the start of its record. Within one record
/// its aliases come first, then its events, which follow what came before the
/// record, then its messages.
enum Stated {
    Message(usize),
    Alias(usize),
    Event(usize),
}

/// Joins the records of each response into one message and resolves
/// parents and event placements through aliases, in byte order: a name is
/// resolved by the aliases of the records before it. An event placed after a
/// later record of a response follows the latest message opened before it in
/// that response's session, as it follows that message on the timeline. The
/// response's session, not the event's: a record without a session (a file
/// history snapshot) is stated for the file's own session though the messages
/// before it may be a resumed session's. Messages a progress record carries
/// are another session's, so they do not move such an event.
fn join<E>(
    out: &mut Output,
    earlier: &mut dyn EarlierRows<Error = E>,
) -> Result<(), SettleError<E>> {
    let mut stated: Vec<(u64, u8, usize, Stated)> = Vec::new();
    for (index, row) in out.messages.iter().enumerate() {
        stated.push((row.offset, 2, index, Stated::Message(index)));
    }
    for (index, alias) in out.aliases.iter().enumerate() {
        stated.push((alias.offset, 0, index, Stated::Alias(index)));
    }
    for (index, event) in out.events.iter().enumerate() {
        let start = event.range.map_or(u64::MAX, |(start, _)| start);
        stated.push((start, 1, index, Stated::Event(index)));
    }
    stated.sort_by_key(|(offset, order, index, _)| (*offset, *order, *index));

    let mut names: HashMap<MessageId, Option<MessageId>> = HashMap::new();
    let mut open: HashMap<(SessionId, String), usize> = HashMap::new();
    let mut removed = vec![false; out.messages.len()];
    let mut joined = Vec::new();
    let mut reordered = false;
    let mut opened: HashMap<SessionId, MessageId> = HashMap::new();
    let mut sessions: HashMap<MessageId, SessionId> = HashMap::new();
    for (_, _, _, item) in stated {
        match item {
            Stated::Alias(index) => {
                let alias = &mut out.aliases[index];
                if let Some(target) = alias.target.take() {
                    alias.target = resolve(&names, earlier, target)?;
                }
                names.insert(alias.name.clone(), alias.target.clone());
            }
            Stated::Event(index) => {
                let event = &mut out.events[index];
                if let Some(after) = event.after.take() {
                    let resolved = resolve(&names, earlier, after.clone())?;
                    let session = match &resolved {
                        Some(id) if resolved.as_ref() != Some(&after) => match sessions.get(id) {
                            Some(session) => Some(session.clone()),
                            None => earlier.session_of(id).map_err(SettleError::Lookup)?,
                        },
                        _ => None,
                    };
                    event.after = match session {
                        Some(session) => match opened.get(&session) {
                            Some(id) => Some(id.clone()),
                            None => earlier
                                .latest_opened(&session)
                                .map_err(SettleError::Lookup)?
                                .or(resolved),
                        },
                        None => resolved,
                    };
                }
            }
            Stated::Message(index) => {
                let row = &mut out.messages[index];
                if let Some(parent) = row.message.parent.take() {
                    row.message.parent = resolve(&names, earlier, parent)?;
                }
                sessions.insert(row.message.id.clone(), row.message.session.clone());
                let Some(key) = row.response.clone() else {
                    opened.insert(row.message.session.clone(), row.message.id.clone());
                    continue;
                };
                let response = (row.message.session.clone(), key);
                let alias = if let Some(&first) = open.get(&response) {
                    removed[index] = true;
                    let part = out.messages[index].clone();
                    let alias = alias_of(&part, &out.messages[first]);
                    continue_response(&mut out.messages[first], part);
                    alias
                } else if let Some(stated) = earlier
                    .response(&response.0, &response.1)
                    .map_err(SettleError::Lookup)?
                {
                    let part = std::mem::replace(&mut out.messages[index], stated);
                    let alias = alias_of(&part, &out.messages[index]);
                    continue_response(&mut out.messages[index], part);
                    open.insert(response, index);
                    alias
                } else {
                    open.insert(response, index);
                    let row = &out.messages[index];
                    opened.insert(row.message.session.clone(), row.message.id.clone());
                    continue;
                };
                reordered = true;
                if alias.target.as_ref() != Some(&alias.name) {
                    names.insert(alias.name.clone(), alias.target.clone());
                    joined.push(alias);
                }
            }
        }
    }
    if !joined.is_empty() {
        out.aliases.extend(joined);
        out.aliases
            .sort_by(|a, b| (a.offset, &a.name).cmp(&(b.offset, &b.name)));
    }
    if reordered {
        remove_messages(out, &removed);
    }
    Ok(())
}

/// The alias a later record of a response leaves: its own name stands for
/// the response's message.
fn alias_of(part: &MessageRow, message: &MessageRow) -> Alias {
    Alias {
        offset: part.offset,
        end: part.end,
        name: part.message.id.clone(),
        target: Some(message.message.id.clone()),
    }
}

fn resolve<E>(
    names: &HashMap<MessageId, Option<MessageId>>,
    earlier: &mut dyn EarlierRows<Error = E>,
    name: MessageId,
) -> Result<Option<MessageId>, SettleError<E>> {
    if let Some(target) = names.get(&name) {
        return Ok(target.clone());
    }
    Ok(earlier
        .alias(&name)
        .map_err(SettleError::Lookup)?
        .unwrap_or(Some(name)))
}

/// Adds a later record of a response to its message: its blocks follow,
/// and what it states about the whole response (usage, stop reason, cost,
/// duration) replaces what earlier records stated.
fn continue_response(into: &mut MessageRow, part: MessageRow) {
    into.end = into.end.max(part.end);
    let m = &mut into.message;
    let part = part.message;
    let block_offset = m.blocks.len();
    m.blocks.extend(part.blocks);
    if !part.usage.is_empty() {
        m.usage = part.usage;
    }
    m.stop_reason = part.stop_reason.or(m.stop_reason.take());
    m.model = m.model.take().or(part.model);
    m.reported_cost_usd = part.reported_cost_usd.or(m.reported_cost_usd.take());
    m.duration_ms = part.duration_ms.or(m.duration_ms);
    m.production.merge(part.production);
    for (key, mut value) in part.attrs {
        if key == "content" {
            // Per-block details name their block by position.
            if let Some(details) = value.as_array_mut() {
                for detail in details {
                    if let Some(position) = detail.get("position").and_then(Json::as_u64) {
                        detail["position"] = Json::from(position + block_offset as u64);
                    }
                }
            }
            if let Some(Json::Array(existing)) = m.attrs.get_mut(&key) {
                if let Json::Array(additional) = value {
                    existing.extend(additional);
                }
                continue;
            }
        }
        m.attrs.insert(key, value);
    }
}

/// Drops joined message rows, keeping the side maps keyed by message
/// position with their rows, and leaves the rows in byte order.
fn remove_messages(out: &mut Output, removed: &[bool]) {
    let mut order: Vec<usize> = (0..out.messages.len()).filter(|i| !removed[*i]).collect();
    order.sort_by_key(|i| out.messages[*i].offset);
    let position: HashMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(new, old)| (*old, new))
        .collect();
    fn rekey<T>(map: &mut BTreeMap<usize, T>, position: &HashMap<usize, usize>) {
        *map = std::mem::take(map)
            .into_iter()
            .filter_map(|(old, value)| position.get(&old).map(|new| (*new, value)))
            .collect();
    }
    rekey(&mut out.hook_fields, &position);
    rekey(&mut out.stored_fields, &position);
    rekey(&mut out.declared_injections, &position);
    rekey(&mut out.declared_attachment_bindings, &position);
    rekey(&mut out.message_order, &position);
    rekey(&mut out.block_selections, &position);
    let mut rows: Vec<Option<MessageRow>> = std::mem::take(&mut out.messages)
        .into_iter()
        .map(Some)
        .collect();
    out.messages = order
        .into_iter()
        .map(|old| rows[old].take().expect("each row is kept once"))
        .collect();
}

fn dedup<T: PartialEq>(items: &mut Vec<T>) {
    let mut kept: Vec<T> = Vec::with_capacity(items.len());
    for item in items.drain(..) {
        if !kept.contains(&item) {
            kept.push(item);
        }
    }
    *items = kept;
}

#[cfg(test)]
pub(crate) mod tests;
