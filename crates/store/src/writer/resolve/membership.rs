//! Source membership for one worker snapshot: every input chain's reading of
//! each message, document, event and link the batches touch, kept in one
//! table shape for all four kinds. Touched message versions merge compatible
//! block observations before composition; the shown value of each touched
//! item comes from `writer/compose.rs` over each input's latest chain.
use super::*;
use crate::encoding::{PreparedDocument, PreparedMessage};
use crate::writer::compose;
use polylogue_model::{Completion, FieldGroup, IdKind, Route};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct MessageChoice {
    /// The main route's reading: its version is the stored one.
    pub input: i64,
    pub chain: i64,
    pub start: i64,
    pub end: i64,
    pub hash: Vec<u8>,
    pub session: i64,
    pub route: Route,
    pub order: (i64, (i64, i64)),
    /// Each field group taken from another input chain than the main route.
    pub routes: Vec<(FieldGroup, i64, i64)>,
    pub changed: bool,
    /// The composed message and the main version's prepared rows, when
    /// changed.
    pub message: Option<Message>,
    pub prepared: Option<PreparedMessage>,
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct DocumentChoice {
    pub input: i64,
    pub chain: i64,
    pub route: Route,
    pub start: i64,
    pub end: i64,
    pub changed: bool,
    pub hash: Vec<u8>,
    pub document: Option<polylogue_model::Document>,
    pub prepared: Option<PreparedDocument>,
}
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Choices {
    pub messages: BTreeMap<i64, Option<MessageChoice>>,
    pub message_versions: BTreeMap<BlobHash, MessageVersion>,
    pub documents: BTreeMap<i64, Option<DocumentChoice>>,
    pub sessions: BTreeMap<i64, Session>,
    pub message_mains: Vec<bool>,
    pub keep_messages: Vec<bool>,
    pub keep_documents: Vec<bool>,
    pub events: BTreeMap<i64, Option<EventChoice>>,
    pub links: BTreeMap<i64, Option<LinkChoice>>,
    /// Older-chain rows of content-identified messages that the input's
    /// latest chain, read to its end, no longer holds: (message, input, chain).
    pub dropped_message_sources: Vec<(i64, i64, i64)>,
    pub event_values: Vec<PackedValue>,
    pub event_ids: Vec<String>,
    pub event_sources: Vec<(i64, EventChoice)>,
    pub link_values: Vec<PackedValue>,
    pub live_events: Vec<Event>,
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct MessageVersion {
    pub message: Message,
    pub prepared: PreparedMessage,
    pub replace: bool,
    pub admitted: Option<u64>,
}

/// One input chain's reading of an item.
trait Member {
    fn input(&self) -> i64;
    fn chain(&self) -> i64;
}

macro_rules! members {
    ($($kind:ty),*) => {$(
        impl Member for $kind {
            fn input(&self) -> i64 {
                self.input
            }
            fn chain(&self) -> i64 {
                self.chain
            }
        }
    )*};
}
members!(MessageMember, DocumentMember, ValueMember);

/// Every input chain's reading of each item of one kind, and the items a
/// step touched.
struct Table<M> {
    members: BTreeMap<i64, Vec<M>>,
    touched: BTreeSet<i64>,
}
impl<M: Member> Table<M> {
    fn new() -> Self {
        Self {
            members: BTreeMap::new(),
            touched: BTreeSet::new(),
        }
    }
    /// Forgets the chain's readings; each item they were of is touched.
    fn reset(&mut self, input: i64, chain: i64) {
        for (pk, members) in &mut self.members {
            let before = members.len();
            members.retain(|m| m.input() != input || m.chain() != chain);
            if members.len() != before {
                self.touched.insert(*pk);
            }
        }
    }
    fn find(&self, pk: i64, input: i64, chain: i64) -> Option<&M> {
        self.members
            .get(&pk)?
            .iter()
            .find(|m| m.input() == input && m.chain() == chain)
    }
    /// Records the chain's reading of the item, replacing an earlier one.
    fn put(&mut self, pk: i64, member: M) {
        let members = self.members.entry(pk).or_default();
        match members
            .iter()
            .position(|m| m.input() == member.input() && m.chain() == member.chain())
        {
            Some(n) => members[n] = member,
            None => members.push(member),
        }
        self.touched.insert(pk);
    }
    /// The touched items with each input's newest chain among their
    /// readings, and clears the touched set.
    fn take_touched(&mut self) -> Vec<(i64, Vec<&M>)> {
        let touched = std::mem::take(&mut self.touched);
        touched
            .into_iter()
            .map(|pk| {
                let members = self.members.get(&pk).map_or(&[][..], Vec::as_slice);
                (pk, latest(members))
            })
            .collect()
    }
}

/// Each input's newest chain among an item's readings.
fn latest<M: Member>(members: &[M]) -> Vec<&M> {
    let mut newest = BTreeMap::<i64, i64>::new();
    for m in members {
        let chain = newest.entry(m.input()).or_insert(m.chain());
        *chain = (*chain).max(m.chain());
    }
    members
        .iter()
        .filter(|m| newest.get(&m.input()) == Some(&m.chain()))
        .collect()
}

#[derive(Clone)]
struct MessageMember {
    input: i64,
    input_id: String,
    chain: i64,
    start: i64,
    end: i64,
    session: i64,
    main: bool,
    route: Route,
    completion: Completion,
    seq: i64,
    position: i64,
    hash: Vec<u8>,
    /// The message is named by its content, not by the source.
    content: bool,
}

#[derive(Clone)]
struct DocumentMember {
    input: i64,
    input_id: String,
    chain: i64,
    start: i64,
    end: i64,
    route: Route,
    hash: Vec<u8>,
    time: Option<Timestamp>,
}

/// An event's or link's reading: a value shared by hash.
#[derive(Clone)]
struct ValueMember {
    input: i64,
    input_id: String,
    chain: i64,
    route: Route,
    value_pk: i64,
    hash: Vec<u8>,
    body: Vec<u8>,
    event: Option<EventChoice>,
}

/// A closed-set value stored by its name.
fn named<T: polylogue_model::StoredName>(name: String) -> rusqlite::Result<T> {
    T::from_name(&name).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            format!("unknown stored name {name}").into(),
        )
    })
}

pub(in crate::writer) trait ValueSource {
    fn session(&self, pk: i64, input: i64, chain: i64) -> Result<Option<Session>>;
    fn assertion(&self, kind: Kind, pk: i64) -> Result<Option<Vec<u8>>>;
    fn event_source(&self, pk: i64, input: i64, chain: i64) -> Result<Option<EventChoice>>;
    fn message(&self, hash: &[u8]) -> Result<Option<(Message, PreparedMessage)>>;
    fn document(
        &self,
        hash: &[u8],
    ) -> Result<Option<(polylogue_model::Document, PreparedDocument)>>;
}

fn merge_message(
    mut previous: (Message, PreparedMessage),
    mut incoming: (Message, PreparedMessage),
) -> Result<(Message, PreparedMessage)> {
    crate::writer::attachments::restore_values(&mut previous.0.blocks, &previous.1)?;
    crate::writer::attachments::restore_values(&mut incoming.0.blocks, &incoming.1)?;
    let (message, projection) = previous
        .1
        .projection
        .merge(&previous.0, &incoming.1.projection, &incoming.0)?
        .ok_or(polylogue_model::bundle::BundleError::SourceVersionMismatch)?;
    if projection == previous.1.projection {
        return Ok(previous);
    }
    if projection == incoming.1.projection {
        return Ok(incoming);
    }
    let prepared = PreparedMessage::projected(&message, &projection)?;
    Ok((message, prepared))
}

fn known_message(
    index: &Connection,
    values: Option<&dyn ValueSource>,
    hash: &[u8],
) -> Result<Option<(Message, PreparedMessage)>> {
    let stored = crate::versions::read_message(index, hash)?
        .map(|(message, projection)| {
            let prepared = PreparedMessage::projected(&message, &projection)?;
            Ok::<_, StoreError>((message, prepared))
        })
        .transpose()?;
    let staged = values
        .map(|values| values.message(hash))
        .transpose()?
        .flatten()
        .map(|(mut message, prepared)| {
            crate::writer::attachments::restore_values(&mut message.blocks, &prepared)?;
            Ok::<_, StoreError>((message, prepared))
        })
        .transpose()?;
    match (stored, staged) {
        (Some(stored), Some(staged)) => Ok(Some(merge_message(stored, staged)?)),
        (stored, staged) => Ok(stored.or(staged)),
    }
}

/// What a message shows now: the main route's version, session and layout,
/// and each field group taken from another input chain with that reading's
/// version.
#[derive(Clone, PartialEq)]
struct Shown {
    hash: Vec<u8>,
    route: Route,
    session: i64,
    layout: Option<Vec<u8>>,
    routes: Vec<(FieldGroup, i64, i64, Vec<u8>)>,
}

/// A stored message's readings and what it shows.
fn load_message(
    index: &Connection,
    pk: i64,
    layouts: &mut BTreeMap<Vec<u8>, Option<Vec<u8>>>,
) -> Result<(Vec<MessageMember>, Option<Shown>)> {
    let shown = index
        .query_row(
            "SELECT s.version_hash,s.route,m.session_pk,p.layout FROM message m
             JOIN message_source s ON s.message_pk=m.pk AND s.input_pk=m.input_pk AND s.chain=m.source_chain
             LEFT JOIN message_projection p ON p.version_hash=s.version_hash WHERE m.pk=?1",
            [pk],
            |r| {
                Ok(Shown {
                    hash: r.get(0)?,
                    route: named(r.get(1)?)?,
                    session: r.get(2)?,
                    layout: r.get(3)?,
                    routes: Vec::new(),
                })
            },
        )
        .optional()?;
    let shown = match shown {
        Some(mut shown) => {
            shown.routes = index
                .prepare_cached(
                    "SELECT r.field_group,r.input_pk,r.chain,s.version_hash FROM message_route r
                     JOIN message_source s ON s.message_pk=r.message_pk AND s.input_pk=r.input_pk AND s.chain=r.chain
                     WHERE r.message_pk=?1 ORDER BY r.field_group",
                )?
                .query_map([pk], |r| {
                    let group: String = r.get(0)?;
                    Ok((group, r.get(1)?, r.get(2)?, r.get(3)?))
                })?
                .map(|row| {
                    let (group, input, chain, hash) = row?;
                    let group = FieldGroup::from_name(&group).ok_or_else(|| {
                        StoreError::Invalid(format!("unknown field group {group}"))
                    })?;
                    Ok((group, input, chain, hash))
                })
                .collect::<Result<Vec<_>>>()?;
            shown.routes.sort();
            Some(shown)
        }
        None => None,
    };
    let mut statement = index.prepare_cached(
        "SELECT s.input_pk,s.chain,s.source_start,s.source_end,s.session_pk,s.main,s.completion,s.route,s.seq,s.record_position,s.version_hash,s.id_kind,p.layout,i.id
         FROM message_source s JOIN input_ref i ON i.pk=s.input_pk
         LEFT JOIN message_projection p ON p.version_hash=s.version_hash WHERE message_pk=?1",
    )?;
    let rows = statement
        .query_map([pk], |r| {
            Ok((
                MessageMember {
                    input: r.get(0)?,
                    input_id: r.get(13)?,
                    chain: r.get(1)?,
                    start: r.get(2)?,
                    end: r.get(3)?,
                    session: r.get(4)?,
                    main: r.get(5)?,
                    completion: named(r.get(6)?)?,
                    route: named(r.get(7)?)?,
                    seq: r.get(8)?,
                    position: r.get(9)?,
                    hash: r.get(10)?,
                    content: r.get::<_, String>(11)? == IdKind::Content.as_str(),
                },
                r.get::<_, Option<Vec<u8>>>(12)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let members = rows
        .into_iter()
        .map(|(member, layout)| {
            layouts.insert(member.hash.clone(), layout);
            member
        })
        .collect();
    Ok((members, shown))
}

fn load_document(index: &Connection, pk: i64) -> Result<Vec<DocumentMember>> {
    Ok(index
        .prepare_cached("SELECT s.input_pk,s.chain,s.source_start,s.source_end,s.route,s.version_hash,s.time,i.id FROM document_source s JOIN input_ref i ON i.pk=s.input_pk WHERE s.document_pk=?1")?
        .query_map([pk], |r| {
            Ok(DocumentMember {
                input: r.get(0)?,
                input_id: r.get(7)?,
                chain: r.get(1)?,
                start: r.get(2)?,
                end: r.get(3)?,
                route: named(r.get(4)?)?,
                hash: r.get(5)?,
                time: crate::encoding::read_time(r, 6)?.0,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?)
}

fn load_event(index: &Connection, pk: i64) -> Result<Vec<ValueMember>> {
    Ok(index
        .prepare_cached("SELECT s.input_pk,s.chain,s.route,s.value_pk,v.hash,s.session_pk,s.seq,s.source_start,s.source_end,s.after_message_id,s.at,s.admitted,i.id FROM event_source s JOIN event_value v ON v.pk=s.value_pk JOIN input_ref i ON i.pk=s.input_pk WHERE s.event_pk=?1")?
        .query_map([pk], |r| {
            Ok(ValueMember {
                input: r.get(0)?,
                input_id: r.get(12)?,
                chain: r.get(1)?,
                route: named(r.get(2)?)?,
                value_pk: r.get(3)?,
                hash: r.get(4)?,
                body: Vec::new(),
                event: Some(EventChoice {
                    admitted: r.get(11)?,
                    kind: String::new(),
                    fields: Vec::new(),
                    imported: String::new(),
                    value_pk: r.get(3)?,
                    input: r.get(0)?,
                    chain: r.get(1)?,
                    session: r.get(5)?,
                    seq: r.get(6)?,
                    start: r.get(7)?,
                    end: r.get(8)?,
                    after: r.get(9)?,
                    at: r.get(10)?,
                    body: Vec::new(),
                }),
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?)
}

fn load_link(index: &Connection, pk: i64) -> Result<Vec<ValueMember>> {
    Ok(index
        .prepare_cached("SELECT s.input_pk,s.chain,s.route,s.value_pk,v.hash,i.id FROM link_source s JOIN link_value v ON v.pk=s.value_pk JOIN input_ref i ON i.pk=s.input_pk WHERE s.link_pk=?1")?
        .query_map([pk], |r| {
            Ok(ValueMember {
                input: r.get(0)?,
                input_id: r.get(5)?,
                chain: r.get(1)?,
                route: named(r.get(2)?)?,
                value_pk: r.get(3)?,
                hash: r.get(4)?,
                body: Vec::new(),
                event: None,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?)
}

pub(in crate::writer) struct Projection<'a> {
    resolver: &'a Resolver,
    index: &'a Connection,
    keys: &'a ResolvedKeys,
    messages: Table<MessageMember>,
    documents: Table<DocumentMember>,
    events: Table<ValueMember>,
    links: Table<ValueMember>,
    shown_messages: BTreeMap<i64, Option<Shown>>,
    shown_documents: BTreeMap<i64, Option<Vec<u8>>>,
    message_values: BTreeMap<Vec<u8>, (Message, PreparedMessage)>,
    message_layouts: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    document_values: BTreeMap<Vec<u8>, (polylogue_model::Document, PreparedDocument)>,
    session_facts: BTreeMap<(i64, i64, i64), Session>,
    reset_inputs: BTreeSet<(i64, i64)>,
    values: Option<&'a dyn ValueSource>,
}
impl<'a> Projection<'a> {
    pub(in crate::writer) fn new(
        resolver: &'a Resolver,
        index: &'a Connection,
        keys: &'a ResolvedKeys,
        values: Option<&'a dyn ValueSource>,
    ) -> Result<Self> {
        let mut projection = Self {
            resolver,
            index,
            keys,
            messages: Table::new(),
            documents: Table::new(),
            events: Table::new(),
            links: Table::new(),
            shown_messages: BTreeMap::new(),
            shown_documents: BTreeMap::new(),
            message_values: BTreeMap::new(),
            message_layouts: BTreeMap::new(),
            document_values: BTreeMap::new(),
            session_facts: BTreeMap::new(),
            reset_inputs: BTreeSet::new(),
            values,
        };
        for (key, row) in &keys.rows {
            let pk = row.pk;
            match key.kind {
                Kind::Message if row.existing => {
                    let (members, shown) =
                        load_message(index, pk, &mut projection.message_layouts)?;
                    projection.shown_messages.insert(pk, shown);
                    projection.messages.members.insert(pk, members);
                }
                Kind::Document if row.existing => {
                    projection.shown_documents.insert(
                        pk,
                        index
                            .query_row("SELECT version_hash FROM document WHERE pk=?1", [pk], |r| {
                                r.get::<_, Vec<u8>>(0)
                            })
                            .optional()?,
                    );
                    projection
                        .documents
                        .members
                        .insert(pk, load_document(index, pk)?);
                }
                Kind::Event if row.existing => {
                    projection.events.members.insert(pk, load_event(index, pk)?);
                }
                Kind::Link if row.existing => {
                    projection.links.members.insert(pk, load_link(index, pk)?);
                }
                Kind::Message => {
                    projection.messages.members.insert(pk, Vec::new());
                }
                Kind::Document => {
                    projection.documents.members.insert(pk, Vec::new());
                }
                _ => {}
            }
        }
        Ok(projection)
    }

    pub(in crate::writer) fn step(
        &mut self,
        input_key: &InputKey,
        chain: i64,
        step: &mut ParseStep,
    ) -> Result<()> {
        let input = self.keys.get(Kind::Input, input_key.id.as_bytes())?.pk;
        let mut choices = Choices::default();
        if step.reset {
            self.reset_inputs.insert((input, chain));
            self.messages.reset(input, chain);
            self.documents.reset(input, chain);
            self.events.reset(input, chain);
            self.links.reset(input, chain);
            self.session_facts
                .retain(|(_, i, c), _| *i != input || *c != chain);
        }
        self.sessions(input, chain, step, &mut choices)?;
        self.message_readings(input_key, input, chain, step, &mut choices)?;
        self.document_readings(input_key, input, chain, step, &mut choices)?;
        if matches!(step.progress, polylogue_model::batch::StepProgress::AtEnd) {
            self.drop_stale_content(input, chain, &mut choices)?;
        }
        self.event_readings(input_key, input, chain, step, &mut choices)?;
        self.link_readings(input_key, input, chain, step, &mut choices)?;
        self.choose_messages(&mut choices)?;
        self.choose_documents(&mut choices)?;
        self.choose_events(&mut choices)?;
        self.choose_links(&mut choices)?;
        step.prepared.choices = Some(choices);
        if self.values.is_some() {
            self.message_values.clear();
            self.document_values.clear();
            self.session_facts.clear();
            for member in self
                .events
                .members
                .values_mut()
                .chain(self.links.members.values_mut())
                .flatten()
            {
                member.body.clear();
                if let Some(event) = &mut member.event {
                    event.body.clear();
                    event.fields.clear();
                    event.kind.clear();
                    event.imported.clear();
                }
            }
        }
        Ok(())
    }

    /// The facts this chain states about each session, merged with what it
    /// stated before; a session's kind decides whether its messages' readings
    /// come from a main session.
    fn sessions(
        &mut self,
        input: i64,
        chain: i64,
        step: &ParseStep,
        choices: &mut Choices,
    ) -> Result<()> {
        for session in &step.output.sessions {
            let pk = self
                .keys
                .get(Kind::Session, session.id.to_string().as_bytes())?
                .pk;
            let key = (pk, input, chain);
            if let std::collections::btree_map::Entry::Vacant(slot) = self.session_facts.entry(key)
                && let Some(session) = self
                    .values
                    .map(|values| values.session(pk, input, chain))
                    .transpose()?
                    .flatten()
            {
                slot.insert(session);
            }
            if !self.session_facts.contains_key(&key)
                && !self.reset_inputs.contains(&(input, chain))
            {
                let body:Option<String>=self.index.query_row("SELECT facts FROM session_record WHERE session_pk=?1 AND input_pk=?2 AND chain=?3",params![pk,input,chain],|r|r.get(0)).optional()?;
                if let Some(body) = body {
                    self.session_facts.insert(key, serde_json::from_str(&body)?);
                }
            }
            let merged = self
                .session_facts
                .entry(key)
                .or_insert_with(|| session.clone());
            merged.merge(session);
            choices.sessions.insert(pk, merged.clone());
            let main = merged.kind == polylogue_model::SessionKind::Main;
            for (message_pk, members) in &mut self.messages.members {
                for member in members
                    .iter_mut()
                    .filter(|m| m.session == pk && m.input == input && m.chain == chain)
                {
                    if member.main != main {
                        self.messages.touched.insert(*message_pk);
                    }
                    member.main = main;
                }
            }
        }
        Ok(())
    }

    fn message_readings(
        &mut self,
        input_key: &InputKey,
        input: i64,
        chain: i64,
        step: &mut ParseStep,
        choices: &mut Choices,
    ) -> Result<()> {
        let mut positions = BTreeMap::<u64, i64>::new();
        for (i, (row, prepared)) in step
            .output
            .messages
            .iter()
            .zip(&step.prepared.messages)
            .enumerate()
        {
            let pk = self
                .keys
                .get(Kind::Message, row.message.id.to_string().as_bytes())?
                .pk;
            let session = self
                .keys
                .get(Kind::Session, row.message.session.to_string().as_bytes())?
                .pk;
            let previous = self.messages.find(pk, input, chain).map(|m| m.start);
            let keep = !previous.is_some_and(|start| {
                if step.repeats_whole_conversations {
                    (row.offset as i64) < start
                } else {
                    (row.offset as i64) > start
                }
            });
            choices.keep_messages.push(keep);
            // A non-selected source value still belongs to the archive. Keyed
            // parser continuation can cite it without changing shown membership.
            let incoming_layout = crate::versions::message_layout(&prepared.projection)?;
            let prior_layout = self.message_layouts.get(prepared.hash.as_slice());
            // Complete values already proved this hash. Sparse values alone
            // require payload comparison and enrichment of a known version.
            let complete = prepared.projection.block_count == prepared.blocks.len() as u64;
            let previous_value = if prior_layout.is_none()
                || (complete
                    && prior_layout.is_some_and(|layout| {
                        layout
                            .as_ref()
                            .is_none_or(|layout| layout == &incoming_layout)
                    })) {
                None
            } else {
                match self.message_values.get(prepared.hash.as_slice()) {
                    Some(value) => Some(value.clone()),
                    None => known_message(self.index, self.values, &prepared.hash)?,
                }
            };
            let incoming = (row.message.clone(), prepared.clone());
            let merged = match previous_value {
                Some(previous) => merge_message(previous, incoming)?,
                None => incoming,
            };
            let layout = crate::versions::message_layout(&merged.1.projection)?;
            let replace = prior_layout
                .and_then(Option::as_ref)
                .is_some_and(|before| before != &layout)
                || choices
                    .message_versions
                    .get(&BlobHash(prepared.hash))
                    .is_some_and(|previous| previous.replace);
            self.message_layouts
                .insert(prepared.hash.to_vec(), Some(layout));
            for (block, hash) in merged.0.blocks.iter().zip(&merged.1.texts) {
                if let (Some(text), Some(hash)) = (crate::encoding::block_text(block), hash)
                    && let std::collections::hash_map::Entry::Vacant(slot) =
                        step.prepared.texts.entry(*hash)
                {
                    slot.insert(crate::encoding::pack_text(text)?);
                }
            }
            choices.message_versions.insert(
                BlobHash(prepared.hash),
                MessageVersion {
                    message: merged.0.clone(),
                    prepared: merged.1.clone(),
                    replace,
                    admitted: step.output.message_receipts.get(&i).copied().max(
                        choices
                            .message_versions
                            .get(&BlobHash(prepared.hash))
                            .and_then(|version| version.admitted),
                    ),
                },
            );
            self.message_values.insert(prepared.hash.to_vec(), merged);
            self.messages.touched.insert(pk);
            if !keep {
                choices.message_mains.push(false);
                continue;
            }
            if let std::collections::btree_map::Entry::Vacant(slot) =
                self.session_facts.entry((session, input, chain))
                && let Some(facts) = self
                    .values
                    .map(|values| values.session(session, input, chain))
                    .transpose()?
                    .flatten()
            {
                slot.insert(facts);
            }
            let main=match self.session_facts.get(&(session,input,chain)) {
                    Some(facts)=>facts.kind==polylogue_model::SessionKind::Main,
                    None if self.reset_inputs.contains(&(input,chain))=>true,
                    None=>self.index.query_row("SELECT coalesce((SELECT json_extract(facts,'$.kind') FROM session_record WHERE session_pk=?1 AND input_pk=?2 AND chain=?3),'main')",params![session,input,chain],|r|r.get::<_,String>(0))?=="main",
                };
            choices.message_mains.push(main);
            let (seq, position) = match step.message_position(i)? {
                Some(order) => (order, 0),
                None => {
                    let n = positions.entry(row.offset).or_default();
                    let p = *n;
                    *n += 1;
                    (row.offset as i64, p)
                }
            };
            let route = row.route(input_key.route);
            self.messages.put(
                pk,
                MessageMember {
                    input,
                    input_id: input_key.id.clone(),
                    chain,
                    start: row.offset as i64,
                    end: row.end.max(row.offset) as i64,
                    session,
                    main,
                    route,
                    completion: Completion::of(&row.message, route),
                    seq,
                    position,
                    hash: prepared.hash.to_vec(),
                    content: row.id_kind == IdKind::Content,
                },
            );
        }
        Ok(())
    }

    fn document_readings(
        &mut self,
        input_key: &InputKey,
        input: i64,
        chain: i64,
        step: &ParseStep,
        choices: &mut Choices,
    ) -> Result<()> {
        for (row, prepared) in step.output.documents.iter().zip(&step.prepared.documents) {
            let pk = self
                .keys
                .get(Kind::Document, row.document.id.to_string().as_bytes())?
                .pk;
            let time = row.document.source_time;
            let route = input_key.route;
            // A chain that holds the document twice shows the reading the
            // document choice would prefer between them.
            let keep = !self.documents.find(pk, input, chain).is_some_and(|prior| {
                let readings = [
                    compose::DocumentReading {
                        input: &input_key.id,
                        time: prior.time,
                        route: prior.route,
                        hash: &prior.hash,
                    },
                    compose::DocumentReading {
                        input: &input_key.id,
                        time,
                        route,
                        hash: &prepared.hash,
                    },
                ];
                compose::document(&readings) == Some(0) && prior.hash != prepared.hash
            });
            choices.keep_documents.push(keep);
            if !keep {
                continue;
            }
            self.documents.put(
                pk,
                DocumentMember {
                    input,
                    input_id: input_key.id.clone(),
                    chain,
                    start: row.offset as i64,
                    end: row.end as i64,
                    route,
                    hash: prepared.hash.to_vec(),
                    time,
                },
            );
            self.document_values.insert(
                prepared.hash.to_vec(),
                (row.document.clone(), prepared.clone()),
            );
        }
        Ok(())
    }

    /// Once an input's latest chain is read to its end, a message named by
    /// its content that the chain no longer holds is gone from that input:
    /// the rewrite deleted or edited it, and an edit has a new ID.
    fn drop_stale_content(&mut self, input: i64, chain: i64, choices: &mut Choices) -> Result<()> {
        // A rebuild may read a newer chain before an older one; the
        // input's latest chain is whichever is newest so far.
        let stored_latest: Option<i64> = self
            .index
            .prepare_cached("SELECT max(chain) FROM message_source WHERE input_pk=?1")?
            .query_row([input], |r| r.get(0))?;
        let latest = self
            .messages
            .members
            .values()
            .flatten()
            .filter(|m| m.input == input)
            .map(|m| m.chain)
            .chain(stored_latest)
            .chain([chain])
            .max()
            .unwrap_or(chain);
        let mut stale = self
            .index
            .prepare_cached(
                "SELECT DISTINCT message_pk FROM message_source
                 WHERE input_pk=?1 AND chain<?2 AND id_kind='content'",
            )?
            .query_map(params![input, latest], |r| r.get::<_, i64>(0))?
            .collect::<rusqlite::Result<BTreeSet<i64>>>()?;
        stale.extend(
            self.messages
                .members
                .iter()
                .filter(|(_, members)| {
                    members
                        .iter()
                        .any(|m| m.content && m.input == input && m.chain < latest)
                })
                .map(|(pk, _)| *pk),
        );
        for pk in stale {
            if let std::collections::btree_map::Entry::Vacant(entry) =
                self.messages.members.entry(pk)
            {
                let (members, shown) = load_message(self.index, pk, &mut self.message_layouts)?;
                self.shown_messages.insert(pk, shown);
                entry.insert(members);
            }
            let members = self.messages.members.get_mut(&pk).expect("loaded");
            if members
                .iter()
                .any(|m| m.input == input && m.chain == latest)
            {
                continue;
            }
            members.retain(|m| {
                let gone = m.content && m.input == input && m.chain < latest;
                if gone {
                    choices.dropped_message_sources.push((pk, m.input, m.chain));
                    self.messages.touched.insert(pk);
                }
                !gone
            });
        }
        Ok(())
    }

    fn event_readings(
        &mut self,
        input_key: &InputKey,
        input: i64,
        chain: i64,
        step: &ParseStep,
        choices: &mut Choices,
    ) -> Result<()> {
        let event_ids = step.event_identities(&input_key.id, chain)?;
        let mut packer = crate::encoding::RowPacker::new()?;
        for (i, (event, packed)) in step
            .output
            .events
            .iter()
            .zip(&step.prepared.events)
            .enumerate()
        {
            let imported = step.output.imported_events.get(&i);
            let id = event_ids[i].clone();
            let pk = self.keys.get(Kind::Event, id.as_bytes())?.pk;
            let data: Json = crate::encoding::unpack_json(&packed.1)?;
            let value = self.resolver.value(self.index, Kind::EventValue, &data)?;
            let session = event
                .session
                .as_ref()
                .map(|id| {
                    self.keys
                        .get(Kind::Session, id.to_string().as_bytes())
                        .map(|row| row.pk)
                })
                .transpose()?;
            let seq = step
                .order_from
                .map(|n| n as i64)
                .or((imported.is_none())
                    .then_some(event.range)
                    .flatten()
                    .map(|r| r.0 as i64))
                .or(step
                    .repeats_whole_conversations
                    .then_some(step.start as i64));
            let mut fields = event.clone();
            fields.data = Json::Null;
            let fields = packer.pack_json(&fields)?;
            let mut original = imported
                .map(|identity| identity.sources.clone())
                .unwrap_or_default();
            if let Some(previous) = self
                .events
                .find(pk, input, chain)
                .filter(|previous| previous.value_pk == value.key.pk)
            {
                let previous = previous.event_source(self.index, pk, self.values)?;
                if previous.fields == fields {
                    original.extend(serde_json::from_str::<
                        Vec<polylogue_model::bundle::SourceClaim>,
                    >(&previous.imported)?);
                }
            }
            original.sort();
            original.dedup();
            let local_range = if imported.is_some() {
                Some((step.start, step.parsed_upto))
            } else {
                event.range.or(Some((step.start, step.parsed_upto)))
            };
            let choice = EventChoice {
                admitted: step
                    .output
                    .event_receipts
                    .get(&i)
                    .copied()
                    .map(i64::try_from)
                    .transpose()
                    .map_err(|_| {
                        StoreError::Invalid(
                            "canonical event admission exceeds SQLite's integer range".into(),
                        )
                    })?,
                kind: event.kind.clone(),
                fields,
                imported: serde_json::to_string(&original)?,
                value_pk: value.key.pk,
                input,
                chain,
                session,
                seq,
                start: local_range.map(|r| r.0 as i64),
                end: local_range.map(|r| r.1 as i64),
                after: event.after.as_ref().map(ToString::to_string),
                at: event.at.map(Timestamp::unix_ms),
                body: value.body.clone(),
            };
            self.events.put(
                pk,
                ValueMember {
                    input,
                    input_id: input_key.id.clone(),
                    chain,
                    route: input_key.route,
                    value_pk: value.key.pk,
                    hash: value.hash.to_vec(),
                    body: value.body.clone(),
                    event: Some(choice.clone()),
                },
            );
            choices.event_ids.push(id);
            choices.event_sources.push((pk, choice));
            choices.event_values.push(value);
            choices.live_events.push(Event {
                data,
                ..event.clone()
            });
        }
        Ok(())
    }

    fn link_readings(
        &mut self,
        input_key: &InputKey,
        input: i64,
        chain: i64,
        step: &ParseStep,
        choices: &mut Choices,
    ) -> Result<()> {
        for link in &step.output.links {
            let pk = self.keys.get(Kind::Link, &link_key(link)?)?.pk;
            let mut merged = link.clone();
            if let Some(previous) = self.links.find(pk, input, chain) {
                let previous: Link = crate::encoding::unpack_json(&previous.body(
                    self.index,
                    Kind::LinkValue,
                    self.values,
                )?)?;
                merged = merge_link(merged, previous);
            }
            let value = self.resolver.value(self.index, Kind::LinkValue, &merged)?;
            self.links.put(
                pk,
                ValueMember {
                    input,
                    input_id: input_key.id.clone(),
                    chain,
                    route: input_key.route,
                    value_pk: value.key.pk,
                    hash: value.hash.to_vec(),
                    body: value.body.clone(),
                    event: None,
                },
            );
            choices.link_values.push(value);
        }
        Ok(())
    }

    /// Composes each touched message from each input's latest reading.
    fn choose_messages(&mut self, choices: &mut Choices) -> Result<()> {
        let Self {
            index,
            messages,
            shown_messages,
            message_values,
            values,
            ..
        } = self;
        for (pk, current) in messages.take_touched() {
            if current.is_empty() {
                shown_messages.insert(pk, None);
                choices.messages.insert(pk, None);
                continue;
            }
            // One reading is shown whole; several need every value to
            // compose them field group by field group.
            if current.len() > 1 {
                for member in &current {
                    if !message_values.contains_key(&member.hash) {
                        let value =
                            known_message(index, *values, &member.hash)?.ok_or_else(|| {
                                StoreError::Invalid("a message reading has no value".into())
                            })?;
                        message_values.insert(member.hash.clone(), value);
                    }
                }
            }
            let composition = if current.len() > 1 {
                let readings: Vec<_> = current
                    .iter()
                    .map(|m| compose::MessageReading {
                        input: &m.input_id,
                        route: m.route,
                        main: m.main,
                        completion: m.completion,
                        hash: &m.hash,
                        message: &message_values[&m.hash].0,
                    })
                    .collect();
                compose::message(&readings)
            } else {
                None
            };
            let (main, groups, composed) = match composition {
                Some(composition) => (
                    current[composition.main],
                    composition
                        .groups
                        .iter()
                        .map(|(group, n)| (*group, current[*n]))
                        .collect::<Vec<_>>(),
                    Some(composition.message),
                ),
                None => (current[0], Vec::new(), None),
            };
            let mut routes: Vec<_> = groups
                .iter()
                .map(|(group, m)| (*group, m.input, m.chain, m.hash.clone()))
                .collect();
            routes.sort();
            let shown = shown_messages.get(&pk).and_then(Option::as_ref);
            let layout = message_values
                .get(&main.hash)
                .map(|(_, prepared)| crate::versions::message_layout(&prepared.projection))
                .transpose()?;
            let changed = shown.is_none_or(|shown| {
                shown.hash != main.hash
                    || shown.route != main.route
                    || shown.session != main.session
                    || shown.routes != routes
                    || shown
                        .layout
                        .as_ref()
                        .zip(layout.as_ref())
                        .is_some_and(|(before, after)| before != after)
            });
            if changed && !message_values.contains_key(&main.hash) {
                let value = known_message(index, *values, &main.hash)?.ok_or_else(|| {
                    StoreError::Invalid("a selected message version has no value".into())
                })?;
                message_values.insert(main.hash.clone(), value);
            }
            let (message, prepared) = if changed {
                let (version, mut prepared) = message_values[&main.hash].clone();
                let message = composed.unwrap_or(version);
                prepared.time = crate::encoding::time_bytes(message.at, message.time_evidence())?;
                (Some(message), Some(prepared))
            } else {
                (None, None)
            };
            let layout = message_values
                .get(&main.hash)
                .map(|(_, prepared)| crate::versions::message_layout(&prepared.projection))
                .transpose()?
                .or_else(|| shown.and_then(|shown| shown.layout.clone()));
            shown_messages.insert(
                pk,
                Some(Shown {
                    hash: main.hash.clone(),
                    route: main.route,
                    session: main.session,
                    layout,
                    routes: routes.clone(),
                }),
            );
            choices.messages.insert(
                pk,
                Some(MessageChoice {
                    input: main.input,
                    chain: main.chain,
                    start: main.start,
                    end: main.end,
                    hash: main.hash.clone(),
                    session: main.session,
                    route: main.route,
                    order: (main.input, (main.seq, main.position)),
                    routes: routes
                        .into_iter()
                        .map(|(group, input, chain, _)| (group, input, chain))
                        .collect(),
                    changed,
                    message,
                    prepared,
                }),
            );
        }
        Ok(())
    }

    fn choose_documents(&mut self, choices: &mut Choices) -> Result<()> {
        let Self {
            index,
            documents,
            shown_documents,
            document_values,
            values,
            ..
        } = self;
        for (pk, current) in documents.take_touched() {
            let readings: Vec<_> = current
                .iter()
                .map(|m| compose::DocumentReading {
                    input: &m.input_id,
                    time: m.time,
                    route: m.route,
                    hash: &m.hash,
                })
                .collect();
            let Some(winner) = compose::document(&readings).map(|n| current[n]) else {
                shown_documents.insert(pk, None);
                choices.documents.insert(pk, None);
                continue;
            };
            let changed = shown_documents.get(&pk).and_then(Option::as_ref) != Some(&winner.hash);
            if changed && !document_values.contains_key(&winner.hash) {
                let prior = match values {
                    Some(values) => values.document(&winner.hash)?,
                    None => None,
                };
                let value = match prior {
                    Some(value) => value,
                    None => {
                        let document: polylogue_model::Document = crate::versions::read(
                            index,
                            crate::versions::VersionKind::Document,
                            &winner.hash,
                        )?;
                        let prepared = PreparedDocument::new(&document)?;
                        (document, prepared)
                    }
                };
                document_values.insert(winner.hash.clone(), value);
            }
            let (document, prepared) = if changed {
                let (document, prepared) = document_values[&winner.hash].clone();
                (Some(document), Some(prepared))
            } else {
                (None, None)
            };
            shown_documents.insert(pk, Some(winner.hash.clone()));
            choices.documents.insert(
                pk,
                Some(DocumentChoice {
                    input: winner.input,
                    chain: winner.chain,
                    route: winner.route,
                    start: winner.start,
                    end: winner.end,
                    changed,
                    hash: winner.hash.clone(),
                    document,
                    prepared,
                }),
            );
        }
        Ok(())
    }

    fn choose_events(&mut self, choices: &mut Choices) -> Result<()> {
        let (index, values) = (self.index, self.values);
        for (pk, current) in self.events.take_touched() {
            let readings: Vec<_> = current
                .iter()
                .map(|m| (m.route, m.hash.as_slice(), m.input_id.as_str()))
                .collect();
            let winner = compose::value(&readings).map(|n| current[n]);
            let mut choice = winner
                .map(|member| member.event_source(index, pk, values))
                .transpose()?;
            if let Some(choice) = &mut choice
                && choice.body.is_empty()
            {
                choice.body =
                    winner
                        .expect("event winner")
                        .body(index, Kind::EventValue, values)?;
            }
            choices.events.insert(pk, choice);
        }
        Ok(())
    }

    /// A link shows its preferred reading's input; its optional coordinates
    /// and evidence merge over every reading.
    fn choose_links(&mut self, choices: &mut Choices) -> Result<()> {
        let (index, values) = (self.index, self.values);
        let touched: Vec<_> = self
            .links
            .take_touched()
            .into_iter()
            .map(|(pk, current)| {
                let readings: Vec<_> = current
                    .iter()
                    .map(|m| (m.route, m.hash.as_slice(), m.input_id.as_str()))
                    .collect();
                (
                    pk,
                    compose::value(&readings).map(|n| (current[n].input, current[n].chain)),
                )
            })
            .collect();
        for (pk, winner) in touched {
            let Some((input, chain)) = winner else {
                choices.links.insert(pk, None);
                continue;
            };
            let members = &self.links.members[&pk];
            let winner = members
                .iter()
                .find(|m| m.input == input && m.chain == chain)
                .expect("link winner");
            let mut link: Link =
                crate::encoding::unpack_json(&winner.body(index, Kind::LinkValue, values)?)?;
            for member in members {
                let other =
                    crate::encoding::unpack_json(&member.body(index, Kind::LinkValue, values)?)?;
                link = merge_link(link, other);
            }
            choices
                .links
                .insert(pk, Some(LinkChoice { input, chain, link }));
        }
        Ok(())
    }
}

pub(super) fn resolve(
    resolver: &Resolver,
    index: &Connection,
    batches: &mut [&mut Batch],
    chains: &[i64],
    keys: &ResolvedKeys,
) -> Result<()> {
    let mut projection = Projection::new(resolver, index, keys, None)?;
    for (batch, &chain) in batches.iter_mut().zip(chains) {
        for step in &mut batch.parse {
            projection.step(&batch.input, chain, step)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct EventChoice {
    pub admitted: Option<i64>,
    pub kind: String,
    pub fields: Vec<u8>,
    pub imported: String,
    pub value_pk: i64,
    pub input: i64,
    pub chain: i64,
    pub session: Option<i64>,
    pub seq: Option<i64>,
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub after: Option<String>,
    pub at: Option<i64>,
    pub body: Vec<u8>,
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct LinkChoice {
    pub input: i64,
    pub chain: i64,
    pub link: Link,
}

impl ValueMember {
    fn event_source(
        &self,
        index: &Connection,
        pk: i64,
        values: Option<&dyn ValueSource>,
    ) -> Result<EventChoice> {
        let mut event = self
            .event
            .clone()
            .ok_or_else(|| StoreError::Invalid("event membership has no fields".into()))?;
        if !event.fields.is_empty() {
            return Ok(event);
        }
        if let Some(resolved) = values
            .map(|source| source.event_source(pk, self.input, self.chain))
            .transpose()?
            .flatten()
        {
            return Ok(resolved);
        }
        let (fields, imported): (Vec<u8>, String) = index.query_row(
            "SELECT fields,imported_provenance FROM event_source WHERE event_pk=?1 AND input_pk=?2 AND chain=?3",
            params![pk,self.input,self.chain], |row| Ok((row.get(0)?,row.get(1)?)))?;
        let canonical: Event = crate::encoding::unpack_json(&fields)?;
        event.kind = canonical.kind;
        event.fields = fields;
        event.imported = imported;
        Ok(event)
    }
    fn body(
        &self,
        index: &Connection,
        kind: Kind,
        values: Option<&dyn ValueSource>,
    ) -> Result<Vec<u8>> {
        if !self.body.is_empty() {
            return Ok(self.body.clone());
        }
        if let Some(body) = values
            .map(|values| values.assertion(kind, self.value_pk))
            .transpose()?
            .flatten()
        {
            return Ok(body);
        }
        Ok(index.query_row(
            &format!("SELECT body FROM {} WHERE pk=?1", kind.table()),
            [self.value_pk],
            |r| r.get(0),
        )?)
    }
}

// A relation's optional coordinates and evidence merge independently of which
// contributor supplies its source membership.
fn merge_link(mut link: Link, other: Link) -> Link {
    link.at = link.at.into_iter().chain(other.at).min();
    link.call_id = link.call_id.into_iter().chain(other.call_id).min();
    link.evidence = link.evidence.min(other.evidence);
    link
}
