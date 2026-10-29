//! Source membership overlays for one worker snapshot. Touched message versions
//! merge compatible block observations before source selection; values stay shared by hash.
use super::*;
use crate::encoding::{PreparedDocument, PreparedMessage};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct MessageChoice {
    pub input: i64,
    pub chain: i64,
    pub start: i64,
    pub end: i64,
    pub complete: bool,
    pub rank: i64,
    pub hash: Vec<u8>,
    pub session: i64,
    pub order: (i64, (i64, i64)),
    pub changed: bool,
    pub message: Option<Message>,
    pub prepared: Option<PreparedMessage>,
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct DocumentChoice {
    pub input: i64,
    pub chain: i64,
    pub start: i64,
    pub end: i64,
    pub rank: i64,
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
#[derive(Clone)]
struct Source {
    input: i64,
    id: String,
    chain: i64,
    start: i64,
    end: i64,
    session: i64,
    main: bool,
    complete: bool,
    rank: i64,
    seq: i64,
    position: i64,
    hash: Vec<u8>,
    time: Option<Timestamp>,
}
type MessagePriority<'a> = (
    std::cmp::Reverse<bool>,
    std::cmp::Reverse<bool>,
    i64,
    &'a [u8],
    &'a str,
    std::cmp::Reverse<i64>,
);
type DocumentPriority<'a> = (
    std::cmp::Reverse<Option<Timestamp>>,
    i64,
    std::cmp::Reverse<&'a [u8]>,
    &'a str,
    std::cmp::Reverse<i64>,
);
impl Source {
    fn message_order(&self) -> MessagePriority<'_> {
        (
            std::cmp::Reverse(self.main),
            std::cmp::Reverse(self.complete),
            self.rank,
            &self.hash,
            &self.id,
            std::cmp::Reverse(self.chain),
        )
    }
    fn document_order(&self) -> DocumentPriority<'_> {
        (
            std::cmp::Reverse(self.time),
            self.rank,
            std::cmp::Reverse(&self.hash),
            &self.id,
            std::cmp::Reverse(self.chain),
        )
    }
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

pub(in crate::writer) struct Projection<'a> {
    resolver: &'a Resolver,
    index: &'a Connection,
    keys: &'a ResolvedKeys,
    messages: BTreeMap<i64, Vec<Source>>,
    shown_messages: BTreeMap<i64, Option<ShownMessage>>,
    shown_documents: BTreeMap<i64, Option<Vec<u8>>>,
    documents: BTreeMap<i64, Vec<Source>>,
    message_values: BTreeMap<Vec<u8>, (Message, PreparedMessage)>,
    message_layouts: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    document_values: BTreeMap<Vec<u8>, (polylogue_model::Document, PreparedDocument)>,
    assertions: Assertions,
    session_facts: BTreeMap<(i64, i64, i64), Session>,
    reset_inputs: BTreeSet<(i64, i64)>,
    values: Option<&'a dyn ValueSource>,
}
type ShownMessage = (Vec<u8>, i64, Option<Vec<u8>>);
impl<'a> Projection<'a> {
    pub(in crate::writer) fn new(
        resolver: &'a Resolver,
        index: &'a Connection,
        keys: &'a ResolvedKeys,
        values: Option<&'a dyn ValueSource>,
    ) -> Result<Self> {
        let mut messages = BTreeMap::<i64, Vec<Source>>::new();
        let mut shown_messages = BTreeMap::new();
        let mut shown_documents = BTreeMap::new();
        let mut documents = BTreeMap::<i64, Vec<Source>>::new();
        let message_values = BTreeMap::<Vec<u8>, (Message, PreparedMessage)>::new();
        let mut message_layouts = BTreeMap::new();
        let document_values =
            BTreeMap::<Vec<u8>, (polylogue_model::Document, PreparedDocument)>::new();
        for (key, row) in &keys.rows {
            if !matches!(key.kind, Kind::Message | Kind::Document) {
                continue;
            }
            let message = key.kind == Kind::Message;
            if !row.existing {
                if message {
                    messages.insert(row.pk, Vec::new());
                } else {
                    documents.insert(row.pk, Vec::new());
                }
                continue;
            }
            let sql = if message {
                "SELECT s.input_pk,i.id,s.chain,s.source_start,s.source_end,s.session_pk,s.main,s.complete,s.arrival_rank,s.seq,s.record_position,s.version_hash,NULL,p.layout FROM message_source s JOIN input_ref i ON i.pk=s.input_pk LEFT JOIN message_projection p ON p.version_hash=s.version_hash WHERE message_pk=?1"
            } else {
                "SELECT s.input_pk,i.id,s.chain,s.source_start,s.source_end,0,0,1,s.arrival_rank,0,0,s.version_hash,s.time FROM document_source s JOIN input_ref i ON i.pk=s.input_pk WHERE document_pk=?1"
            };
            if message {
                shown_messages.insert(row.pk,index.query_row("SELECT s.version_hash,m.session_pk,p.layout FROM message_source s JOIN message m ON m.pk=s.message_pk LEFT JOIN message_projection p ON p.version_hash=s.version_hash WHERE s.message_pk=?1 AND s.shown=1",[row.pk],|r|Ok((r.get::<_,Vec<u8>>(0)?,r.get::<_,i64>(1)?,r.get::<_,Option<Vec<u8>>>(2)?))).optional()?);
            } else {
                shown_documents.insert(
                    row.pk,
                    index
                        .query_row(
                            "SELECT version_hash FROM document WHERE pk=?1",
                            [row.pk],
                            |r| r.get::<_, Vec<u8>>(0),
                        )
                        .optional()?,
                );
            }
            let values = index
                .prepare(sql)?
                .query_map([row.pk], |r| {
                    Ok((
                        Source {
                            input: r.get(0)?,
                            id: r.get(1)?,
                            chain: r.get(2)?,
                            start: r.get(3)?,
                            end: r.get(4)?,
                            session: r.get(5)?,
                            main: r.get(6)?,
                            complete: r.get(7)?,
                            rank: r.get(8)?,
                            seq: r.get(9)?,
                            position: r.get(10)?,
                            hash: r.get(11)?,
                            time: if message {
                                None
                            } else {
                                crate::encoding::read_time(r, 12)?.0
                            },
                        },
                        if message {
                            r.get::<_, Option<Vec<u8>>>(13)?
                        } else {
                            None
                        },
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            let values = values
                .into_iter()
                .map(|(source, layout)| {
                    if message {
                        message_layouts.insert(source.hash.clone(), layout);
                    }
                    source
                })
                .collect();
            if message {
                messages.insert(row.pk, values);
            } else {
                documents.insert(row.pk, values);
            }
        }
        let assertions = Assertions::load(index, keys)?;
        Ok(Self {
            resolver,
            index,
            keys,
            messages,
            shown_messages,
            shown_documents,
            documents,
            message_values,
            message_layouts,
            document_values,
            assertions,
            session_facts: BTreeMap::new(),
            reset_inputs: BTreeSet::new(),
            values,
        })
    }
    pub(in crate::writer) fn step(
        &mut self,
        input_key: &InputKey,
        chain: i64,
        step: &mut ParseStep,
    ) -> Result<()> {
        let Self {
            resolver,
            index,
            keys,
            messages,
            shown_messages,
            shown_documents,
            documents,
            message_values,
            message_layouts,
            document_values,
            assertions,
            session_facts,
            reset_inputs,
            values,
        } = self;
        let input = keys.get(Kind::Input, input_key.id.as_bytes())?.pk;
        let mut choices = Choices::default();
        let mut touched_messages = BTreeSet::new();
        let mut touched_documents = BTreeSet::new();
        if step.reset {
            reset_inputs.insert((input, chain));
            touched_messages.extend(
                messages
                    .iter()
                    .filter(|(_, sources)| {
                        sources.iter().any(|s| s.input == input && s.chain == chain)
                    })
                    .map(|(pk, _)| *pk),
            );
            touched_documents.extend(
                documents
                    .iter()
                    .filter(|(_, sources)| {
                        sources.iter().any(|s| s.input == input && s.chain == chain)
                    })
                    .map(|(pk, _)| *pk),
            );
            for sources in messages.values_mut().chain(documents.values_mut()) {
                sources.retain(|s| s.input != input || s.chain != chain);
            }
            session_facts.retain(|(_, i, c), _| *i != input || *c != chain);
        }
        for session in &step.output.sessions {
            let pk = keys
                .get(Kind::Session, session.id.to_string().as_bytes())?
                .pk;
            let key = (pk, input, chain);
            if let std::collections::btree_map::Entry::Vacant(slot) = session_facts.entry(key)
                && let Some(session) = values
                    .map(|values| values.session(pk, input, chain))
                    .transpose()?
                    .flatten()
            {
                slot.insert(session);
            }
            if !session_facts.contains_key(&key) && !reset_inputs.contains(&(input, chain)) {
                let body:Option<String>=index.query_row("SELECT facts FROM session_record WHERE session_pk=?1 AND input_pk=?2 AND chain=?3",params![pk,input,chain],|r|r.get(0)).optional()?;
                if let Some(body) = body {
                    session_facts.insert(key, serde_json::from_str(&body)?);
                }
            }
            let merged = session_facts.entry(key).or_insert_with(|| session.clone());
            merged.merge(session);
            choices.sessions.insert(pk, merged.clone());
            let main = merged.kind == polylogue_model::SessionKind::Main;
            for (message_pk, sources) in messages.iter_mut() {
                for source in sources
                    .iter_mut()
                    .filter(|s| s.session == pk && s.input == input && s.chain == chain)
                {
                    if source.main != main {
                        touched_messages.insert(*message_pk);
                    }
                    source.main = main;
                }
            }
        }
        let mut positions = BTreeMap::<u64, i64>::new();
        for (i, (row, prepared)) in step
            .output
            .messages
            .iter()
            .zip(&step.prepared.messages)
            .enumerate()
        {
            let pk = keys
                .get(Kind::Message, row.message.id.to_string().as_bytes())?
                .pk;
            let session = keys
                .get(Kind::Session, row.message.session.to_string().as_bytes())?
                .pk;
            let sources = messages.entry(pk).or_default();
            let previous = sources
                .iter()
                .position(|s| s.input == input && s.chain == chain);
            let keep = !previous.is_some_and(|n| {
                if step.repeats_whole_conversations {
                    (row.offset as i64) < sources[n].start
                } else {
                    (row.offset as i64) > sources[n].start
                }
            });
            choices.keep_messages.push(keep);
            if !keep && !previous.is_some_and(|n| sources[n].hash.as_slice() == prepared.hash) {
                choices.message_mains.push(false);
                continue;
            }
            let incoming_layout = crate::versions::message_layout(&prepared.projection)?;
            let prior_layout = message_layouts.get(prepared.hash.as_slice());
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
                match message_values.get(prepared.hash.as_slice()) {
                    Some(value) => Some(value.clone()),
                    None => known_message(index, *values, &prepared.hash)?,
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
            message_layouts.insert(prepared.hash.to_vec(), Some(layout));
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
            message_values.insert(prepared.hash.to_vec(), merged);
            touched_messages.insert(pk);
            if !keep {
                choices.message_mains.push(false);
                continue;
            }
            if let std::collections::btree_map::Entry::Vacant(slot) =
                session_facts.entry((session, input, chain))
                && let Some(facts) = values
                    .map(|values| values.session(session, input, chain))
                    .transpose()?
                    .flatten()
            {
                slot.insert(facts);
            }
            let main=match session_facts.get(&(session,input,chain)) {
                    Some(facts)=>facts.kind==polylogue_model::SessionKind::Main,
                    None if reset_inputs.contains(&(input,chain))=>true,
                    None=>index.query_row("SELECT coalesce((SELECT json_extract(facts,'$.kind') FROM session_record WHERE session_pk=?1 AND input_pk=?2 AND chain=?3),'main')",params![session,input,chain],|r|r.get::<_,String>(0))?=="main",
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
            let source = Source {
                input,
                id: input_key.id.clone(),
                chain,
                start: row.offset as i64,
                end: row.end.max(row.offset) as i64,
                session,
                main,
                complete: !(row.message.role == Role::Assistant
                    && row.message.stop_reason.is_none()),
                rank: i64::from(input_key.arrival_rank),
                seq,
                position,
                hash: prepared.hash.to_vec(),
                time: None,
            };
            if let Some(n) = previous {
                sources[n] = source;
            } else {
                sources.push(source);
            }
        }
        for (row, prepared) in step.output.documents.iter().zip(&step.prepared.documents) {
            let pk = keys
                .get(Kind::Document, row.document.id.to_string().as_bytes())?
                .pk;
            let sources = documents.entry(pk).or_default();
            let previous = sources
                .iter()
                .position(|s| s.input == input && s.chain == chain);
            let time = row.document.source_time;
            let rank = i64::from(input_key.arrival_rank);
            let keep = !previous.is_some_and(|n| {
                (time, std::cmp::Reverse(rank), prepared.hash.as_slice())
                    < (
                        sources[n].time,
                        std::cmp::Reverse(sources[n].rank),
                        sources[n].hash.as_slice(),
                    )
            });
            choices.keep_documents.push(keep);
            if !keep {
                continue;
            }
            let source = Source {
                input,
                id: input_key.id.clone(),
                chain,
                start: row.offset as i64,
                end: row.end as i64,
                session: 0,
                main: false,
                complete: true,
                rank,
                seq: 0,
                position: 0,
                hash: prepared.hash.to_vec(),
                time,
            };
            if let Some(n) = previous {
                sources[n] = source;
            } else {
                sources.push(source);
            }
            touched_documents.insert(pk);
            document_values.insert(
                prepared.hash.to_vec(),
                (row.document.clone(), prepared.clone()),
            );
        }
        for (&pk, sources) in messages
            .iter()
            .filter(|(pk, _)| touched_messages.contains(pk))
        {
            let current: Vec<_> = sources
                .iter()
                .filter(|s| {
                    !sources
                        .iter()
                        .any(|n| n.input == s.input && n.chain > s.chain)
                })
                .collect();
            let winner = current
                .iter()
                .min_by(|a, b| a.message_order().cmp(&b.message_order()));
            let choice = if let Some(source) = winner {
                let order = current
                    .iter()
                    .min_by_key(|s| (&s.id, s.seq, s.position))
                    .expect("winner has order");
                let shown = shown_messages.get(&pk).and_then(Option::as_ref);
                let layout = message_values
                    .get(&source.hash)
                    .map(|(_, prepared)| crate::versions::message_layout(&prepared.projection))
                    .transpose()?;
                let changed = shown.map(|(hash, session, _)| (hash, *session))
                    != Some((&source.hash, source.session))
                    || shown
                        .and_then(|(_, _, layout)| layout.as_ref())
                        .zip(layout.as_ref())
                        .is_some_and(|(before, after)| before != after);
                if changed && !message_values.contains_key(&source.hash) {
                    let value = known_message(index, *values, &source.hash)?.ok_or_else(|| {
                        StoreError::Invalid("a selected message version has no value".into())
                    })?;
                    message_values.insert(source.hash.clone(), value);
                }
                let (message, prepared) = if changed {
                    let (message, prepared) = message_values[&source.hash].clone();
                    (Some(message), Some(prepared))
                } else {
                    (None, None)
                };
                let layout = message_values
                    .get(&source.hash)
                    .map(|(_, prepared)| crate::versions::message_layout(&prepared.projection))
                    .transpose()?
                    .or_else(|| shown.and_then(|(_, _, layout)| layout.clone()));
                shown_messages.insert(pk, Some((source.hash.clone(), source.session, layout)));
                Some(MessageChoice {
                    input: source.input,
                    chain: source.chain,
                    start: source.start,
                    end: source.end,
                    complete: source.complete,
                    rank: source.rank,
                    hash: source.hash.clone(),
                    session: source.session,
                    order: (order.input, (order.seq, order.position)),
                    changed,
                    message,
                    prepared,
                })
            } else {
                None
            };
            if choice.is_none() {
                shown_messages.insert(pk, None);
            }
            choices.messages.insert(pk, choice);
        }
        for (&pk, sources) in documents
            .iter()
            .filter(|(pk, _)| touched_documents.contains(pk))
        {
            let choice = if let Some(source) = sources
                .iter()
                .min_by(|a, b| a.document_order().cmp(&b.document_order()))
            {
                let changed =
                    shown_documents.get(&pk).and_then(Option::as_ref) != Some(&source.hash);
                if changed && !document_values.contains_key(&source.hash) {
                    let prior = match values {
                        Some(values) => values.document(&source.hash)?,
                        None => None,
                    };
                    let (document, prepared) = match prior {
                        Some(value) => value,
                        None => {
                            let document: polylogue_model::Document = crate::versions::read(
                                index,
                                crate::versions::VersionKind::Document,
                                &source.hash,
                            )?;
                            let prepared = PreparedDocument::new(&document)?;
                            (document, prepared)
                        }
                    };
                    document_values.insert(source.hash.clone(), (document, prepared));
                }
                let (document, prepared) = if changed {
                    let (document, prepared) = document_values[&source.hash].clone();
                    (Some(document), Some(prepared))
                } else {
                    (None, None)
                };
                shown_documents.insert(pk, Some(source.hash.clone()));
                Some(DocumentChoice {
                    input: source.input,
                    chain: source.chain,
                    start: source.start,
                    end: source.end,
                    rank: source.rank,
                    changed,
                    hash: source.hash.clone(),
                    document,
                    prepared,
                })
            } else {
                None
            };
            if choice.is_none() {
                shown_documents.insert(pk, None);
            }
            choices.documents.insert(pk, choice);
        }
        assertions.step(
            AssertionStep {
                resolver,
                index,
                keys,
                input: input_key,
                chain,
                values: values.as_deref(),
            },
            step,
            &mut choices,
        )?;
        step.prepared.choices = Some(choices);
        if values.is_some() {
            message_values.clear();
            document_values.clear();
            session_facts.clear();
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
    pub kind: String,
    pub fields: Vec<u8>,
    pub imported: Vec<polylogue_model::bundle::EventSourceClaim>,
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
#[derive(Clone)]
struct Assertion {
    input: i64,
    id: String,
    chain: i64,
    rank: i64,
    value_pk: i64,
    body: Vec<u8>,
    event: Option<EventChoice>,
}

struct Assertions {
    events: BTreeMap<i64, Vec<Assertion>>,
    links: BTreeMap<i64, Vec<Assertion>>,
}
struct AssertionStep<'a> {
    resolver: &'a Resolver,
    index: &'a Connection,
    keys: &'a ResolvedKeys,
    input: &'a InputKey,
    chain: i64,
    values: Option<&'a dyn ValueSource>,
}
impl Assertions {
    fn load(index: &Connection, keys: &ResolvedKeys) -> Result<Self> {
        let mut assertions = Self {
            events: BTreeMap::new(),
            links: BTreeMap::new(),
        };
        for (key, row) in &keys.rows {
            if !row.existing {
                continue;
            }
            match key.kind {
                Kind::Event => {
                    let values=index.prepare("SELECT s.input_pk,i.id,s.chain,s.arrival_rank,s.value_pk,s.session_pk,s.seq,s.source_start,s.source_end,s.after_message_id,s.at FROM event_source s JOIN input_ref i ON i.pk=s.input_pk WHERE s.event_pk=?1")?.query_map([row.pk],|r| {let body=Vec::new();Ok(Assertion {input:r.get(0)?,id:r.get(1)?,chain:r.get(2)?,rank:r.get(3)?,value_pk:r.get(4)?,body:body.clone(),event:Some(EventChoice {kind:String::new(),fields:Vec::new(),imported:Vec::new(),value_pk:r.get(4)?,input:r.get(0)?,chain:r.get(2)?,session:r.get(5)?,seq:r.get(6)?,start:r.get(7)?,end:r.get(8)?,after:r.get(9)?,at:r.get(10)?,body})})})?.collect::<rusqlite::Result<Vec<_>>>()?;
                    assertions.events.insert(row.pk, values);
                }
                Kind::Link => {
                    let values=index.prepare("SELECT s.input_pk,i.id,s.chain,s.arrival_rank,s.value_pk FROM link_source s JOIN input_ref i ON i.pk=s.input_pk WHERE s.link_pk=?1")?.query_map([row.pk],|r|Ok(Assertion {input:r.get(0)?,id:r.get(1)?,chain:r.get(2)?,rank:r.get(3)?,value_pk:r.get(4)?,body:Vec::new(),event:None}))?.collect::<rusqlite::Result<Vec<_>>>()?;
                    assertions.links.insert(row.pk, values);
                }
                _ => {}
            }
        }
        Ok(assertions)
    }
    fn step(
        &mut self,
        context: AssertionStep<'_>,
        step: &ParseStep,
        choices: &mut Choices,
    ) -> Result<()> {
        let AssertionStep {
            resolver,
            index,
            keys,
            input,
            chain,
            values,
        } = context;
        let input_pk = keys.get(Kind::Input, input.id.as_bytes())?.pk;
        let rank = i64::from(input.arrival_rank);
        let mut touched_events = BTreeSet::new();
        let mut touched_links = BTreeSet::new();
        if step.reset {
            touched_events.extend(
                self.events
                    .iter()
                    .filter(|(_, sources)| {
                        sources
                            .iter()
                            .any(|s| s.input == input_pk && s.chain == chain)
                    })
                    .map(|(pk, _)| *pk),
            );
            touched_links.extend(
                self.links
                    .iter()
                    .filter(|(_, sources)| {
                        sources
                            .iter()
                            .any(|s| s.input == input_pk && s.chain == chain)
                    })
                    .map(|(pk, _)| *pk),
            );
            for sources in self.events.values_mut().chain(self.links.values_mut()) {
                sources.retain(|s| s.input != input_pk || s.chain != chain);
            }
        }
        let event_ids = step.event_identities(&input.id, chain)?;
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
            let pk = keys.get(Kind::Event, id.as_bytes())?.pk;
            let data: Json = crate::encoding::unpack_json(&packed.1)?;
            let value = resolver.value(index, Kind::EventValue, &data)?;
            let session = event
                .session
                .as_ref()
                .map(|id| {
                    keys.get(Kind::Session, id.to_string().as_bytes())
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
            if let Some(previous) = self.events.get(&pk).and_then(|sources| {
                sources.iter().find(|source| {
                    source.input == input_pk
                        && source.chain == chain
                        && source.value_pk == value.key.pk
                })
            }) {
                let previous = previous.event_source(index, pk, values)?;
                if previous.fields == fields {
                    original.extend(previous.imported);
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
                kind: event.kind.clone(),
                fields,
                imported: original,
                value_pk: value.key.pk,
                input: input_pk,
                chain,
                session,
                seq,
                start: local_range.map(|r| r.0 as i64),
                end: local_range.map(|r| r.1 as i64),
                after: event.after.as_ref().map(ToString::to_string),
                at: event.at.map(Timestamp::unix_ms),
                body: value.body.clone(),
            };
            touched_events.insert(pk);
            let sources = self.events.entry(pk).or_default();
            sources.retain(|s| s.input != input_pk || s.chain != chain);
            sources.push(Assertion {
                input: input_pk,
                id: input.id.clone(),
                chain,
                rank,
                value_pk: value.key.pk,
                body: value.body.clone(),
                event: Some(choice.clone()),
            });
            choices.event_ids.push(id);
            choices.event_sources.push((pk, choice));
            choices.event_values.push(value);
            choices.live_events.push(Event {
                data,
                ..event.clone()
            });
        }
        for link in &step.output.links {
            let pk = keys.get(Kind::Link, &link_key(link)?)?.pk;
            touched_links.insert(pk);
            let sources = self.links.entry(pk).or_default();
            let mut merged = link.clone();
            if let Some(previous) = sources
                .iter()
                .find(|s| s.input == input_pk && s.chain == chain)
            {
                let previous: Link = crate::encoding::unpack_json(&previous.body(
                    index,
                    Kind::LinkValue,
                    values,
                )?)?;
                merged.at = merged.at.into_iter().chain(previous.at).min();
                merged.call_id = merged.call_id.into_iter().chain(previous.call_id).min();
                merged.evidence = merged.evidence.min(previous.evidence);
            }
            let value = resolver.value(index, Kind::LinkValue, &merged)?;
            sources.retain(|s| s.input != input_pk || s.chain != chain);
            sources.push(Assertion {
                input: input_pk,
                id: input.id.clone(),
                chain,
                rank,
                value_pk: value.key.pk,
                body: value.body.clone(),
                event: None,
            });
            choices.link_values.push(value);
        }
        for (&pk, sources) in self
            .events
            .iter()
            .filter(|(pk, _)| touched_events.contains(pk))
        {
            let winner = sources
                .iter()
                .min_by_key(|s| (s.rank, &s.id, std::cmp::Reverse(s.chain)));
            let mut choice = winner
                .map(|source| source.event_source(index, pk, values))
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
        for (&pk, sources) in self
            .links
            .iter()
            .filter(|(pk, _)| touched_links.contains(pk))
        {
            let choice = if let Some(winner) = sources
                .iter()
                .min_by_key(|s| (s.rank, &s.id, std::cmp::Reverse(s.chain)))
            {
                let mut link: Link =
                    crate::encoding::unpack_json(&winner.body(index, Kind::LinkValue, values)?)?;
                for source in sources {
                    let other: Link = crate::encoding::unpack_json(&source.body(
                        index,
                        Kind::LinkValue,
                        values,
                    )?)?;
                    link.at = link.at.into_iter().chain(other.at).min();
                    link.call_id = link.call_id.into_iter().chain(other.call_id).min();
                }
                Some(LinkChoice {
                    input: winner.input,
                    chain: winner.chain,
                    link,
                })
            } else {
                None
            };
            choices.links.insert(pk, choice);
        }
        if values.is_some() {
            for source in self
                .events
                .values_mut()
                .chain(self.links.values_mut())
                .flatten()
            {
                source.body.clear();
                if let Some(event) = &mut source.event {
                    event.body.clear();
                    event.fields.clear();
                    event.kind.clear();
                    event.imported.clear();
                }
            }
        }
        Ok(())
    }
}

impl Assertion {
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
        event.imported = serde_json::from_str(&imported)?;
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
