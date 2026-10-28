//! Source membership overlays for one worker snapshot. Payloads are decoded only
//! for versions the batch will show; all other readings remain shared by hash.
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
    pub message: Message,
    pub prepared: PreparedMessage,
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct DocumentChoice {
    pub input: i64,
    pub chain: i64,
    pub start: i64,
    pub end: i64,
    pub rank: i64,
    pub document: polylogue_model::Document,
    pub prepared: PreparedDocument,
}
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Choices {
    pub messages: BTreeMap<i64, Option<MessageChoice>>,
    pub documents: BTreeMap<i64, Option<DocumentChoice>>,
    pub sessions: BTreeMap<i64, Session>,
    pub message_mains: Vec<bool>,
    pub keep_messages: Vec<bool>,
    pub keep_documents: Vec<bool>,
    pub events: BTreeMap<i64, Option<EventChoice>>,
    pub links: BTreeMap<i64, Option<LinkChoice>>,
    pub event_values: Vec<PackedValue>,
    pub link_values: Vec<PackedValue>,
    pub live_events: Vec<Event>,
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
    time: Option<i64>,
}
impl Source {
    fn message_order(
        &self,
    ) -> (
        std::cmp::Reverse<bool>,
        std::cmp::Reverse<bool>,
        i64,
        &[u8],
        &str,
        std::cmp::Reverse<i64>,
    ) {
        (
            std::cmp::Reverse(self.main),
            std::cmp::Reverse(self.complete),
            self.rank,
            &self.hash,
            &self.id,
            std::cmp::Reverse(self.chain),
        )
    }
    fn document_order(
        &self,
    ) -> (
        std::cmp::Reverse<Option<i64>>,
        i64,
        std::cmp::Reverse<&[u8]>,
        &str,
        std::cmp::Reverse<i64>,
    ) {
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
    fn message(&self, hash: &[u8]) -> Result<Option<(Message, PreparedMessage)>>;
    fn document(
        &self,
        hash: &[u8],
    ) -> Result<Option<(polylogue_model::Document, PreparedDocument)>>;
}

pub(in crate::writer) struct Projection<'a> {
    resolver: &'a Resolver,
    index: &'a Connection,
    keys: &'a ResolvedKeys,
    messages: BTreeMap<i64, Vec<Source>>,
    documents: BTreeMap<i64, Vec<Source>>,
    message_values: BTreeMap<Vec<u8>, (Message, PreparedMessage)>,
    document_values: BTreeMap<Vec<u8>, (polylogue_model::Document, PreparedDocument)>,
    assertions: Assertions,
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
        let mut messages = BTreeMap::<i64, Vec<Source>>::new();
        let mut documents = BTreeMap::<i64, Vec<Source>>::new();
        let mut message_values = BTreeMap::<Vec<u8>, (Message, PreparedMessage)>::new();
        let mut document_values =
            BTreeMap::<Vec<u8>, (polylogue_model::Document, PreparedDocument)>::new();
        for (key, row) in &keys.rows {
            if !matches!(key.kind, Kind::Message | Kind::Document) {
                continue;
            }
            let message = key.kind == Kind::Message;
            let sql = if message {
                "SELECT s.input_pk,i.id,s.chain,s.source_start,s.source_end,s.session_pk,s.main,s.complete,s.arrival_rank,s.seq,s.record_position,s.version_hash,NULL FROM message_source s JOIN input_ref i ON i.pk=s.input_pk WHERE message_pk=?1"
            } else {
                "SELECT s.input_pk,i.id,s.chain,s.source_start,s.source_end,0,0,1,s.arrival_rank,0,0,s.version_hash,s.source_time FROM document_source s JOIN input_ref i ON i.pk=s.input_pk WHERE document_pk=?1"
            };
            let values = index
                .prepare(sql)?
                .query_map([row.pk], |r| {
                    Ok(Source {
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
                        time: r.get(12)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            if message {
                messages.insert(row.pk, values);
            } else {
                documents.insert(row.pk, values);
            }
        }
        let mut assertions = Assertions::load(index, keys)?;
        Ok(Self {
            resolver,
            index,
            keys,
            messages,
            documents,
            message_values,
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
            documents,
            message_values,
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
            if !keep {
                choices.message_mains.push(false);
                continue;
            }
            let main=match session_facts.get(&(session,input,chain)) {
                    Some(facts)=>facts.kind==polylogue_model::SessionKind::Main,
                    None if reset_inputs.contains(&(input,chain))=>true,
                    None=>index.query_row("SELECT coalesce((SELECT json_extract(facts,'$.kind') FROM session_record WHERE session_pk=?1 AND input_pk=?2 AND chain=?3),'main')",params![session,input,chain],|r|r.get::<_,String>(0))?=="main",
                };
            choices.message_mains.push(main);
            let (seq, position) = match step.order_from {
                Some(from) => ((from + i as u64) as i64, 0),
                None if step.repeats_whole_conversations => (i as i64, 0),
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
            touched_messages.insert(pk);
            message_values.insert(
                prepared.hash.to_vec(),
                (row.message.clone(), prepared.clone()),
            );
        }
        for (row, prepared) in step.output.documents.iter().zip(&step.prepared.documents) {
            let pk = keys
                .get(Kind::Document, row.document.id.to_string().as_bytes())?
                .pk;
            let sources = documents.entry(pk).or_default();
            let previous = sources
                .iter()
                .position(|s| s.input == input && s.chain == chain);
            let time = row.document.source_time.map(Timestamp::unix_ms);
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
                if !message_values.contains_key(&source.hash) {
                    let prior = match values {
                        Some(values) => values.message(&source.hash)?,
                        None => None,
                    };
                    let (message, prepared) = match prior {
                        Some(value) => value,
                        None => {
                            let message: Message = crate::versions::read(
                                index,
                                crate::versions::VersionKind::Message,
                                &source.hash,
                            )?;
                            let prepared = PreparedMessage::new(&message)?;
                            (message, prepared)
                        }
                    };
                    message_values.insert(source.hash.clone(), (message, prepared));
                }
                let (message, prepared) = message_values[&source.hash].clone();
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
                    message,
                    prepared,
                })
            } else {
                None
            };
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
                if !document_values.contains_key(&source.hash) {
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
                let (document, prepared) = document_values[&source.hash].clone();
                Some(DocumentChoice {
                    input: source.input,
                    chain: source.chain,
                    start: source.start,
                    end: source.end,
                    rank: source.rank,
                    document,
                    prepared,
                })
            } else {
                None
            };
            choices.documents.insert(pk, choice);
        }
        assertions.step(resolver, index, keys, &input_key, chain, step, &mut choices)?;
        step.prepared.choices = Some(choices);
        if values.is_some() {
            message_values.clear();
            document_values.clear();
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
    body: Vec<u8>,
    event: Option<EventChoice>,
}

struct Assertions {
    events: BTreeMap<i64, Vec<Assertion>>,
    links: BTreeMap<i64, Vec<Assertion>>,
}
impl Assertions {
    fn load(index: &Connection, keys: &ResolvedKeys) -> Result<Self> {
        let mut assertions = Self {
            events: BTreeMap::new(),
            links: BTreeMap::new(),
        };
        for (key, row) in &keys.rows {
            match key.kind {
                Kind::Event => {
                    let values=index.prepare("SELECT s.input_pk,i.id,s.chain,s.arrival_rank,s.value_pk,s.session_pk,s.seq,s.source_start,s.source_end,s.after_message_id,s.at FROM event_source s JOIN input_ref i ON i.pk=s.input_pk WHERE s.event_pk=?1")?.query_map([row.pk],|r| {let body=Vec::new();Ok(Assertion {input:r.get(0)?,id:r.get(1)?,chain:r.get(2)?,rank:r.get(3)?,body:body.clone(),event:Some(EventChoice {value_pk:r.get(4)?,input:r.get(0)?,chain:r.get(2)?,session:r.get(5)?,seq:r.get(6)?,start:r.get(7)?,end:r.get(8)?,after:r.get(9)?,at:r.get(10)?,body})})})?.collect::<rusqlite::Result<Vec<_>>>()?;
                    assertions.events.insert(row.pk, values);
                }
                Kind::Link => {
                    let values=index.prepare("SELECT s.input_pk,i.id,s.chain,s.arrival_rank,v.body FROM link_source s JOIN input_ref i ON i.pk=s.input_pk JOIN link_value v ON v.pk=s.value_pk WHERE s.link_pk=?1")?.query_map([row.pk],|r|Ok(Assertion {input:r.get(0)?,id:r.get(1)?,chain:r.get(2)?,rank:r.get(3)?,body:r.get(4)?,event:None}))?.collect::<rusqlite::Result<Vec<_>>>()?;
                    assertions.links.insert(row.pk, values);
                }
                _ => {}
            }
        }
        Ok(assertions)
    }
    fn step(
        &mut self,
        resolver: &Resolver,
        index: &Connection,
        keys: &ResolvedKeys,
        input: &InputKey,
        chain: i64,
        step: &ParseStep,
        choices: &mut Choices,
    ) -> Result<()> {
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
        let mut occurrences = BTreeMap::<(u64, u64), u64>::new();
        for (event, packed) in step.output.events.iter().zip(&step.prepared.events) {
            let occurrence = if event.session.is_none() {
                let next = occurrences
                    .entry(event.range.unwrap_or_default())
                    .or_default();
                let value = *next;
                *next += 1;
                value
            } else {
                0
            };
            let id = event_id(&input.id, chain, event, &packed.0, occurrence)?;
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
                .or(event.range.map(|r| r.0 as i64))
                .or(step
                    .repeats_whole_conversations
                    .then_some(step.start as i64));
            let choice = EventChoice {
                value_pk: value.key.pk,
                input: input_pk,
                chain,
                session,
                seq,
                start: event.range.map(|r| r.0 as i64),
                end: event.range.map(|r| r.1 as i64),
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
                body: value.body.clone(),
                event: Some(choice),
            });
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
                let previous: Link = crate::encoding::unpack_json(&previous.body)?;
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
            let mut choice = winner.and_then(|s| s.event.clone());
            if let Some(choice) = &mut choice {
                if choice.body.is_empty() {
                    choice.body = index.query_row(
                        "SELECT body FROM event_value WHERE pk=?1",
                        [choice.value_pk],
                        |r| r.get(0),
                    )?;
                }
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
                let mut link: Link = crate::encoding::unpack_json(&winner.body)?;
                for source in sources {
                    let other: Link = crate::encoding::unpack_json(&source.body)?;
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
        Ok(())
    }
}
