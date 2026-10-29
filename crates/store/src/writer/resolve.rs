//! Worker-owned natural keys. A lease covers the batch until its commit result,
//! so a following worker opens its snapshot after overlapping work has settled.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Condvar;
pub(crate) mod versions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Kind {
    Input,
    Session,
    Message,
    Event,
    Document,
    Text,
    Link,
    EventValue,
    LinkValue,
}
impl Kind {
    fn table(self) -> &'static str {
        match self {
            Self::Input => "input_ref",
            Self::Session => "session",
            Self::Message => "message",
            Self::Event => "event",
            Self::Document => "document",
            Self::Text => "text",
            Self::Link => "link",
            Self::EventValue => "event_value",
            Self::LinkValue => "link_value",
        }
    }
    fn pk(self) -> &'static str {
        if self == Self::Text { "id" } else { "pk" }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Key {
    pub(super) kind: Kind,
    bytes: Vec<u8>,
}
impl Key {
    fn new(kind: Kind, bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            kind,
            bytes: bytes.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowKey {
    pub pk: i64,
    pub existing: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ResolvedKeys {
    pub(super) rows: BTreeMap<Key, RowKey>,
}
impl ResolvedKeys {
    pub(crate) fn get(&self, kind: Kind, bytes: &[u8]) -> Result<RowKey> {
        self.rows
            .get(&Key::new(kind, bytes))
            .copied()
            .ok_or_else(|| StoreError::Invalid(format!("unresolved {} key", kind.table())))
    }
}

struct Active {
    keys: BTreeSet<Key>,
    next: BTreeMap<Kind, i64>,
    // IDs outlive their payload owners: an older snapshot can resolve a value
    // after another group committed it and released its prepared bytes.
    values: BTreeMap<Key, (i64, std::sync::Weak<Reservation>)>,
}
pub(super) struct Resolver {
    archive: Archive,
    active: Mutex<Active>,
    available: Condvar,
    first: Option<Arc<crate::first_build::FirstBuild>>,
    #[cfg(test)]
    waiting: Mutex<Option<std::sync::mpsc::Sender<()>>>,
}
pub(super) struct Lease {
    resolver: Arc<Resolver>,
    keys: BTreeSet<Key>,
}
impl Drop for Lease {
    fn drop(&mut self) {
        let mut active = self.resolver.active.lock().expect("resolver lock");
        for key in &self.keys {
            active.keys.remove(key);
        }
        drop(active);
        self.resolver.available.notify_all();
    }
}

impl Resolver {
    #[cfg(test)]
    pub(super) fn observe_wait(&self, observer: std::sync::mpsc::Sender<()>) {
        *self.waiting.lock().expect("wait observer") = Some(observer);
    }
    pub(super) fn location(&self) -> &Archive {
        &self.archive
    }
    pub(super) fn new(
        archive: &Archive,
        index: &Connection,
        first: Option<Arc<crate::first_build::FirstBuild>>,
    ) -> Result<Arc<Self>> {
        let mut next = BTreeMap::new();
        for kind in [
            Kind::Input,
            Kind::Session,
            Kind::Message,
            Kind::Event,
            Kind::Document,
            Kind::Text,
            Kind::Link,
            Kind::EventValue,
            Kind::LinkValue,
        ] {
            let max: i64 = index.query_row(
                &format!(
                    "SELECT coalesce(max({}),0) FROM {}",
                    kind.pk(),
                    kind.table()
                ),
                [],
                |r| r.get(0),
            )?;
            next.insert(
                kind,
                max.checked_add(1)
                    .ok_or_else(|| StoreError::Invalid("row identity exhausted".into()))?,
            );
        }
        Ok(Arc::new(Self {
            archive: archive.clone(),
            active: Mutex::new(Active {
                keys: BTreeSet::new(),
                next,
                values: BTreeMap::new(),
            }),
            available: Condvar::new(),
            first,
            #[cfg(test)]
            waiting: Mutex::new(None),
        }))
    }

    pub(super) fn lease(self: &Arc<Self>, keys: BTreeSet<Key>) -> Lease {
        let mut active = self.active.lock().expect("resolver lock");
        while keys.iter().any(|key| active.keys.contains(key)) {
            #[cfg(test)]
            if let Some(waiting) = self.waiting.lock().expect("wait observer").as_ref() {
                let _ = waiting.send(());
            }
            active = self.available.wait(active).expect("resolver lock");
        }
        active.keys.extend(keys.iter().cloned());
        Lease {
            resolver: Arc::clone(self),
            keys,
        }
    }

    pub(super) fn allocate(&self, kind: Kind) -> Result<i64> {
        let mut active = self.active.lock().expect("resolver lock");
        let next = active.next.get_mut(&kind).expect("row allocator");
        let pk = *next;
        *next = next
            .checked_add(1)
            .ok_or_else(|| StoreError::Invalid("row identity exhausted".into()))?;
        Ok(pk)
    }

    pub(super) fn remember(&self, key: &Key, pk: i64) {
        if let Some(first) = &self.first {
            match key.kind {
                Kind::Message => {
                    first.add_message(std::str::from_utf8(&key.bytes).expect("message key"), pk)
                }
                Kind::Event => {
                    first.add_event(std::str::from_utf8(&key.bytes).expect("event key"), pk)
                }
                Kind::Text => {
                    first.add_text(key.bytes.as_slice().try_into().expect("text hash"), pk)
                }
                _ => {}
            }
        }
    }
    pub(super) fn find(&self, index: &Connection, key: &Key) -> Result<Option<i64>> {
        let first = self.first.as_deref();
        match key.kind {
            Kind::Message if first.is_some() => first
                .expect("first build")
                .message(index, std::str::from_utf8(&key.bytes).expect("string key")),
            Kind::Event if first.is_some() => first
                .expect("first build")
                .event(index, std::str::from_utf8(&key.bytes).expect("string key")),
            Kind::Text if first.is_some() => first
                .expect("first build")
                .text(index, key.bytes.as_slice().try_into().expect("text hash")),
            Kind::Link => {
                let (parent, child, kind, how): (String, String, String, String) =
                    serde_json::from_slice(&key.bytes)?;
                Ok(index.query_row("SELECT pk FROM link WHERE parent_id=?1 AND child_id=?2 AND kind=?3 AND how=?4",params![parent,child,kind,how],|r|r.get(0)).optional()?)
            }
            kind => {
                let column = if kind == Kind::Text { "hash" } else { "id" };
                let value = if kind == Kind::Text {
                    rusqlite::types::Value::Blob(key.bytes.clone())
                } else {
                    String::from_utf8(key.bytes.clone())
                        .expect("string key")
                        .into()
                };
                Ok(index
                    .query_row(
                        &format!(
                            "SELECT {} FROM {} WHERE {column}=?1",
                            kind.pk(),
                            kind.table()
                        ),
                        [value],
                        |r| r.get(0),
                    )
                    .optional()?)
            }
        }
    }
}

pub(super) fn link_key(link: &Link) -> Result<Vec<u8>> {
    let (kind, how) = link.kind.columns();
    Ok(serde_json::to_vec(&(
        link.parent.to_string(),
        link.child.to_string(),
        kind,
        how,
    ))?)
}

pub(super) fn event_id(
    input: &str,
    chain: i64,
    event: &Event,
    content_id: &str,
    occurrence: u64,
) -> Result<String> {
    if event.session.is_some() {
        return Ok(content_id.to_owned());
    }
    let (start, end) = event
        .range
        .filter(|(start, end)| start < end)
        .ok_or_else(|| {
            StoreError::Invalid("an event about its input has no record range".into())
        })?;
    if input.is_empty() || event.after.is_some() {
        return Err(StoreError::Invalid(
            "an input event needs its input identity and cannot follow a message".into(),
        ));
    }
    let mut hash = blake3::Hasher::new_derive_key("polylogue input event v1");
    hash.update(&(input.len() as u64).to_le_bytes());
    hash.update(input.as_bytes());
    hash.update(&chain.to_le_bytes());
    hash.update(&start.to_le_bytes());
    hash.update(&end.to_le_bytes());
    hash.update(&occurrence.to_le_bytes());
    hash.update(content_id.as_bytes());
    Ok(hash.finalize().to_hex().to_string())
}

impl Resolver {
    pub(super) fn gather(
        &self,
        batches: &mut [&mut Batch],
        index: &Connection,
        archive: &Connection,
    ) -> Result<(BTreeSet<Key>, Vec<i64>)> {
        let mut keys = BTreeSet::new();
        let mut chains = Vec::new();
        let mut latest = BTreeMap::<String, Option<(i64, i64)>>::new();
        for b in batches {
            keys.insert(Key::new(Kind::Input, b.input.id.as_bytes()));
            let prior = if matches!(b.change, Change::Reparse { .. }) {
                None
            } else {
                match latest.get(&b.input.id) {Some(prior)=>*prior,None=>archive.query_row("SELECT number,chain FROM revision WHERE input=?1 ORDER BY number DESC LIMIT 1",[&b.input.id],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?))).optional()?}
            };
            let chain = match (&b.change, b.continues, prior) {
                (Change::Reparse { chain }, _, _) => *chain,
                (_, true, Some((_, chain)))
                | (
                    Change::Appended { .. } | Change::Unchanged | Change::Disappeared,
                    _,
                    Some((_, chain)),
                ) => chain,
                _ => prior.map_or(1, |(number, _)| number + 1),
            };
            if !matches!(b.change, Change::Reparse { .. }) {
                let number = match (&b.change, b.continues, prior) {
                    (_, true, Some((number, _)))
                    | (Change::Unchanged | Change::Disappeared, _, Some((number, _))) => number,
                    (_, _, prior) => prior.map_or(1, |(number, _)| number + 1),
                };
                latest.insert(b.input.id.clone(), Some((number, chain)));
            }
            chains.push(chain);
            for p in &mut b.parse {
                p.prepare();
                p.prepared.validate(&p.output)?;
                if p.reset {
                    for (kind, table, column) in [
                        (Kind::Message, "message", "id"),
                        (Kind::Event, "event", "id"),
                        (Kind::Document, "document", "id"),
                    ] {
                        let sql = format!(
                            "SELECT r.{column} FROM {table} r JOIN {table}_source s ON s.{table}_pk=r.pk JOIN input_ref i ON i.pk=s.input_pk WHERE i.id=?1 AND s.chain=?2"
                        );
                        let mut stmt = index.prepare(&sql)?;
                        let mut rows = stmt.query(params![b.input.id, chain])?;
                        while let Some(row) = rows.next()? {
                            keys.insert(Key::new(kind, row.get::<_, String>(0)?.into_bytes()));
                        }
                    }
                    let mut stmt=index.prepare("SELECT l.parent_id,l.child_id,l.kind,l.how FROM link l JOIN link_source s ON s.link_pk=l.pk JOIN input_ref i ON i.pk=s.input_pk WHERE i.id=?1 AND s.chain=?2")?;
                    let mut rows = stmt.query(params![b.input.id, chain])?;
                    while let Some(row) = rows.next()? {
                        let values: (String, String, String, String) =
                            (row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?);
                        keys.insert(Key::new(Kind::Link, serde_json::to_vec(&values)?));
                    }
                    let mut stmt=index.prepare("SELECT s.id FROM session s JOIN parsed p ON p.session_pk=s.pk JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1 AND p.chain=?2")?;
                    let mut rows = stmt.query(params![b.input.id, chain])?;
                    while let Some(row) = rows.next()? {
                        keys.insert(Key::new(
                            Kind::Session,
                            row.get::<_, String>(0)?.into_bytes(),
                        ));
                    }
                }
                for s in &p.output.sessions {
                    keys.insert(Key::new(Kind::Session, s.id.to_string().into_bytes()));
                    let mut stmt=index.prepare("SELECT m.id FROM message m JOIN message_source v ON v.message_pk=m.pk JOIN session s ON s.pk=v.session_pk JOIN input_ref i ON i.pk=v.input_pk WHERE s.id=?1 AND i.id=?2 AND v.chain=?3")?;
                    let mut rows = stmt.query(params![s.id.to_string(), b.input.id, chain])?;
                    while let Some(row) = rows.next()? {
                        keys.insert(Key::new(
                            Kind::Message,
                            row.get::<_, String>(0)?.into_bytes(),
                        ));
                    }
                }
                for m in &p.output.messages {
                    keys.insert(Key::new(
                        Kind::Session,
                        m.message.session.to_string().into_bytes(),
                    ));
                    keys.insert(Key::new(
                        Kind::Message,
                        m.message.id.to_string().into_bytes(),
                    ));
                }
                for d in &p.output.documents {
                    keys.insert(Key::new(
                        Kind::Document,
                        d.document.id.to_string().into_bytes(),
                    ));
                }
                for hash in p.prepared.texts.keys() {
                    keys.insert(Key::new(Kind::Text, hash.as_slice()));
                }
                let mut occurrences = BTreeMap::<(u64, u64), u64>::new();
                for (event, packed) in p.output.events.iter().zip(&p.prepared.events) {
                    if let Some(session) = &event.session {
                        keys.insert(Key::new(Kind::Session, session.to_string().into_bytes()));
                    }
                    if let Some(message) = &event.after {
                        keys.insert(Key::new(Kind::Message, message.to_string().into_bytes()));
                    }
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
                    keys.insert(Key::new(
                        Kind::Event,
                        event_id(&b.input.id, chain, event, &packed.0, occurrence)?.into_bytes(),
                    ));
                }
                for link in &p.output.links {
                    keys.insert(Key::new(Kind::Link, link_key(link)?));
                    keys.insert(Key::new(Kind::Session, link.child.to_string().into_bytes()));
                    keys.insert(Key::new(
                        Kind::Session,
                        link.parent.to_string().into_bytes(),
                    ));
                }
                for update in &p.output.background {
                    keys.insert(Key::new(
                        Kind::Session,
                        update.session.to_string().into_bytes(),
                    ));
                }
            }
        }
        let event_keys: Vec<_> = keys
            .iter()
            .filter(|key| key.kind == Kind::Event)
            .cloned()
            .collect();
        for key in event_keys {
            if let Some(pk) = self.find(index, &key)? {
                let mut stmt=index.prepare("SELECT after_message_id FROM event_source WHERE event_pk=?1 AND after_message_id IS NOT NULL")?;
                let mut rows = stmt.query([pk])?;
                while let Some(row) = rows.next()? {
                    keys.insert(Key::new(
                        Kind::Message,
                        row.get::<_, String>(0)?.into_bytes(),
                    ));
                }
            }
        }
        let contents: Vec<_> = keys
            .iter()
            .filter(|key| matches!(key.kind, Kind::Message | Kind::Document))
            .cloned()
            .collect();
        for key in contents {
            if let Some(pk) = self.find(index, &key)? {
                let kind = key.kind.table();
                let mut statement=index.prepare_cached(&format!("SELECT t.hash FROM {kind}_source s JOIN {kind}_version_text v ON v.version_hash=s.version_hash JOIN text t ON t.id=v.text_id WHERE s.{kind}_pk=?1"))?;
                let mut rows = statement.query([pk])?;
                while let Some(row) = rows.next()? {
                    keys.insert(Key::new(Kind::Text, row.get::<_, Vec<u8>>(0)?));
                }
            }
        }
        Ok((keys, chains))
    }

    pub(super) fn batches(self: &Arc<Self>, batches: &mut [&mut Batch]) -> Result<()> {
        if batches.iter().all(|b| b.parse.is_empty()) {
            return Ok(());
        }
        loop {
            let index = crate::archive::open_read(&self.archive.index_db())?;
            let archive = crate::archive::open_read(&self.archive.archive_db())?;
            let (keys, _) = self.gather(batches, &index, &archive)?;
            drop(index);
            drop(archive);
            let lease = self.lease(keys.clone());
            let index = crate::archive::open_read(&self.archive.index_db())?;
            let snapshot = index.unchecked_transaction()?;
            let archive = crate::archive::open_read(&self.archive.archive_db())?;
            let (fresh, chains) = self.gather(batches, &snapshot, &archive)?;
            if fresh != keys {
                drop(snapshot);
                drop(lease);
                continue;
            }
            let pending: HashMap<_, _> = batches
                .iter()
                .flat_map(|batch| batch.new_blob.iter().chain(&batch.blobs))
                .map(|blob| (blob.hash, blob.size))
                .collect();
            for batch in batches.iter_mut() {
                attachments::normalize(batch, &archive, &pending)?;
            }
            let mut resolved = ResolvedKeys::default();
            for key in keys {
                let existing = self.find(&snapshot, &key)?;
                let row = RowKey {
                    pk: match existing {
                        Some(pk) => pk,
                        None => self.allocate(key.kind)?,
                    },
                    existing: existing.is_some(),
                };
                self.remember(&key, row.pk);
                resolved.rows.insert(key, row);
            }
            versions::resolve(self, &snapshot, batches, &chains, &resolved)?;
            let resolved = Arc::new(resolved);
            let lease = Arc::new(lease);
            for (batch, chain) in batches.iter_mut().zip(chains) {
                for step in &mut batch.parse {
                    step.prepared.keys = Some(Arc::clone(&resolved));
                    step.prepared.chain = Some(chain);
                }
                let old = std::mem::take(&mut batch.release);
                let lease = Arc::clone(&lease);
                batch.release = InFlight::new(move || {
                    drop(old);
                    drop(lease);
                });
            }
            return Ok(());
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct Reservation {
    pub pk: i64,
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PackedValue {
    #[serde(with = "reservation")]
    pub key: Arc<Reservation>,
    pub hash: [u8; 32],
    pub body: Vec<u8>,
}
impl Resolver {
    pub(super) fn value(
        &self,
        index: &Connection,
        kind: Kind,
        value: &impl serde::Serialize,
    ) -> Result<PackedValue> {
        let mut sorted = serde_json::to_value(value)?;
        sorted.sort_all_objects();
        let hash = *blake3::hash(&serde_json::to_vec(&sorted)?).as_bytes();
        let body = crate::encoding::RowPacker::new()?.pack_json(&sorted)?;
        let pk = index
            .query_row(
                &format!("SELECT pk FROM {} WHERE hash=?1", kind.table()),
                [hash.as_slice()],
                |r| r.get::<_, i64>(0),
            )
            .optional()?;
        let key = if let Some(pk) = pk {
            Arc::new(Reservation { pk })
        } else {
            let mut active = self.active.lock().expect("resolver lock");
            let key = Key::new(kind, hash.as_slice());
            if let Some((pk, prior)) = active.values.get_mut(&key) {
                match prior.upgrade() {
                    Some(prior) => prior,
                    None => {
                        let reservation = Arc::new(Reservation { pk: *pk });
                        *prior = Arc::downgrade(&reservation);
                        reservation
                    }
                }
            } else {
                let next = active.next.get_mut(&kind).expect("value allocator");
                let pk = *next;
                *next = next
                    .checked_add(1)
                    .ok_or_else(|| StoreError::Invalid("row identity exhausted".into()))?;
                let reservation = Arc::new(Reservation { pk });
                active
                    .values
                    .insert(key, (pk, Arc::downgrade(&reservation)));
                reservation
            }
        };
        Ok(PackedValue { key, hash, body })
    }
}

mod reservation {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        value: &Arc<Reservation>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&value.pk, serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Arc<Reservation>, D::Error> {
        Ok(Arc::new(Reservation {
            pk: serde::Deserialize::deserialize(deserializer)?,
        }))
    }
}

#[cfg(test)]
pub(super) fn prepare_index(
    index: &Connection,
    batch: &mut Batch,
    chain: i64,
    first: bool,
) -> Result<()> {
    let first = first
        .then(|| crate::first_build::FirstBuild::load(index).map(Arc::new))
        .transpose()?;
    let resolver = Resolver::new(
        &Archive::at(std::path::Path::new("unused-test-location")),
        index,
        first,
    )?;
    let change = std::mem::replace(&mut batch.change, Change::Reparse { chain });
    let (keys, chains) = resolver.gather(&mut [batch], index, index)?;
    batch.change = change;
    let mut resolved = ResolvedKeys::default();
    for key in keys {
        let existing = resolver.find(index, &key)?;
        let row = RowKey {
            pk: match existing {
                Some(pk) => pk,
                None => resolver.allocate(key.kind)?,
            },
            existing: existing.is_some(),
        };
        resolver.remember(&key, row.pk);
        resolved.rows.insert(key, row);
    }
    versions::resolve(&resolver, index, &mut [batch], &chains, &resolved)?;
    let keys = Arc::new(resolved);
    for step in &mut batch.parse {
        step.prepared.keys = Some(Arc::clone(&keys));
        step.prepared.chain = Some(chain);
    }
    Ok(())
}
