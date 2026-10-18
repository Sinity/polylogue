//! Point-read predecessor and touched cells. The caller pins input/chain/parser/
//! companion basis; no historical enumeration is available to interpretation.
use polylogue_model::{
    BlobHash, BlockContent, LinkHow, LinkKind, Message, MessageId, Role, Session,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::Infallible;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResponseKey {
    pub role: Role,
    pub response: String,
    pub request: String,
}
impl ResponseKey {
    pub fn user(uuid: &str) -> Self {
        Self {
            role: Role::User,
            response: super::native_key(&polylogue_model::NativeId::unicode(uuid))
                .expect("native key serialization"),
            request: super::native_key(&polylogue_model::NativeId::unicode(""))
                .expect("native key serialization"),
        }
    }
    pub fn new(response: &str, request: &str) -> Self {
        Self {
            role: Role::Assistant,
            response: super::native_key(&polylogue_model::NativeId::unicode(response))
                .expect("native key serialization"),
            request: super::native_key(&polylogue_model::NativeId::unicode(request))
                .expect("native key serialization"),
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    /// Assigned only by the owner after complete-record framing, not by this parser.
    pub parsed_upto: u64,
    pub title: Option<(u8, String)>,
    pub owner: Option<String>,
    pub next_session_ordinal: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveLeaf {
    /// Last marker in source order, including an invalid later marker. This is
    /// not an inferred last Message and never grants membership by itself.
    pub record: crate::selection::RecordRange,
    pub kind: String,
    pub value: polylogue_model::RawJson,
    pub scoped: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCell {
    pub session: Session,
    pub ordinal: u64,
    pub seen_own_user: bool,
    pub next_message_ordinal: u64,
    pub last_message: Option<MessageId>,
    pub facts_pending: bool,
    #[serde(default)]
    pub active_leaf: Option<ActiveLeaf>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseCell {
    /// Source metadata only. Blocks/version/time aggregates are not complete here.
    pub message: Message,
    /// First response occurrence in THIS session; UUID spelling/time have no ordering authority.
    pub ordinal: u64,
    pub author_selection: Option<super::selection::AuthorSelection>,
    pub items: u64,
    pub revision: u64,
    /// Selected raw usage, stop_reason, model, timestamp, in this fixed order.
    pub selected: [Option<Vec<u8>>; 6],
    pub incomplete_bodies: u64,
    pub selected_from: [Option<u64>; 6],
    pub selected_facts: [Vec<super::projection::Fact>; 6],
    pub selected_origins: [Option<crate::selection::SelectedRaw>; 6],
    /// Fixed extra leaf selected by the provider, with its exact original donor.
    pub stop_sequence: Option<super::projection::Fact>,
    /// Canonical RAW selected subtree, independently of normalized attrs.
    pub stop_sequence_raw: Option<Vec<u8>>,
    /// Opaque supplied response metadata, with no finish/status interpretation.
    pub stop_details: Option<super::projection::Fact>,
    pub stop_details_raw: Option<Vec<u8>>,
    /// One last actually supplied immediate record extra per exact attrs key.
    /// Encoded with this response under the existing parser-state admission.
    pub record_extras: BTreeMap<String, SelectedRecordExtra>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedRecordExtra {
    pub item: u64,
    pub fact: super::projection::Fact,
    pub raw: Vec<u8>,
}
impl SelectedRecordExtra {
    /// ASCII hex of exact UTF-8 key bytes keeps the existing framing tag policy.
    pub fn correction_tag(key: &str) -> String {
        use std::fmt::Write;
        let mut tag = String::from("response/record_extra/");
        for byte in key.bytes() {
            write!(&mut tag, "{byte:02x}").expect("writing to String");
        }
        tag
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordRelation {
    pub uuid: polylogue_model::NativeId,
    pub record: crate::selection::RecordRange,
    pub parent: Option<super::parent::Link>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordAddress {
    /// Small current relation proof; ancestry never hydrates content blocks.
    #[serde(default)]
    pub relation: Option<RecordRelation>,
    pub response: ResponseKey,
    pub message: MessageId,
    pub item: u64,
    pub role: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyPayload {
    pub blob: BlobHash,
    pub length: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequiredBody {
    pub content_item: u32,
    pub payload: Option<BodyPayload>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackgroundAddress {
    pub session: String,
    pub response: ResponseKey,
    pub message: MessageId,
    pub item: u64,
    pub local: u32,
    pub call: String,
    pub ambiguous: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackgroundCompletion {
    pub raw: Vec<u8>,
    pub range: crate::selection::RecordRange,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contribution {
    pub uuid: String,
    /// Effective raw field-presence overlay; exact numeric tokens survive.
    pub raw: Vec<u8>,
    pub origins: super::origins::RecordOrigins,
    /// Stable contribution-local positions, not final response BlockIds.
    pub blocks: Vec<BlockContent>,
    pub bodies: Vec<RequiredBody>,
    pub completion: Option<BackgroundCompletion>,
    pub projection: super::projection::RecordFacts,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinkKey {
    pub parent: String,
    pub child: String,
    pub kind: LinkKind,
    pub how: LinkHow,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingMessage {
    pub session_native: String,
    pub response: ResponseKey,
    pub ordinal: u64,
    pub message: MessageId,
    pub revision: u64,
    pub items: u64,
    pub incomplete: bool,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delta {
    pub sessions: Vec<SessionCell>,
    pub responses: Vec<(String, ResponseKey, ResponseCell)>,
    pub addresses: Vec<(String, String, RecordAddress)>,
    pub parent_nodes: Vec<(String, String, super::parent::Node)>,
    pub occurrences: Vec<(String, u64)>,
    pub backgrounds: Vec<(String, BackgroundAddress)>,
    pub contributions: Vec<(MessageId, u64, Contribution)>,
    pub sides: Vec<(String, String)>,
    pub links: Vec<LinkKey>,
    pub pending_messages: Vec<PendingMessage>,
}

pub trait ReadView {
    type Error: std::fmt::Display;
    fn header(&self) -> Result<Header, Self::Error>;
    fn occurrence(&self, hash: &str) -> Result<u64, Self::Error>;
    fn background(&self, task: &str) -> Result<Option<BackgroundAddress>, Self::Error>;
    fn session(&self, native: &str) -> Result<Option<SessionCell>, Self::Error>;
    fn response(
        &self,
        session: &str,
        key: &ResponseKey,
    ) -> Result<Option<ResponseCell>, Self::Error>;
    fn parent_node(
        &self,
        session: &str,
        uuid: &str,
    ) -> Result<Option<super::parent::Node>, Self::Error>;
    fn record(&self, session: &str, uuid: &str) -> Result<Option<RecordAddress>, Self::Error>;
    fn contribution(
        &self,
        message: &MessageId,
        item: u64,
    ) -> Result<Option<Contribution>, Self::Error>;
    fn side(&self, uuid: &str) -> Result<Option<String>, Self::Error>;
    fn has_link(&self, key: &LinkKey) -> Result<bool, Self::Error>;
    /// Required body availability on the same pinned companion basis. A failed
    /// lookup is an error, never a recorded absence or output-not-recorded.
    fn body(&self, path: &str) -> Result<Option<BodyPayload>, Self::Error>;
}
impl<T: ReadView + ?Sized> ReadView for &T {
    type Error = T::Error;
    fn occurrence(&self, hash: &str) -> Result<u64, Self::Error> {
        (**self).occurrence(hash)
    }
    fn background(&self, task: &str) -> Result<Option<BackgroundAddress>, Self::Error> {
        (**self).background(task)
    }
    fn header(&self) -> Result<Header, Self::Error> {
        (**self).header()
    }
    fn session(&self, native: &str) -> Result<Option<SessionCell>, Self::Error> {
        (**self).session(native)
    }
    fn response(
        &self,
        session: &str,
        key: &ResponseKey,
    ) -> Result<Option<ResponseCell>, Self::Error> {
        (**self).response(session, key)
    }
    fn parent_node(
        &self,
        session: &str,
        uuid: &str,
    ) -> Result<Option<super::parent::Node>, Self::Error> {
        (**self).parent_node(session, uuid)
    }
    fn record(&self, session: &str, uuid: &str) -> Result<Option<RecordAddress>, Self::Error> {
        (**self).record(session, uuid)
    }
    fn contribution(
        &self,
        message: &MessageId,
        item: u64,
    ) -> Result<Option<Contribution>, Self::Error> {
        (**self).contribution(message, item)
    }
    fn side(&self, uuid: &str) -> Result<Option<String>, Self::Error> {
        (**self).side(uuid)
    }
    fn has_link(&self, key: &LinkKey) -> Result<bool, Self::Error> {
        (**self).has_link(key)
    }
    fn body(&self, path: &str) -> Result<Option<BodyPayload>, Self::Error> {
        (**self).body(path)
    }
}

/// Legitimate test adapter, not an alternate parser or serialized legacy carry.
#[derive(Default)]
pub struct MemoryReadView {
    header: Header,
    sessions: HashMap<String, SessionCell>,
    responses: HashMap<(String, ResponseKey), ResponseCell>,
    records: HashMap<(String, String), RecordAddress>,
    parent_nodes: HashMap<(String, String), super::parent::Node>,
    occurrences: HashMap<String, u64>,
    backgrounds: HashMap<String, BackgroundAddress>,
    contributions: HashMap<(MessageId, u64), Contribution>,
    sides: HashMap<String, String>,
    links: HashSet<LinkKey>,
    bodies: HashMap<String, BodyPayload>,
}
impl MemoryReadView {
    pub fn set_body(&mut self, path: impl Into<String>, payload: Option<BodyPayload>) {
        let path = path.into();
        match payload {
            Some(body) => {
                self.bodies.insert(path, body);
            }
            None => {
                self.bodies.remove(&path);
            }
        }
    }
    pub fn apply(&mut self, header: Header, delta: Delta) {
        self.header = header;
        self.occurrences.extend(delta.occurrences);
        self.backgrounds.extend(delta.backgrounds);
        for cell in delta.sessions {
            self.sessions.insert(cell.session.native_id.clone(), cell);
        }
        for (s, k, c) in delta.responses {
            self.responses.insert((s, k), c);
        }
        for (s, u, c) in delta.parent_nodes {
            self.parent_nodes.insert((s, u), c);
        }
        for (s, u, c) in delta.addresses {
            self.records.insert((s, u), c);
        }
        for (m, i, c) in delta.contributions {
            self.contributions.insert((m, i), c);
        }
        self.sides.extend(delta.sides);
        self.links.extend(delta.links);
    }
}
impl ReadView for MemoryReadView {
    type Error = Infallible;
    fn background(&self, task: &str) -> Result<Option<BackgroundAddress>, Self::Error> {
        Ok(self.backgrounds.get(task).cloned())
    }
    fn occurrence(&self, hash: &str) -> Result<u64, Self::Error> {
        Ok(self.occurrences.get(hash).copied().unwrap_or(0))
    }
    fn header(&self) -> Result<Header, Self::Error> {
        Ok(self.header.clone())
    }
    fn session(&self, native: &str) -> Result<Option<SessionCell>, Self::Error> {
        Ok(self.sessions.get(native).cloned())
    }
    fn response(
        &self,
        session: &str,
        key: &ResponseKey,
    ) -> Result<Option<ResponseCell>, Self::Error> {
        Ok(self.responses.get(&(session.into(), key.clone())).cloned())
    }
    fn parent_node(
        &self,
        session: &str,
        uuid: &str,
    ) -> Result<Option<super::parent::Node>, Self::Error> {
        Ok(self
            .parent_nodes
            .get(&(session.into(), uuid.into()))
            .cloned())
    }
    fn record(&self, session: &str, uuid: &str) -> Result<Option<RecordAddress>, Self::Error> {
        Ok(self.records.get(&(session.into(), uuid.into())).cloned())
    }
    fn contribution(
        &self,
        message: &MessageId,
        item: u64,
    ) -> Result<Option<Contribution>, Self::Error> {
        Ok(self.contributions.get(&(message.clone(), item)).cloned())
    }
    fn side(&self, uuid: &str) -> Result<Option<String>, Self::Error> {
        Ok(self.sides.get(uuid).cloned())
    }
    fn has_link(&self, key: &LinkKey) -> Result<bool, Self::Error> {
        Ok(self.links.contains(key))
    }
    fn body(&self, path: &str) -> Result<Option<BodyPayload>, Self::Error> {
        Ok(self.bodies.get(path).cloned())
    }
}

/// The owner pins `PendingMessage.revision` and its durable predecessor while
/// preparing layout and version evidence. Each call addresses one contribution.
/// Unit counting and hashing are scheduled owner work, not parser finalization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContributionCursor {
    pub item: u64,
}
impl ResponseCell {
    pub fn next_contribution(&self, cursor: &mut ContributionCursor) -> Option<(MessageId, u64)> {
        if cursor.item >= self.items {
            return None;
        }
        let item = cursor.item;
        cursor.item += 1;
        Some((self.message.id.clone(), item))
    }
    /// After the owner has reduced effective-record metadata in contribution
    /// order, append ONLY the selected raw subtrees that differ, in this order.
    pub fn metadata_units(&self, baseline: &[Option<Vec<u8>>; 8]) -> Vec<EvidenceUnit> {
        [
            "response/usage",
            "response/stop_reason",
            "response/model",
            "response/completed_at",
            "response/cost_usd",
            "response/duration_ms",
            "response/stop_sequence",
            "response/stop_details",
        ]
        .iter()
        .enumerate()
        .filter_map(|(i, tag)| {
            let selected = match i {
                6 => &self.stop_sequence_raw,
                7 => &self.stop_details_raw,
                _ => &self.selected[i],
            };
            if *selected == baseline[i] {
                return None;
            }
            selected.as_ref().map(|raw| EvidenceUnit::Json {
                tag: (*tag).into(),
                canonical: raw.clone(),
                whole_record: false,
            })
        })
        .collect()
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceUnit {
    Json {
        tag: String,
        canonical: Vec<u8>,
        whole_record: bool,
    },
    Body {
        tag: String,
        payload: BodyPayload,
    },
}
impl Contribution {
    /// Only an already-accounted immediate record extra can supply this value.
    pub fn record_extra_raw(&self, key: &str) -> Result<&[u8], String> {
        if !self.projection.facts.iter().any(|fact| {
            matches!(&fact.field, super::projection::Field::Extra { local: None, path }
                if path.as_slice() == [key])
        }) {
            return Err("Claude immediate extra baseline lost its source fact".into());
        }
        // Effective records are already canonical raw JSON. Borrow its exact
        // subtree rather than making another temporary value-sized allocation.
        super::raw::optional_field(&self.raw, &[key])?
            .ok_or_else(|| "Claude immediate extra lost its raw value".to_owned())
    }
    pub fn completion_unit(&self, item: u64) -> Option<EvidenceUnit> {
        self.completion.as_ref().map(|c| EvidenceUnit::Json {
            tag: format!("background-completion/{item}"),
            canonical: c.raw.clone(),
            whole_record: false,
        })
    }
    pub fn record_unit(&self, item: u64) -> EvidenceUnit {
        EvidenceUnit::Json {
            tag: format!("record/{item}"),
            canonical: self.raw.clone(),
            whole_record: true,
        }
    }
    /// One admitted effective record, not a historical response scan. The owner
    /// reduces these fields while reading `next_contribution` slots in order.
    pub fn final_fields(&self) -> Result<[Option<Vec<u8>>; 8], String> {
        let mut fields: [Option<Vec<u8>>; 8] = Default::default();
        for (i, path) in super::METADATA_PATHS.iter().enumerate() {
            fields[i] = super::raw::optional_field(&self.raw, path)?
                .map(super::raw::canonical)
                .transpose()?;
        }
        fields[6] = super::raw::optional_field(&self.raw, &["message", "stop_sequence"])?
            .map(super::raw::canonical)
            .transpose()?;
        fields[7] = super::raw::optional_field(&self.raw, &["message", "stop_details"])?
            .map(super::raw::canonical)
            .transpose()?;
        Ok(fields)
    }
    /// Body units follow all raw-record/final-field units, in (item,content-item)
    /// order. A missing required payload is explicit unresolved work, not a hash
    /// of a missing marker. Optional duplicated inline files create no entries.
    pub fn body_unit(&self, item: u64, body: usize) -> Result<Option<EvidenceUnit>, String> {
        self.bodies
            .get(body)
            .map(|b| {
                b.payload
                    .clone()
                    .map(|payload| EvidenceUnit::Body {
                        tag: format!("record/{item}/tool_result/{}/body", b.content_item),
                        payload,
                    })
                    .ok_or_else(|| "required Claude body remains unavailable".to_owned())
            })
            .transpose()
    }
}
