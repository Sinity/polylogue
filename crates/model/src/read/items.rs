use super::{BranchState, ItemPlace, ReadOptions, View};
use crate::{Author, Block, MessageId, Session, SessionId, TimeAccuracy, Timestamp};
use serde::{Deserialize, Serialize};

/// Missing states carry no value; zero is an ordinary measured value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
pub enum ValueState<T> {
    Value {
        value: T,
    },
    Unknown {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    NotRecorded,
    Unavailable {
        reason: String,
    },
    NotApplicable,
    Partial {
        value: T,
        unknown: u64,
    },
    Inferred {
        value: T,
    },
    Pending,
    Error {
        reason: String,
    },
    Stale {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value: Option<T>,
        reason: String,
    },
    Inherited {
        value: T,
    },
}
impl<T> Default for ValueState<T> {
    fn default() -> Self {
        Self::Unknown { reason: None }
    }
}
impl<T> ValueState<T> {
    pub fn known(value: T) -> Self {
        Self::Value { value }
    }
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Value { value }
            | Self::Partial { value, .. }
            | Self::Inferred { value }
            | Self::Inherited { value } => Some(value),
            Self::Stale { value, .. } => value.as_ref(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Figure {
    pub name: String,
    pub value: ValueState<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

/// One admitted invocation. Transports derive their forms when presenting it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(transparent)]
pub struct ReadyCall {
    pub request: command_surface::RunnableRequest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Source {
    #[serde(rename = "ref")]
    pub reference: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<MessageId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_session: Option<SessionId>,
    #[serde(default)]
    pub inherited: bool,
}
impl Source {
    pub fn record(reference: impl Into<String>) -> Self {
        Self {
            reference: reference.into(),
            message: None,
            block: None,
            source_session: None,
            inherited: false,
        }
    }
    pub fn block(message: &MessageId, session: &SessionId, ordinal: u64, inherited: bool) -> Self {
        Self {
            reference: format!("block:{message}#{ordinal}"),
            message: Some(message.clone()),
            block: Some(ordinal),
            source_session: Some(session.clone()),
            inherited,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TimeValue {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<Timestamp>,
    pub evidence: TimeAccuracy,
}
impl TimeValue {
    pub fn unknown() -> Self {
        Self {
            at: None,
            evidence: TimeAccuracy::UNKNOWN,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadHeader {
    pub target: String,
    pub view: View,
    pub title: ValueState<String>,
    pub kind: HeaderKind,
    #[serde(default)]
    pub figures: Vec<Figure>,
    /// The logical-day zone the reading shows clock times in (an IANA name).
    /// Text views print times in it with their offset and say so once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zone: Option<String>,
    /// What a streamed reading states once, in its `begin` event: the
    /// snapshot it was read at and the options as accepted, which a bundle's
    /// manifest records. A page carries its compact snapshot beside its
    /// document instead, inside the page's byte budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<Box<crate::bundle::ReadSnapshot>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Box<ReadOptions>>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HeaderKind {
    SavedQuery(Box<SavedQueryHeader>),
    Session {
        session: Box<Session>,
    },
    Logical {
        root: SessionId,
        /// The physical sessions the reading joins, in reading order.
        physical_sessions: Vec<SessionId>,
        own_messages: ValueState<u64>,
        inherited_messages: ValueState<u64>,
    },
    Document {
        purpose: ValueState<String>,
        author: ValueState<String>,
    },
    Event {
        event_type: String,
    },
    Record {
        record_type: String,
    },
    Query {
        query: String,
    },
}

/// Identity stays in every page; growing saved history is carried by items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SavedQueryHeader {
    pub name: String,
    pub selected_revision: String,
    pub query_id: crate::Reference,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SavedQueryItem {
    Revision(crate::saved_queries::SavedRevision),
    Kept(crate::saved_queries::KeptSummary),
    Watch {
        reference: Option<String>,
    },
    Dependent {
        reference: String,
        resave: Vec<ReadyCall>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnnotationMark {
    pub reference: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    pub author: Author,
    pub kind: String,
    pub value: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open: Option<ReadyCall>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AlternateMark {
    pub reference: String,
    pub key: String,
    pub original: String,
    pub stale: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_original: Option<ReadyCall>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MessageHeader {
    pub time: TimeValue,
    pub author: ValueState<String>,
    pub model: ValueState<String>,
    #[serde(default)]
    pub annotations: Vec<AnnotationMark>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alternate: Option<AlternateMark>,
    #[serde(default)]
    pub inherited: bool,
    #[serde(default)]
    pub branch: BranchState,
    /// The context meter, when the reading shows it: the provider-reported
    /// input (`value`), the running estimate since the last boundary
    /// (`inferred`), or `unknown`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ValueState<u64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadItem {
    pub source: Source,
    pub kind: ItemKind,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ItemKind {
    LinkIssue {
        issue: super::ParentIssue,
    },
    MessageHeader {
        header: Box<MessageHeader>,
    },
    Text {
        text: String,
    },
    Thinking {
        words: ValueState<u64>,
        content: ThinkingContent,
    },
    Call {
        call: Box<CallItem>,
    },
    Attachment {
        attachment: Box<crate::Attachment>,
        supplied_by: ValueState<String>,
        availability: ValueState<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        link: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        inline_bytes: Option<Vec<u8>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        verified_hash: Option<crate::BlobHash>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        open: Option<Box<ReadyCall>>,
    },
    TextFragment {
        record: Box<crate::bundle::TextFragmentRecord>,
        origin: ValueState<super::FragmentOrigin>,
    },
    Other {
        type_name: String,
        data: serde_json::Value,
    },
    Indicator {
        indicator: IndicatorKind,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expand: Option<Box<ReadyCall>>,
    },
    Omission {
        count: ValueState<u64>,
        unit: String,
        why: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        continue_call: Option<Box<ReadyCall>>,
    },
    Card {
        card: Box<Card>,
    },
    Raw {
        record: Box<RawRecord>,
    },
    /// A lossless chunk of a complete encoded ReadItem. Its outer Source names
    /// the acknowledged query, not the original native record inside the JSON.
    Fragment {
        text: String,
        position: ItemPlace,
        until: u64,
        total: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ThinkingContent {
    Collapsed,
    Encrypted,
    Redacted,
    Text { text: String },
    Absent { blocks: u64 },
}
stored_names! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum IndicatorKind {
        Continues => "continues",
        Branches => "branches",
        Compacted => "compacted",
        Microcompacted => "microcompacted",
        RateLimited => "rate_limited",
        ModelSwitch => "model_switch",
        AwaySummary => "away_summary",
        Harness => "harness",
        Turn => "turn",
        LinkIssue => "link_issue",
        Stale => "stale",
        Alternate => "alternate",
        Selected => "selected",
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CallItem {
    /// A result with no prepared call, distinct from a call whose tool name is unknown.
    #[serde(default)]
    pub orphan: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub tool: ValueState<String>,
    pub target: ValueState<String>,
    pub action: crate::ToolAction,
    pub outcome: ValueState<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit: Option<i64>,
    pub duration_ms: ValueState<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_line: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_excerpt: Option<OutputExcerpt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_source: Option<Source>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub child: Option<Box<DelegationCard>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit: Option<EditView>,
}
/// Explicitly omitted middle of a tool result. Original bytes remain in the
/// canonical record; `open` reads that exact source block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputExcerpt {
    pub parts: Vec<OutputPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open: Option<ReadyCall>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OutputPart {
    Text {
        text: String,
        from: u64,
        until: u64,
    },
    Omission {
        lines: u64,
        characters: u64,
        from: u64,
        until: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DelegationCard {
    pub child: String,
    pub objective: ValueState<String>,
    pub result: ValueState<String>,
    pub duration_ms: ValueState<u64>,
    pub cost: ValueState<serde_json::Value>,
    pub retries: Vec<String>,
    pub result_use: ValueState<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open: Option<ReadyCall>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EditView {
    pub path: String,
    pub before: ValueState<String>,
    pub after: ValueState<String>,
    pub diff: ValueState<String>,
    pub applied: bool,
    pub failed_edits: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Card {
    SavedQuery {
        item: Box<SavedQueryItem>,
    },
    Trace(Box<super::TraceCard>),
    Run {
        query_id: Option<crate::Reference>,
        read_position: crate::ChangePosition,
        author: Author,
        count: Option<u64>,
        exact: Option<bool>,
        outcome: Option<crate::query_objects::RunOutcome>,
    },
    Fields {
        title: String,
        figures: Vec<Figure>,
    },
    Session {
        session: Box<Session>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<AnnotationMark>,
        #[serde(default)]
        lineage_complete: ValueState<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        open: Option<Box<ReadyCall>>,
    },
    LinkNode {
        reference: String,
        lineage: Option<Box<super::SessionLineage>>,
        title: ValueState<String>,
        parents: Vec<String>,
        own_messages: ValueState<u64>,
        inherited_messages: ValueState<u64>,
        own_cost: ValueState<serde_json::Value>,
        tree_cost: ValueState<serde_json::Value>,
        /// This is the session the links view was asked about.
        asked: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        open: Option<Box<ReadyCall>>,
    },
    LinkEdge {
        from: String,
        to: String,
        relation: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        branch_point: Option<String>,
        spawning_call: Option<String>,
        copies_prefix: bool,
        cross_link: bool,
    },
    Change {
        edit: EditView,
    },
    Live {
        state: ValueState<String>,
        waits_on: ValueState<String>,
        call: ValueState<String>,
        task: ValueState<String>,
        peers: Vec<String>,
        processes: Vec<Process>,
    },
    Delegation {
        card: Box<DelegationCard>,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Process {
    pub reference: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    pub command: ValueState<String>,
    pub state: ValueState<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RawRecord {
    pub reference: String,
    pub input: String,
    pub media_type: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReferencedBy {
    pub reference: String,
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open: Option<ReadyCall>,
}
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFooter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub coverage: Vec<Coverage>,
    #[serde(default)]
    pub referenced_by: Vec<ReferencedBy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<ReadyCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<ReadyCall>,
    /// On the last page of a session reading: the call that reads what is
    /// appended after it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub follow: Option<ReadyCall>,
    /// Layout items emitted on this page. Selection counts belong in header figures.
    pub shown: u64,
    /// For a session reading: the shown index entries before this page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<u64>,
    /// The reading's total in the unit `position` counts (a session
    /// reading's shown index entries), only when known without scanning the
    /// selection.
    pub count: ValueState<u64>,
    /// The source records a stream carried beside its items, by kind, so a
    /// receiver checks it got them all. Pages carry none.
    #[serde(default, skip_serializing_if = "crate::bundle::Counts::is_empty")]
    pub counts: crate::bundle::Counts,
}

/// Aggregated omission counts stay bounded by the finite target kinds/views.
/// Individual omitted references remain ordinary streamed Omission items.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Coverage {
    pub target: TargetKind,
    pub view: View,
    pub omitted: u64,
    pub accepted_views: Vec<View>,
}
stored_names! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TargetKind {
        Session => "session",
        Message => "message",
        Block => "block",
        Document => "document",
        TextFragment => "text_fragment",
        Event => "event",
        Authored => "authored",
        Query => "query",
    }
}
/// A bounded page. Full reads emit the same parts without collecting a session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadDocument {
    pub header: ReadHeader,
    pub items: Vec<ReadItem>,
    pub footer: ReadFooter,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "part", content = "value", rename_all = "snake_case")]
pub enum ReadPart {
    Header(ReadHeader),
    Item(ReadItem),
    Footer(Box<ReadFooter>),
}
impl ReadDocument {
    pub fn into_parts(self) -> impl Iterator<Item = ReadPart> {
        std::iter::once(ReadPart::Header(self.header))
            .chain(self.items.into_iter().map(ReadPart::Item))
            .chain(std::iter::once(ReadPart::Footer(Box::new(self.footer))))
    }
}
/// The original ordinal remains meaningful when a view omits earlier blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceBlock {
    pub source: Source,
    pub block: Block,
}
