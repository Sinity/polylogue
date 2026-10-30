//! Declared accepted history survives retirement of its native commands.
//! These records cannot construct a current UserRequest or PreparedRequest.
use crate::{
    BlobHash, Timestamp, authored::ImportHistory, authorship::Authorship, user::UserOutcome,
};
use serde::{Deserialize, Serialize};

pub(crate) fn read_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> serde_json::Result<T> {
    let mut reader = serde_json::Deserializer::from_slice(bytes);
    reader.disable_recursion_limit();
    let value = T::deserialize(&mut reader)?;
    reader.end()?;
    Ok(value)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "schema", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceSchema {
    PolylogueArchive {
        version: u32,
    },
    LocalProjection {
        index_version: u32,
        source_schema_version: u32,
    },
    RetainedPublication {
        source: String,
        source_version: u32,
        source_fingerprint: String,
        parser: String,
        parser_version: u32,
    },
    NativeProjection {
        source: String,
        source_version: u32,
        source_fingerprint: String,
        kind: String,
        kind_version: u64,
        kind_definition_hash: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AcceptedHashRule {
    MachineAndIntent,
    MachineAuthorshipAndIntent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "evidence", rename_all = "snake_case", deny_unknown_fields)]
pub enum AcceptedHashEvidence {
    Observed,
    Derived { rule: AcceptedHashRule },
}
impl AcceptedHashEvidence {
    fn validate(&self, source: &SourceSchema) -> Result<(), InvalidHistory> {
        match (source, self) {
            (
                SourceSchema::NativeProjection { .. }
                | SourceSchema::RetainedPublication { .. }
                | SourceSchema::LocalProjection { .. },
                Self::Derived {
                    rule: AcceptedHashRule::MachineAuthorshipAndIntent,
                },
            ) => Ok(()),
            (
                SourceSchema::NativeProjection { .. } | SourceSchema::RetainedPublication { .. },
                _,
            )
            | (_, Self::Derived { .. }) => Err(InvalidHistory::HashEvidence),
            (_, Self::Observed) => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeRecordLocation {
    pub material: String,
    pub anchor: u64,
}

/// Verified retained publication facts. These identify durable producer custody,
/// never a fetched event, canonical application or native execution eligibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueuedPublicationCustody {
    pub producer_family: String,
    pub producer_identity: String,
    pub receipt_number: i64,
    pub frame: String,
    pub frame_content_hash: BlobHash,
    pub material: String,
    pub blob: BlobHash,
    pub length: u64,
    pub source: String,
    pub source_version: u32,
    pub source_fingerprint: String,
    pub parser: String,
    pub parser_version: u32,
}
impl QueuedPublicationCustody {
    pub fn source_schema(&self) -> SourceSchema {
        SourceSchema::RetainedPublication {
            source: self.source.clone(),
            source_version: self.source_version,
            source_fingerprint: self.source_fingerprint.clone(),
            parser: self.parser.clone(),
            parser_version: self.parser_version,
        }
    }
}

/// What the authenticated owner fetch actually supplied. Record/projection
/// hashes are distinct from an accepted intent hash; these fields confer no
/// original-capture, native-execution or importing-actor authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeProjectionProvenance {
    pub event: String,
    pub run: String,
    pub commit_seq: i64,
    pub output_index: u64,
    pub record_hash: String,
    pub projection_epoch: i64,
    pub projection_payload_hash: String,
    pub location: Option<NativeRecordLocation>,
    pub parents: Vec<String>,
    pub producer_family: Option<String>,
    pub producer_identity: Option<String>,
    pub ts_orig: Option<Timestamp>,
    pub ts_quality: String,
    pub ts_coided: Option<Timestamp>,
    pub ts_persisted: Timestamp,
    pub record_state: String,
    pub class: String,
    pub validation: String,
    pub equivalence_key: Option<String>,
    pub reading_order: i64,
    pub projected_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeProjectionObservation {
    pub source: SourceSchema,
    pub provenance: NativeProjectionProvenance,
}

impl NativeProjectionObservation {
    /// A fetched observation key is not an original-capture proof or privilege.
    pub fn same_observation(&self, other: &Self) -> bool {
        let a = &self.provenance;
        let b = &other.provenance;
        (
            &a.event,
            &a.record_hash,
            &a.projection_payload_hash,
            a.projection_epoch,
        ) == (
            &b.event,
            &b.record_hash,
            &b.projection_payload_hash,
            b.projection_epoch,
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeProjectionHistory {
    pub observed: Vec<NativeProjectionObservation>,
    pub declared: Vec<NativeProjectionObservation>,
}
impl NativeProjectionHistory {
    pub fn is_empty(&self) -> bool {
        self.observed.is_empty() && self.declared.is_empty()
    }
    pub fn observe(
        &mut self,
        observation: NativeProjectionObservation,
    ) -> Result<(), InvalidHistory> {
        for known in &self.observed {
            if known.same_observation(&observation) {
                return if known == &observation {
                    Ok(())
                } else {
                    Err(InvalidHistory::NativeObservation)
                };
            }
        }
        self.observed.push(observation);
        Ok(())
    }
    pub fn declare(&mut self, observations: impl IntoIterator<Item = NativeProjectionObservation>) {
        for observation in observations {
            if !self.declared.contains(&observation) {
                self.declared.push(observation);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NoteKind {
    Tag,
    Note,
    Title,
    Decision,
    Claim,
    Lesson,
    Blocker,
    Question,
    Finding,
    Judgment,
    Outcome,
    Hidden,
    Correction,
    Handoff,
    Goal,
    Problem,
    Forecast,
    SavedQuery,
    Ping,
    Signal,
    Withdrawal,
    Rubric,
    Distillate,
    Comparison,
    Page,
    Slot,
    Member,
    Setting,
}
stored_names!(NoteKind {
    Tag=>"tag",Note=>"note",Title=>"title",Decision=>"decision",Claim=>"claim",
    Lesson=>"lesson",Blocker=>"blocker",Question=>"question",Finding=>"finding",
    Judgment=>"judgment",Outcome=>"outcome",Hidden=>"hidden",Correction=>"correction",
    Handoff=>"handoff",Goal=>"goal",Problem=>"problem",Forecast=>"forecast",
    SavedQuery=>"saved_query",Ping=>"ping",Signal=>"signal",Withdrawal=>"withdrawal",
    Rubric=>"rubric",Distillate=>"distillate",Comparison=>"comparison",Page=>"page",Slot=>"slot",Member=>"member",Setting=>"setting"
});

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum RecordedKind {
    Known(NoteKind),
    Unrecognized(String),
}
impl RecordedKind {
    pub fn name(&self) -> &str {
        match self {
            Self::Known(kind) => kind.as_str(),
            Self::Unrecognized(name) => name,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    Answers,
    Challenges,
    DerivedFrom,
    Blocks,
    Corrects,
    MemberOf,
    Revises,
    Withdraws,
    SlotOf,
    OnPage,
    Subject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum RecordedRelation {
    Known(Relation),
    Unrecognized(String),
}

/// A predecessor assertion remains evidence under its recorded value schema.
/// Its malformed or missing current fields never become a new native fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HistoricalAssertionEvidence {
    pub id: String,
    pub root: String,
    pub source: SourceSchema,
    pub kind: RecordedKind,
    pub key: Option<String>,
    pub value: Option<String>,
    pub recorded_author: Option<String>,
    pub machine: Option<String>,
    pub created_at: Option<Timestamp>,
    pub valid_until: Option<Timestamp>,
    pub targets: Vec<String>,
    pub relations: Vec<(RecordedRelation, String)>,
    pub links: Vec<crate::assertion::AssertionLink>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predecessor: Option<crate::Json>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(
    tag = "form",
    content = "assertion",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum AssertionEvidence {
    Current(crate::assertion::Assertion),
    Historical {
        fact: HistoricalAssertionEvidence,
        commit_seq: u64,
        imports: ImportHistory,
    },
}
impl AssertionEvidence {
    pub fn reference(&self) -> crate::Reference {
        let (root, id) = match self {
            Self::Current(assertion) => (&assertion.fact.root, &assertion.fact.id),
            Self::Historical { fact, .. } => (&fact.root, &fact.id),
        };
        crate::Reference::Assertion(crate::RevisionReference::exact(root.clone(), id.clone()))
    }
}

/// Original spellings are declared provenance, never runtime reference aliases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoteIntent {
    pub kind: NoteKind,
    pub key: Option<String>,
    pub value: Option<String>,
    pub targets: Vec<String>,
    pub relations: Vec<(Relation, String)>,
    pub revises: Option<String>,
    pub valid_until: Option<Timestamp>,
    pub claimed_author: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum Intent {
    Note {
        note: NoteIntent,
    },
    SavedQuery {
        note: NoteIntent,
    },
    QueryAnnotation {
        query: String,
        note: NoteIntent,
    },
    Withdrawal {
        note: String,
        reason: Option<String>,
    },
    Setting {
        key: String,
        value: String,
        machine: Option<String>,
        claimed_author: Option<String>,
    },
    Variant {
        variant: HistoricalVariantIntent,
    },
    /// Compatible payload fields whose original request metadata or effect
    /// encoding cannot be represented as an unchanged current request.
    Other {
        write: Box<crate::user::UserWrite>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HistoricalVariantIntent {
    pub base: Option<String>,
    pub source: String,
    pub source_hash: BlobHash,
    pub kind: crate::VariantKind,
    pub from_lang: Option<String>,
    pub to_lang: Option<String>,
    pub claimed_author: Option<String>,
    pub claimed_via: Option<String>,
    pub method: Option<String>,
    pub parts: Vec<crate::NewVariantPart>,
}

/// A migration's typed effect projection accompanies the original bytes. It
/// restores data through historical apply, never by executing the old intent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConservedEffects {
    pub current: crate::authored::AuthoredEffect,
    pub historical_assertions: Vec<HistoricalAssertionEvidence>,
}

/// The predecessor cache's partial envelope, preserved as historical metadata.
/// Missing full fetch fields remain absent; this is not a native observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CachedNativeContext {
    pub event: String,
    pub admitted: Option<i64>,
    pub position: i64,
    pub source_order: i64,
    pub state: CachedRecordState,
    pub native: Option<crate::native::NativeFactContext>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CachedRecordState {
    Live,
    Archived,
    Tombstoned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OriginalBytes {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cached: Vec<CachedNativeContext>,
    /// Exact retained JSON in the named source encoding. NativeProjection
    /// identifies an SDK projection encoding, not unavailable ingress spelling.
    pub request: Option<Vec<u8>>,
    /// Exact semantic intent encoding covered by the original content hash.
    pub intent: Option<Vec<u8>>,
    pub outcome: Option<Vec<u8>>,
    pub effect: Option<Vec<u8>>,
    /// Exact available predecessor envelopes and alternate encodings of one
    /// observed JSON value, separate from unavailable acquisition spelling.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub records: Vec<Vec<u8>>,
    /// The original store's recorded import classifications. These are source
    /// metadata, never observations minted at the receiving store.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imports: Option<ImportHistory>,
}

/// Original encoding retained when a numbered conversion can represent the
/// same accepted action and effects in the current typed prepared request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AcceptedEncoding {
    pub source: SourceSchema,
    pub content_hash: BlobHash,
    pub hash_evidence: AcceptedHashEvidence,
    pub original: OriginalBytes,
    #[serde(default, skip_serializing_if = "NativeProjectionHistory::is_empty")]
    pub native: NativeProjectionHistory,
}

impl AcceptedEncoding {
    pub fn validate(
        &self,
        request: &crate::user::UserRequest,
        outcome: &UserOutcome,
    ) -> Result<(), InvalidHistory> {
        self.hash_evidence.validate(&self.source)?;
        let intent = self
            .original
            .intent
            .as_deref()
            .ok_or(InvalidHistory::Intent)?;
        if BlobHash::of(intent) != self.content_hash {
            return Err(InvalidHistory::Hash);
        }
        let original: crate::user::UserRequest = read_json(
            self.original
                .request
                .as_deref()
                .ok_or(InvalidHistory::Request)?,
        )
        .map_err(|_| InvalidHistory::Request)?;
        if original != *request {
            return Err(InvalidHistory::Request);
        }
        let original = self
            .original
            .outcome
            .as_ref()
            .and_then(|bytes| read_json::<UserOutcome>(bytes).ok());
        if original.as_ref() != Some(outcome) {
            return Err(InvalidHistory::OutcomeBytes);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HistoricalAcceptedRequest {
    pub id: String,
    pub source: SourceSchema,
    pub content_hash: BlobHash,
    pub hash_evidence: AcceptedHashEvidence,
    /// Missing historical bodies and outcomes remain unrecorded.
    pub original: Option<OriginalBytes>,
    pub intent: Option<Intent>,
    pub machine: Option<String>,
    pub at: Option<Timestamp>,
    pub authorship: Option<Authorship>,
    pub outcome: Option<UserOutcome>,
    pub conserved_effects: Option<ConservedEffects>,
    pub facts: Vec<crate::Reference>,
    pub imports: ImportHistory,
    #[serde(default, skip_serializing_if = "NativeProjectionHistory::is_empty")]
    pub native: NativeProjectionHistory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(
    tag = "form",
    content = "record",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ConservedAccepted {
    Compatible(Box<crate::authored::PreparedRequest>),
    Historical(Box<HistoricalAcceptedRequest>),
}
impl ConservedAccepted {
    pub fn id(&self) -> &str {
        match self {
            Self::Compatible(prepared) => &prepared.request.id,
            Self::Historical(record) => &record.id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrity {
    HashMatches,
    Unrecorded,
}

#[derive(Debug, thiserror::Error)]
pub enum InvalidHistory {
    #[error("historical request identity is empty")]
    Identity,
    #[error("historical request bytes do not match its original content hash")]
    Hash,
    #[error("historical request intent and original bytes must be recorded together")]
    Intent,
    #[error("historical request outcome has no recorded identity")]
    Outcome,
    #[error("historical request body disagrees with its recorded identity")]
    Request,
    #[error("historical request outcome disagrees with its original bytes")]
    OutcomeBytes,
    #[error("native observations disagree under the same actual event, hashes and fetched epoch")]
    NativeObservation,
    #[error("accepted hash evidence disagrees with the named source encoding")]
    HashEvidence,
    #[error("historical run is invalid: {0}")]
    Run(String),
}

#[cfg(test)]
mod tests;

impl HistoricalAcceptedRequest {
    /// This checks retained bytes and structure, never native admission or the
    /// truth of an importing sender's historical authorship claims.
    pub fn validate(&self) -> Result<Integrity, InvalidHistory> {
        self.hash_evidence.validate(&self.source)?;
        if self.id.is_empty() {
            return Err(InvalidHistory::Identity);
        }
        if self
            .outcome
            .as_ref()
            .is_some_and(|outcome| outcome.id.is_empty())
        {
            return Err(InvalidHistory::Outcome);
        }
        match (&self.original, &self.intent) {
            (Some(original), Some(_))
                if original
                    .intent
                    .as_deref()
                    .is_some_and(|intent| BlobHash::of(intent) == self.content_hash) =>
            {
                #[derive(Deserialize)]
                struct RequestIdentity {
                    id: String,
                }
                let identity: RequestIdentity =
                    read_json(original.request.as_deref().ok_or(InvalidHistory::Request)?)
                        .map_err(|_| InvalidHistory::Request)?;
                if identity.id != self.id {
                    return Err(InvalidHistory::Request);
                }
                match (&original.outcome, &self.outcome) {
                    (Some(bytes), Some(outcome))
                        if read_json::<UserOutcome>(bytes).ok().as_ref() == Some(outcome) => {}
                    (None, None) => {}
                    _ => return Err(InvalidHistory::OutcomeBytes),
                }
                Ok(Integrity::HashMatches)
            }
            (Some(_), Some(_)) => Err(InvalidHistory::Hash),
            (Some(original), None) if original.request.is_none() && original.intent.is_none() => {
                match (&original.outcome, &self.outcome) {
                    (Some(bytes), Some(outcome))
                        if read_json::<UserOutcome>(bytes).ok().as_ref() == Some(outcome) => {}
                    (None, None) => {}
                    _ => return Err(InvalidHistory::OutcomeBytes),
                }
                Ok(Integrity::Unrecorded)
            }
            (None, None) if self.outcome.is_none() => Ok(Integrity::Unrecorded),
            _ => Err(InvalidHistory::Intent),
        }
    }
}

/// Read-only restoration of recorded predecessor evidence, with no native
/// command or current-schema validation implied by its historical value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(
    tag = "kind",
    content = "record",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum HistoricalRecord {
    Accepted(Box<HistoricalAcceptedRequest>),
    Assertion(Box<HistoricalAssertionEvidence>),
    Run(Box<crate::query_objects::HistoricalQueryRun>),
}
impl HistoricalRecord {
    pub fn id(&self) -> &str {
        match self {
            Self::Accepted(record) => &record.id,
            Self::Assertion(fact) => &fact.id,
            Self::Run(run) => run.id.as_str(),
        }
    }
    pub fn reference(&self) -> String {
        match self {
            Self::Accepted(record) => format!("request:{}", record.id),
            Self::Assertion(fact) => format!("assertion:{}@{}", fact.root, fact.id),
            Self::Run(run) => format!("run:{}", run.id),
        }
    }
    pub fn validate(&self) -> Result<(), InvalidHistory> {
        let source = match self {
            Self::Accepted(record) => &record.source,
            Self::Assertion(fact) => &fact.source,
            Self::Run(run) => &run.source,
        };
        if matches!(source, SourceSchema::RetainedPublication { .. }) {
            return Err(InvalidHistory::Request);
        }
        match self {
            Self::Accepted(record) => {
                record.validate()?;
                if record
                    .outcome
                    .as_ref()
                    .is_some_and(|outcome| outcome.pending)
                {
                    return Err(InvalidHistory::OutcomeBytes);
                }
            }
            Self::Run(run) => run.validate().map_err(InvalidHistory::Run)?,
            Self::Assertion(fact) => {
                if fact.id.is_empty() || fact.root.is_empty() {
                    return Err(InvalidHistory::Request);
                }
                let mut keys = std::collections::BTreeSet::new();
                if fact
                    .links
                    .iter()
                    .any(|link| !keys.insert((link.role, &link.reference)))
                {
                    return Err(InvalidHistory::Request);
                }
            }
        }
        Ok(())
    }
}
