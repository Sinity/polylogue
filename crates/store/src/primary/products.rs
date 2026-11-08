//! Full product facts exchanged with the confirmed-stream adapter.
use crate::input_rows::InputRows;
use crate::writer::InputKey;
use polylogue_model::authored::PreparedRequest;
use polylogue_model::{
    BackgroundUpdate, BlobHash, DocumentRow, Event, Link, MessageRow, RecordCounts, RepoLookup,
    Session,
};
use serde::{Deserialize, Serialize};
mod rebuild;
pub(crate) use rebuild::AssembledReading;
mod export;
pub use export::ExportReadingKey;

/// The parser reading and exact source chain that produced a fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Reading {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<polylogue_model::batch::ExportDelivery>,
    pub input: InputKey,
    pub chain: i64,
    pub reading_version: String,
    pub order_from: Option<u64>,
    pub repeats_whole_conversations: bool,
    pub start: u64,
    pub parsed_upto: u64,
    pub progress: polylogue_model::batch::StepProgress,
}

/// Authoritative material descriptors supplied by Sinex, never hash lookups.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Material {
    pub id: String,
    pub hash: BlobHash,
    pub length: u64,
    pub extents: Vec<Vec<u8>>,
}

/// Complete parser and authored facts; rebuilding the cache requires no parsing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "product", rename_all = "snake_case")]
pub enum Product {
    Input {
        rows: Box<InputRows>,
        material: Option<Material>,
    },
    Blob(Material),
    Session {
        reading: Reading,
        facts: Session,
    },
    Message {
        reading: Reading,
        facts: Box<MessageRow>,
        order: Option<u64>,
        projection: Option<polylogue_model::bundle::MessageProjection>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        native_fields: Option<Vec<polylogue_model::HookFieldClaim>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        declared_injections: Option<Vec<polylogue_model::injection::InjectionEvidence>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        declared_attachment_bindings:
            Option<Vec<polylogue_model::attachment_binding::AttachmentBinding>>,
    },
    Document {
        reading: Reading,
        facts: DocumentRow,
    },
    Fragment {
        reading: Reading,
        facts: Box<polylogue_model::batch::FragmentRow>,
    },
    Event {
        id: String,
        reading: Reading,
        facts: Event,
        imported: Option<polylogue_model::bundle::ImportedEventIdentity>,
    },
    Link {
        reading: Reading,
        facts: Link,
    },
    Background {
        reading: Reading,
        facts: BackgroundUpdate,
    },
    Repository {
        reading: Reading,
        facts: RepoLookup,
    },
    ParseProgress {
        reading: Reading,
        counts: RecordCounts,
        reset: bool,
        comparison: Option<crate::tool_databases::Comparison>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        semantic_event_ids: Option<std::collections::BTreeMap<usize, String>>,
    },
    AttachmentBinding {
        record: Box<crate::assets::bindings::AttachmentBindingRecord>,
    },
    Query {
        object: Box<polylogue_model::query_objects::QueryObject>,
    },
    QueryRun {
        run: Box<polylogue_model::query_objects::QueryRun>,
    },
    QueryRunOutcome {
        outcome: Box<polylogue_model::query_objects::QueryRunOutcome>,
    },
    Request(PreparedRequest),
    RestoredRequest {
        prepared: PreparedRequest,
    },
    RestoredAssertion {
        fact: polylogue_model::assertion::AssertionFact,
    },
    Coverage(polylogue_model::capture::Coverage),
}

impl Product {
    pub fn reading(&self) -> Option<&Reading> {
        match self {
            Self::Session { reading, .. }
            | Self::Message { reading, .. }
            | Self::Document { reading, .. }
            | Self::Fragment { reading, .. }
            | Self::Event { reading, .. }
            | Self::Link { reading, .. }
            | Self::Background { reading, .. }
            | Self::Repository { reading, .. }
            | Self::ParseProgress { reading, .. } => Some(reading),
            Self::Input { .. }
            | Self::Blob(_)
            | Self::AttachmentBinding { .. }
            | Self::Query { .. }
            | Self::QueryRun { .. }
            | Self::QueryRunOutcome { .. }
            | Self::Request(_)
            | Self::RestoredRequest { .. }
            | Self::RestoredAssertion { .. }
            | Self::Coverage(_) => None,
        }
    }
}

/// One change from Sinex's ordered confirmed stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Confirmed {
    pub position: i64,
    pub output_index: u64,
    pub event: String,
    pub change: ConfirmedChange,
    /// The kind, payload hash and origin Sinex recorded for the event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native: Option<super::NativeFactContext>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", content = "facts", rename_all = "snake_case")]
pub enum ConfirmedChange {
    Live(Box<Product>),
    Archived,
    Restored,
    Tombstoned,
    /// Polylogue could not read the event (an unknown kind, a refused fetch).
    /// It stays in Sinex and is read again by a rebuild; the index records
    /// the refusal and continues with the next event.
    Refused {
        kind: String,
        reason: String,
        /// The adapter cannot exclude source input semantics for this unreadable event.
        unreadable_input: bool,
    },
    Control(Box<super::ReplayControl>),
}

/// A source capture waiting in the producer spool, before cache application.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CapturedInput {
    pub input: super::InputName,
    pub rows: InputRows,
    pub index: u32,
    pub previous: Option<String>,
    pub hash: BlobHash,
    pub length: u64,
    pub group: Option<CaptureGroup>,
    pub arrival_rank: u8,
    pub parsed: Vec<CapturedReading>,
    /// Original acquired interval, independent of tentative source coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_range: Option<super::RawRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<super::SourceGeneration>,
    /// Actual validated source EOF, independent of group-header exhaustion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_predecessor: Option<String>,
    /// None is legacy unknown evidence, never a freshly observed source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis: Option<super::SourceBasis>,
}

/// Complete parser facts retained with an immutable source acquisition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CapturedReading {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export_key: Option<ExportReadingKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<polylogue_model::batch::ExportDelivery>,
    pub output: polylogue_model::Output,
    pub counts: RecordCounts,
    pub start: u64,
    pub parsed_upto: u64,
    pub progress: polylogue_model::batch::StepProgress,
    pub order_from: Option<u64>,
    pub reading_version: String,
    pub repeats_whole_conversations: bool,
    pub reset: bool,
    pub comparison: Option<crate::tool_databases::Comparison>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
}
impl CapturedReading {
    pub fn from_step(
        step: &crate::ParseStep,
        export_key: Option<ExportReadingKey>,
    ) -> crate::Result<Self> {
        if step.delivery.is_some() {
            let key = export_key.as_ref().ok_or_else(|| {
                crate::StoreError::Invalid(
                    "an export capture needs its actual stored source binding".into(),
                )
            })?;
            key.validate()?;
            if key.reading_version != step.reading_version {
                return Err(crate::StoreError::Invalid(
                    "export capture changed its reading version".into(),
                ));
            }
        } else if export_key.is_some() {
            return Err(crate::StoreError::Invalid(
                "ordinary byte control cannot carry an export source binding".into(),
            ));
        }
        let output = step.complete_output()?;
        if !output.blobs.is_empty() {
            return Err(crate::StoreError::Invalid(
                "decoded bytes need their own durable capture before parser facts".into(),
            ));
        }
        step.counts.check()?;
        step.validate_export()?;
        step.validate_stopped_attempt()?;
        Ok(Self {
            export_key,
            delivery: step.delivery,
            output,
            counts: step.counts.clone(),
            start: step.start,
            parsed_upto: step.parsed_upto,
            progress: step.progress.clone(),
            order_from: step.order_from,
            reading_version: step.reading_version.clone(),
            repeats_whole_conversations: step.repeats_whole_conversations,
            reset: step.reset,
            comparison: step.comparison.clone(),
            checkpoint: step.checkpoint.clone(),
            stopped_attempt: step.stopped_attempt,
        })
    }
    pub fn into_step(self) -> crate::ParseStep {
        let mut step = crate::ParseStep::new(
            self.output,
            self.counts,
            self.start..self.parsed_upto,
            false,
        );
        step.progress = self.progress;
        step.delivery = self.delivery;
        step.order_from = self.order_from;
        step.reading_version = self.reading_version;
        step.repeats_whole_conversations = self.repeats_whole_conversations;
        step.reset = self.reset;
        step.comparison = self.comparison;
        step.checkpoint = self.checkpoint;
        step.stopped_attempt = self.stopped_attempt;
        step
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CaptureGroup {
    pub identity: String,
    /// Stable content intent; legacy missing intent cannot prove a retry.
    #[serde(default)]
    pub intent: String,
    pub ordinal: u64,
    pub count: u64,
}

#[derive(Debug, Clone)]
pub enum PreparedInput {
    Confirmed {
        stored: crate::staging::StoredInput,
        material: Material,
    },
    Pending {
        identity: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PreparedWrite {
    pub identity: String,
    pub pending: bool,
}

/// An outbound fact with its native source record and genuine parent readings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ProductWrite {
    pub product: Product,
    pub location: Option<SourceLocation>,
    pub parents: Vec<String>,
    pub machine: Option<String>,
    pub output_index: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SourceLocation {
    pub material: String,
    pub anchor: u64,
}

/// Creates complete outbound products from one parser step, before any index write.
pub fn parse(
    input: &InputKey,
    chain: i64,
    source_length: Option<u64>,
    step: &crate::ParseStep,
) -> crate::Result<Vec<Product>> {
    if step.delivery.is_some() && source_length.is_none() {
        return Err(crate::StoreError::Invalid(
            "export products need their actual stored source length".into(),
        ));
    }
    let event_ids = step.event_identities(&input.id, chain)?;
    let output = step.complete_output()?;
    if !output.blobs.is_empty() {
        return Err(crate::StoreError::Invalid(
            "decoded bytes must be prepared before their product facts".into(),
        ));
    }
    step.validate_export()?;
    let reading = Reading {
        source_length,
        delivery: step.delivery,
        input: input.clone(),
        chain,
        reading_version: step.reading_version.clone(),
        order_from: step.order_from,
        repeats_whole_conversations: step.repeats_whole_conversations,
        start: step.start,
        parsed_upto: step.parsed_upto,
        progress: step.progress.clone(),
    };
    let mut products = Vec::new();
    products.extend(output.sessions.into_iter().map(|facts| Product::Session {
        reading: reading.clone(),
        facts,
    }));
    let mut message_order = output.message_order;
    let mut message_projections = output.message_projections;
    let mut native_fields = output.native_hook_fields;
    let mut declared_injections = output.declared_injections;
    let mut declared_attachment_bindings = output.declared_attachment_bindings;
    products.extend(
        output
            .messages
            .into_iter()
            .enumerate()
            .map(|(index, facts)| Product::Message {
                reading: reading.clone(),
                facts: Box::new(facts),
                order: message_order.remove(&index),
                projection: message_projections.remove(&index),
                native_fields: native_fields.remove(&index),
                declared_injections: declared_injections.remove(&index),
                declared_attachment_bindings: declared_attachment_bindings.remove(&index),
            }),
    );
    products.extend(output.documents.into_iter().map(|facts| Product::Document {
        reading: reading.clone(),
        facts,
    }));
    products.extend(output.fragments.into_iter().map(|facts| Product::Fragment {
        reading: reading.clone(),
        facts: Box::new(facts),
    }));
    let semantic_event_ids = step.delivery.map(|_| output.event_ids.clone());
    let mut imported_events = output.imported_events;
    products.extend(
        output
            .events
            .into_iter()
            .enumerate()
            .map(|(index, facts)| Product::Event {
                id: event_ids[index].clone(),
                reading: reading.clone(),
                facts,
                imported: imported_events.remove(&index),
            }),
    );
    products.extend(output.links.into_iter().map(|facts| Product::Link {
        reading: reading.clone(),
        facts,
    }));
    products.extend(
        output
            .background
            .into_iter()
            .map(|facts| Product::Background {
                reading: reading.clone(),
                facts,
            }),
    );
    products.extend(
        output
            .repo_lookups
            .into_iter()
            .map(|facts| Product::Repository {
                reading: reading.clone(),
                facts,
            }),
    );
    products.push(Product::ParseProgress {
        reading,
        counts: step.counts.clone(),
        reset: step.reset,
        comparison: step.comparison.clone(),
        checkpoint: step.checkpoint.clone(),
        stopped_attempt: step.stopped_attempt,
        semantic_event_ids,
    });
    Ok(products)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "capture", rename_all = "snake_case")]
pub enum Capture {
    Input(Box<CapturedInput>),
    Blob { hash: BlobHash, length: u64 },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PendingCapture {
    pub identity: String,
    pub captured: Capture,
    pub material: Option<Material>,
    pub anchor: Option<SourceLocation>,
}

#[cfg(test)]
mod tests;
