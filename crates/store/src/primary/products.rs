//! Original interpretations, native eligibility and durable authored facts.
use crate::input_rows::InputRows;
use crate::writer::InputKey;
use polylogue_model::authored::PreparedRequest;
use polylogue_model::{
    BackgroundUpdate, BlobHash, DocumentRow, Event, Link, MessageRow, RecordCounts, RepoLookup,
    Session,
};
use serde::{Deserialize, Serialize};
mod eligibility;
pub use eligibility::SourceRecord;
pub(crate) use eligibility::{
    allowed, needs_publication, previous_reading_bound, scope_hash, selected,
};
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
    pub start: u64,
    pub parsed_upto: u64,
    pub progress: polylogue_model::batch::StepProgress,
}

impl Reading {
    pub(crate) fn from_step(
        input: &InputKey,
        chain: i64,
        source_length: Option<u64>,
        step: &crate::ParseStep,
    ) -> Self {
        Self {
            source_length,
            delivery: step.delivery,
            input: input.clone(),
            chain,
            reading_version: step.reading_version.clone(),
            start: step.start,
            parsed_upto: step.parsed_upto,
            progress: step.progress.clone(),
        }
    }
}

/// The immutable original chain and parser range a publication interprets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedSource {
    pub root_material: String,
    pub reading: Reading,
}

/// Sinex's receipt for a fact frame, saved with the local parser checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicationReceipt {
    pub producer: String,
    pub frame: String,
    pub number: i64,
    /// Native event position when this frame changed readings; absent for a no-op.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_seq: Option<i64>,
    pub root_material: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedPublication {
    Following(super::ReadingScope),
    Pending { identity: String },
    Admitted(PublicationReceipt),
}

/// A tool invocation as its source stated it, independent of local block rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolCall {
    pub session: polylogue_model::SessionId,
    pub message: polylogue_model::MessageId,
    pub block: u64,
    pub source_start: u64,
    pub source_end: u64,
    pub at: Option<polylogue_model::Timestamp>,
    pub at_evidence: polylogue_model::TimeEvidence,
    pub call_id: Option<String>,
    pub tool: Option<String>,
    pub action: polylogue_model::ToolAction,
    pub input: serde_json::Value,
    pub written_text: Option<String>,
    pub command: Option<String>,
    pub path: Option<String>,
    pub run_by_script: bool,
}

/// Authoritative material descriptors supplied by Sinex, never hash lookups.
/// `extents` is the material's one committed blob, or the committed blobs of
/// its content-defined chunks in order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Material {
    pub id: String,
    pub hash: BlobHash,
    pub length: u64,
    pub extents: Vec<Vec<u8>>,
}

impl Material {
    /// The committed blobs that hold the material's bytes, in order. Sinex
    /// records a chunk list only after the chunks read back as the material,
    /// so several descriptors need only add up to its length.
    pub fn descriptors(&self) -> crate::Result<Vec<hashed_blobs::Extent>> {
        let descriptors = self
            .extents
            .iter()
            .map(|bytes| hashed_blobs::Extent::decode(bytes))
            .collect::<hashed_blobs::Result<Vec<_>>>()?;
        let named = match descriptors.as_slice() {
            [] => false,
            [one] => one.hash.0 == self.hash.0 && one.decoded_len == self.length,
            several => several.iter().map(|d| d.decoded_len).sum::<u64>() == self.length,
        };
        if !named || descriptors.iter().any(|descriptor| descriptor.len == 0) {
            return Err(crate::StoreError::Invalid(
                "a material differs from its committed blobs".into(),
            ));
        }
        Ok(descriptors)
    }

    /// Where the bytes of one input step's `extents` begin in this material.
    /// The extents name its blobs in order: a captured step covers the whole
    /// material, and a step trimmed to the confirmed prefix keeps its end, so
    /// they are the material's last blobs, the first possibly entered part
    /// way.
    pub fn offset_of(&self, extents: &[crate::input_rows::Extent]) -> crate::Result<u64> {
        let invalid =
            || crate::StoreError::Invalid("input extents differ from their material".into());
        let descriptors = self.descriptors()?;
        let first = descriptors
            .len()
            .checked_sub(extents.len())
            .filter(|_| !extents.is_empty())
            .ok_or_else(invalid)?;
        let last = extents.len() - 1;
        for (index, (extent, blob)) in extents.iter().zip(&descriptors[first..]).enumerate() {
            let (Ok(offset), Ok(length)) = (
                u64::try_from(extent.blob_offset),
                u64::try_from(extent.length),
            ) else {
                return Err(invalid());
            };
            let end = offset.checked_add(length).ok_or_else(invalid)?;
            if extent.blob.0 != blob.hash.0
                || extent.offset < 0
                || end > blob.decoded_len
                || index > 0 && offset != 0
                || index < last && end != blob.decoded_len
                || index > 0
                    && extents[index - 1].offset + extents[index - 1].length != extent.offset
            {
                return Err(invalid());
            }
        }
        Ok(descriptors[..first]
            .iter()
            .map(|blob| blob.decoded_len)
            .sum::<u64>()
            + extents[0].blob_offset as u64)
    }
}

/// Interpretation writes and confirmed source or authored records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "product", rename_all = "snake_case")]
pub enum Product {
    /// Reconstructible native eligibility, without a copied local model value.
    Interpreted {
        reading: Reading,
        record: SourceRecord,
    },

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
    ToolCall {
        reading: Reading,
        facts: Box<ToolCall>,
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
    PrefixCopy {
        reading: Reading,
        facts: polylogue_model::PrefixCopy,
    },
    PrefixMember {
        reading: Reading,
        facts: polylogue_model::PrefixMember,
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
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
        import: Option<polylogue_model::authored::ImportAttribution>,
    },
    Coverage(polylogue_model::capture::Coverage),
}

impl Product {
    pub fn reading(&self) -> Option<&Reading> {
        match self {
            Self::Interpreted { reading, .. }
            | Self::Session { reading, .. }
            | Self::Message { reading, .. }
            | Self::ToolCall { reading, .. }
            | Self::Document { reading, .. }
            | Self::Fragment { reading, .. }
            | Self::Event { reading, .. }
            | Self::Link { reading, .. }
            | Self::PrefixCopy { reading, .. }
            | Self::PrefixMember { reading, .. }
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
    pub route: polylogue_model::Route,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Eligibility is a local publication instruction; it is not native payload data.
    #[serde(default = "publish_by_default", skip_serializing_if = "is_published")]
    pub publish: bool,
}
fn publish_by_default() -> bool {
    true
}
fn is_published(value: &bool) -> bool {
    *value
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
    let reading = Reading::from_step(input, chain, source_length, step);
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
    for (message_index, row) in output.messages.iter().enumerate() {
        for (block, value) in row.message.blocks.iter().enumerate() {
            if let polylogue_model::Block::ToolCall {
                call_id,
                tool,
                action,
                input,
                written_text,
                command,
                path,
                run_by_script,
            } = value
            {
                products.push(Product::ToolCall {
                    reading: reading.clone(),
                    facts: Box::new(ToolCall {
                        session: row.message.session.clone(),
                        message: row.message.id.clone(),
                        block: message_projections
                            .get(&message_index)
                            .map_or(block as u64, |projection| projection.block_ordinals[block]),
                        source_start: row.offset,
                        source_end: row.end,
                        at: row.message.at,
                        at_evidence: row.message.time_evidence(),
                        call_id: call_id.clone(),
                        tool: tool.clone(),
                        action: *action,
                        input: input.clone(),
                        written_text: written_text.clone(),
                        command: command.clone(),
                        path: path.clone(),
                        run_by_script: *run_by_script,
                    }),
                });
            }
        }
    }
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
            .prefix_copies
            .into_iter()
            .map(|facts| Product::PrefixCopy {
                reading: reading.clone(),
                facts,
            }),
    );
    products.extend(
        output
            .prefix_members
            .into_iter()
            .map(|facts| Product::PrefixMember {
                reading: reading.clone(),
                facts,
            }),
    );
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
        stopped_attempt: step.stopped_attempt,
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
