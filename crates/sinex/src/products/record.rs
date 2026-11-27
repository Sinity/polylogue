//! Polylogue's records as Sinex events: the event value each record
//! becomes, its source identity, and the events one parser step yields.
use crate::{Error, Result};
use polylogue_model::authored::PreparedRequest;
use polylogue_model::{
    BackgroundUpdate, BlobHash, DocumentRow, Event, Link, MessageRow, RecordCounts, RepoLookup,
    Session,
};
use serde::{Deserialize, Serialize};

/// The parser reading and exact source chain that produced a fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Reading {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<polylogue_model::batch::ExportDelivery>,
    pub input: polylogue_store::InputKey,
    pub chain: i64,
    pub parse_version: String,
    pub start: u64,
    pub parsed_upto: u64,
    /// The step read through the end of the stored bytes.
    pub at_end: bool,
    /// The parser could not go on at `parsed_upto`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<polylogue_model::ChainFailure>,
}

impl Reading {
    pub fn from_step(
        input: &polylogue_store::InputKey,
        chain: i64,
        source_length: Option<u64>,
        step: &polylogue_store::ParsePart,
    ) -> Self {
        Self {
            source_length,
            delivery: step.delivery,
            input: input.clone(),
            chain,
            parse_version: step.parse_version.clone(),
            start: step.start,
            parsed_upto: step.parsed_upto,
            at_end: step.at_end,
            failure: step.failure.clone(),
        }
    }
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
    pub at_evidence: polylogue_model::TimeAccuracy,
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

/// One Polylogue record as a Sinex event: a parsed fact with the reading
/// that produced it, an input, a blob, or an authored record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "product", rename_all = "snake_case")]
pub enum Product {
    /// Reconstructible native eligibility, without a copied local model value.
    Interpreted {
        reading: Reading,
        record: SourceRecord,
    },

    Input {
        rows: Box<polylogue_store::input_rows::InputRows>,
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
        projection: Option<polylogue_model::bundle::ShownBlocks>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stated_fields: Option<Vec<polylogue_model::ParsedHookField>>,
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
    },
    AttachmentBinding {
        record: Box<polylogue_store::assets::bindings::AttachmentBindingRecord>,
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
    RestoredAnnotation {
        fact: polylogue_model::annotation::AnnotationFact,
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
            | Self::RestoredAnnotation { .. }
            | Self::Coverage(_) => None,
        }
    }
}

/// An outgoing event with its record offset in a material and its parent
/// readings.
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SourceRecord {
    pub family: String,
    pub reference: String,
    pub value_hash: BlobHash,
    pub start: u64,
    pub end: u64,
    pub session: Option<String>,
    pub blobs: Vec<BlobHash>,
}

impl SourceRecord {
    pub fn of(product: &Product) -> Result<Option<Self>> {
        if let Product::Interpreted { record, .. } = product {
            return Ok(Some(record.clone()));
        }
        let Some(reading) = product.reading() else {
            return Ok(None);
        };
        let (family, reference, range, session, blocks) = match product {
            Product::Session { facts, .. } => (
                "session",
                format!("session:{}", facts.id),
                None,
                Some(facts.id.to_string()),
                &[][..],
            ),
            Product::Message { facts, .. } => (
                "message",
                format!("message:{}:{}", facts.message.id, facts.offset),
                Some((facts.offset, facts.end.max(facts.offset))),
                Some(facts.message.session.to_string()),
                facts.message.blocks.as_slice(),
            ),
            Product::ToolCall { facts, .. } => (
                "tool_call",
                format!("call:{}:{}", facts.message, facts.block),
                Some((facts.source_start, facts.source_end)),
                Some(facts.session.to_string()),
                &[][..],
            ),
            Product::Document { facts, .. } => (
                "document",
                format!("document:{}:{}", facts.document.id, facts.offset),
                Some((facts.offset, facts.end.max(facts.offset))),
                None,
                facts.document.blocks.as_slice(),
            ),
            Product::Fragment { facts, .. } => (
                "fragment",
                format!("fragment:{}:{}", facts.record.reference, facts.offset),
                Some((facts.offset, facts.end.max(facts.offset))),
                None,
                &[][..],
            ),
            Product::Event { id, facts, .. } => (
                "event",
                format!("event:{id}"),
                None,
                facts.session.as_ref().map(ToString::to_string),
                &[][..],
            ),
            Product::Link { facts, .. } => (
                "link",
                facts.selection_tag(),
                None,
                Some(facts.child.to_string()),
                &[][..],
            ),
            Product::PrefixCopy { facts, .. } => (
                "prefix_copy",
                format!("prefix:{}:{}", facts.parent, facts.child),
                None,
                Some(facts.child.to_string()),
                &[][..],
            ),
            Product::PrefixMember { facts, .. } => (
                "prefix_member",
                format!("member:{}:{}:{}", facts.parent, facts.child, facts.message),
                None,
                Some(facts.child.to_string()),
                &[][..],
            ),
            Product::Background { facts, .. } => (
                "background",
                format!(
                    "background:{}",
                    serde_json::to_string(&(&facts.session, &facts.call_id, &facts.task_id))?
                ),
                None,
                Some(facts.session.to_string()),
                &[][..],
            ),
            Product::Repository { facts, .. } => (
                "repository",
                format!("repository:{}", facts.path),
                None,
                None,
                &[][..],
            ),
            Product::ParseProgress { .. } => (
                "parse_progress",
                format!("progress:{}:{}", reading.start, reading.parsed_upto),
                None,
                None,
                &[][..],
            ),
            _ => unreachable!("all source interpretations listed above"),
        };
        let mut value = serde_json::to_value(product)?;
        value
            .as_object_mut()
            .expect("tagged product object")
            .remove("reading");
        let value_hash = BlobHash::of(&serde_json::to_vec(&value)?);
        let mut blobs = std::collections::BTreeSet::new();
        for block in blocks {
            let attachments = block.attachments();
            blobs.extend(attachments.iter().filter_map(|value| value.blob));
        }
        let (start, end) = range.unwrap_or((reading.start, reading.parsed_upto));
        Ok(Some(Self {
            family: family.into(),
            reference,
            value_hash,
            start,
            end,
            session,
            blobs: blobs.into_iter().collect(),
        }))
    }
}

/// Creates complete outbound products from one parser step, before any index write.
pub fn parse(
    input: &polylogue_store::InputKey,
    chain: i64,
    source_length: Option<u64>,
    step: &polylogue_store::ParsePart,
) -> Result<Vec<Product>> {
    if step.delivery.is_some() && source_length.is_none() {
        return Err(Error::invalid(
            "export products need their stored source length",
        ));
    }
    let event_ids = step.event_identities(&input.id, chain)?;
    let output = step.complete_output()?;
    if !output.blobs.is_empty() {
        return Err(Error::invalid(
            "decoded bytes must be prepared before their products",
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
    let mut block_selections = output.block_selections;
    let mut hook_fields = output.hook_fields;
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
                        block: block_selections
                            .get(&message_index)
                            .map_or(block as u64, |selection| selection.block_ordinals[block]),
                        source_start: row.offset,
                        source_end: row.end,
                        at: row.message.at,
                        at_evidence: row.message.time_accuracy(),
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
                projection: block_selections.remove(&index),
                stated_fields: hook_fields.remove(&index),
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
    });
    Ok(products)
}
