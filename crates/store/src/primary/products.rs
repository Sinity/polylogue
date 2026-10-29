//! Full product facts exchanged with the confirmed-stream adapter.
use crate::input_rows::InputRows;
use crate::writer::InputKey;
use polylogue_model::user::UserRequest;
use polylogue_model::{
    BackgroundUpdate, BlobHash, DocumentRow, Event, Link, MessageRow, RecordCounts, RepoLookup,
    Session,
};
use serde::{Deserialize, Serialize};

/// The parser reading and exact source chain that produced a fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Reading {
    pub input: InputKey,
    pub chain: i64,
    pub reading_version: String,
    pub order_from: Option<u64>,
    pub repeats_whole_conversations: bool,
    pub start: u64,
    pub parsed_upto: u64,
    pub at_end: bool,
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
        rows: InputRows,
        material: Option<Material>,
    },
    Blob(Material),
    Session {
        reading: Reading,
        facts: Session,
    },
    Message {
        reading: Reading,
        facts: MessageRow,
        order: Option<u64>,
    },
    Document {
        reading: Reading,
        facts: DocumentRow,
    },
    Event {
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
    },
    Request(UserRequest),
    Coverage(polylogue_model::capture::Coverage),
}

impl Product {
    pub fn reading(&self) -> Option<&Reading> {
        match self {
            Self::Session { reading, .. }
            | Self::Message { reading, .. }
            | Self::Document { reading, .. }
            | Self::Event { reading, .. }
            | Self::Link { reading, .. }
            | Self::Background { reading, .. }
            | Self::Repository { reading, .. }
            | Self::ParseProgress { reading, .. } => Some(reading),
            Self::Input { .. } | Self::Blob(_) | Self::Request(_) | Self::Coverage(_) => None,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", content = "facts", rename_all = "snake_case")]
pub enum ConfirmedChange {
    Live(Product),
    Archived,
    Restored,
    Tombstoned,
    Ignored,
    PositionComplete,
}

/// A source capture waiting in the producer spool, before cache application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CapturedInput {
    pub input: super::InputName,
    pub rows: InputRows,
    pub index: u32,
    pub previous: Option<String>,
    pub hash: BlobHash,
    pub length: u64,
    pub group: Option<CaptureGroup>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CaptureGroup {
    pub identity: String,
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
pub fn parse(input: &InputKey, chain: i64, step: &crate::ParseStep) -> crate::Result<Vec<Product>> {
    let output = step.complete_output()?;
    if !output.blobs.is_empty() {
        return Err(crate::StoreError::Invalid(
            "decoded bytes must be prepared before their product facts".into(),
        ));
    }
    let reading = Reading {
        input: input.clone(),
        chain,
        reading_version: step.reading_version.clone(),
        order_from: step.order_from,
        repeats_whole_conversations: step.repeats_whole_conversations,
        start: step.start,
        parsed_upto: step.parsed_upto,
        at_end: step.at_end,
    };
    let mut products = Vec::new();
    products.extend(output.sessions.into_iter().map(|facts| Product::Session {
        reading: reading.clone(),
        facts,
    }));
    let mut message_order = output.message_order;
    products.extend(
        output
            .messages
            .into_iter()
            .enumerate()
            .map(|(index, facts)| Product::Message {
                reading: reading.clone(),
                facts,
                order: message_order.remove(&index),
            }),
    );
    products.extend(output.documents.into_iter().map(|facts| Product::Document {
        reading: reading.clone(),
        facts,
    }));
    let mut imported_events = output.imported_events;
    products.extend(
        output
            .events
            .into_iter()
            .enumerate()
            .map(|(index, facts)| Product::Event {
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
    });
    Ok(products)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "capture", rename_all = "snake_case")]
pub enum Capture {
    Input(CapturedInput),
    Blob { hash: BlobHash, length: u64 },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PendingCapture {
    pub identity: String,
    pub captured: Capture,
    pub material: Option<Material>,
}
