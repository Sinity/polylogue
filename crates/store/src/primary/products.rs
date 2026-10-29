//! Full product facts exchanged with the confirmed-stream adapter.
use crate::input_rows::InputRows;
use crate::user_writes::UserRequest;
use crate::writer::InputKey;
use polylogue_model::{
    BackgroundUpdate, BlobHash, DocumentRow, Event, Link, MessageRow, RecordCounts, RepoLookup,
    Session,
};
use serde::{Deserialize, Serialize};

/// The parser reading and exact source chain that produced a fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Material {
    pub id: String,
    pub hash: BlobHash,
    pub length: u64,
    pub extents: Vec<Vec<u8>>,
}

/// Complete parser and authored facts; rebuilding the cache requires no parsing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    },
    Document {
        reading: Reading,
        facts: DocumentRow,
    },
    Event {
        reading: Reading,
        facts: Event,
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
            Self::Input { .. } | Self::Blob(_) | Self::Request(_) => None,
        }
    }
}

/// One change from Sinex's ordered confirmed stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Confirmed {
    pub position: i64,
    pub event: String,
    pub change: ConfirmedChange,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", content = "facts", rename_all = "snake_case")]
pub enum ConfirmedChange {
    Live(Product),
    Archived,
    Restored,
    Tombstoned,
}

/// A source capture waiting in the producer spool, before cache application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturedInput {
    pub input: super::InputName,
    pub rows: InputRows,
    pub index: u32,
    pub previous: Option<String>,
    pub hash: BlobHash,
    pub length: u64,
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
