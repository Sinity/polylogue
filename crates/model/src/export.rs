//! The export format: one JSON document holding whole sessions, and the
//! archive's user state. Importing an export into a fresh archive gives the
//! same sessions and IDs, and the same notes, settings, kept results,
//! deliveries, chat actions, backfill jobs and excisions.

use crate::{Event, Link, Message, Session};
use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "polylogue-export";
pub const VERSION: u32 = 4;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub format: String,
    pub version: u32,
    pub sessions: Vec<SessionExport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub documents: Vec<crate::Document>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<EventExport>,
    /// Every request a person or agent made and every excision, in commit
    /// order, each as the store writes it (`{"change": "request", ...}` or
    /// `{"change": "excision", ...}`). Import replays them through the
    /// writer; the store checks each entry.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub user_state: Vec<crate::Json>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionExport {
    pub session: Session,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<Event>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<Link>,
}

/// The sender's event identity and source citation are claims kept in the
/// export. Import makes a new local input and cites the exported record itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventExport {
    pub id: String,
    pub input: String,
    pub source: String,
    pub source_chain: i64,
    pub source_start: Option<u64>,
    pub source_end: Option<u64>,
    pub event: Event,
}
impl EventExport {
    pub fn validate(&self) -> Result<(), ExportError> {
        if self.id.is_empty()
            || self.input.is_empty()
            || self.source.is_empty()
            || self.source_chain < 1
            || self.source_start.is_some() != self.source_end.is_some()
            || self
                .source_start
                .zip(self.source_end)
                .is_some_and(|(a, b)| a > b)
            || (self.event.session.is_none() && self.event.after.is_some())
        {
            return Err(ExportError::InvalidEvent);
        }
        Ok(())
    }
}
impl SessionExport {
    pub fn validate(&self) -> Result<(), ExportError> {
        if self
            .events
            .iter()
            .any(|e| e.session.as_ref() != Some(&self.session.id))
        {
            return Err(ExportError::InvalidEvent);
        }
        Ok(())
    }
}

fn unknown_media(block: &crate::Block) -> bool {
    use crate::{AttachmentDirection, Block};
    match block {
        Block::Image(a) | Block::File(a) => a.direction == AttachmentDirection::Unknown,
        Block::ToolResult { attachments, .. } => attachments
            .iter()
            .any(|a| a.direction == AttachmentDirection::Unknown),
        _ => false,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("not a Polylogue export (format {0:?})")]
    NotAnExport(String),
    #[error("export version {found} is newer than this Polylogue reads ({VERSION})")]
    TooNew { found: u32 },
    #[error("document content requires export version 2")]
    DocumentsNeedVersion2,
    #[error("input events, transcripts and unknown media direction require export version 3")]
    ContentNeedsVersion3,
    #[error("user state requires export version 4")]
    UserStateNeedsVersion4,
    #[error("invalid event association or exported source citation")]
    InvalidEvent,
    #[error("malformed export: {0}")]
    Json(#[from] serde_json::Error),
}

impl Document {
    pub fn new(sessions: Vec<SessionExport>) -> Self {
        let mut document = Self {
            format: FORMAT.to_owned(),
            version: 1,
            sessions,
            documents: Vec::new(),
            events: Vec::new(),
            user_state: Vec::new(),
        };
        document.version = document.required_version();
        document
    }

    pub fn with_documents(mut self, documents: Vec<crate::Document>) -> Self {
        self.documents = documents;
        self.version = self.version.max(self.required_version());
        self
    }

    pub fn with_events(mut self, events: Vec<EventExport>) -> Self {
        self.events = events;
        self.version = self.version.max(self.required_version());
        self
    }

    pub fn with_user_state(mut self, user_state: Vec<crate::Json>) -> Self {
        self.user_state = user_state;
        self.version = self.version.max(self.required_version());
        self
    }

    pub fn required_version(&self) -> u32 {
        if !self.user_state.is_empty() {
            4
        } else if !self.events.is_empty()
            || self.documents.iter().any(|d| {
                d.purpose == crate::DocumentPurpose::Transcript
                    || d.blocks.iter().any(unknown_media)
            })
            || self
                .sessions
                .iter()
                .flat_map(|s| &s.messages)
                .flat_map(|m| &m.blocks)
                .any(unknown_media)
        {
            3
        } else if !self.documents.is_empty() {
            2
        } else {
            1
        }
    }

    pub fn validate(&self) -> Result<(), ExportError> {
        if self.version < 4 && !self.user_state.is_empty() {
            return Err(ExportError::UserStateNeedsVersion4);
        }
        if self.version < 3 && self.required_version() >= 3 {
            return Err(ExportError::ContentNeedsVersion3);
        }
        if self.version < 2 && !self.documents.is_empty() {
            return Err(ExportError::DocumentsNeedVersion2);
        }
        for session in &self.sessions {
            session.validate()?;
        }
        for event in &self.events {
            event.validate()?;
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, ExportError> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Reads an export, refusing other formats and newer versions before
    /// looking at the sessions.
    pub fn from_json(text: &str) -> Result<Self, ExportError> {
        #[derive(Deserialize)]
        struct Head {
            format: String,
            version: u32,
        }
        let head: Head = serde_json::from_str(text)?;
        if head.format != FORMAT {
            return Err(ExportError::NotAnExport(head.format));
        }
        if head.version > VERSION {
            return Err(ExportError::TooNew {
                found: head.version,
            });
        }
        let document: Self = serde_json::from_str(text)?;
        document.validate()?;
        Ok(document)
    }
}

#[cfg(test)]
mod tests;
