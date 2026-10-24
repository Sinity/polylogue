//! The export format: one JSON document holding whole sessions. Importing
//! an export into a fresh archive gives the same sessions and IDs.

use crate::{Event, Link, Message, Session};
use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "polylogue-export";
pub const VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub format: String,
    pub version: u32,
    pub sessions: Vec<SessionExport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub documents: Vec<crate::Document>,
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

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("not a Polylogue export (format {0:?})")]
    NotAnExport(String),
    #[error("export version {found} is newer than this Polylogue reads ({VERSION})")]
    TooNew { found: u32 },
    #[error("document content requires export version 2")]
    DocumentsNeedVersion2,
    #[error("malformed export: {0}")]
    Json(#[from] serde_json::Error),
}

impl Document {
    pub fn new(sessions: Vec<SessionExport>) -> Self {
        Self {
            format: FORMAT.to_owned(),
            version: 1,
            sessions,
            documents: Vec::new(),
        }
    }

    pub fn with_documents(mut self, documents: Vec<crate::Document>) -> Self {
        if !documents.is_empty() {
            self.version = VERSION;
        }
        self.documents = documents;
        self
    }

    pub fn to_json(&self) -> Result<String, ExportError> {
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
        if document.version < 2 && !document.documents.is_empty() {
            return Err(ExportError::DocumentsNeedVersion2);
        }
        Ok(document)
    }
}

#[cfg(test)]
mod tests;
