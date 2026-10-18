//! The export format: one JSON document holding whole sessions. Importing
//! an export into a fresh archive gives the same sessions and IDs.

use crate::{Event, Link, Message, Session};
use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "polylogue-export";
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub format: String,
    pub version: u32,
    pub sessions: Vec<SessionExport>,
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
    #[error("malformed export: {0}")]
    Json(#[from] serde_json::Error),
}

impl Document {
    pub fn new(sessions: Vec<SessionExport>) -> Self {
        Self {
            format: FORMAT.to_owned(),
            version: VERSION,
            sessions,
        }
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
        Ok(serde_json::from_str(text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn sample() -> Document {
        let sid = SessionId::parse("claude-code:s1").unwrap();
        let mut session = Session::new(sid.clone(), SessionKind::Main);
        session.offer_title("Fix the build", TitleSource::Custom);
        let message = Message {
            id: MessageId::parse("claude-code:m1").unwrap(),
            session: sid.clone(),
            parent: None,
            role: Role::User,
            at: Timestamp::from_unix_ms(1_700_000_000_000),
            model: None,
            prompt_kind: Some(PromptKind::Prose),
            blocks: vec![Block::Text {
                text: "hello: 12345678901234567890".into(),
            }],
            usage: Usage::default(),
            stop_reason: None,
            reported_cost_usd: None,
            duration_ms: None,
        };
        Document::new(vec![SessionExport {
            session,
            messages: vec![message],
            events: vec![],
            links: vec![],
        }])
    }

    #[test]
    fn round_trips() {
        let doc = sample();
        let text = doc.to_json().unwrap();
        assert_eq!(Document::from_json(&text).unwrap(), doc);
    }

    #[test]
    fn refuses_newer_versions_and_other_formats() {
        let text = r#"{"format":"polylogue-export","version":99,"sessions":[]}"#;
        assert!(matches!(
            Document::from_json(text),
            Err(ExportError::TooNew { found: 99 })
        ));
        let text = r#"{"format":"other","version":1,"sessions":[]}"#;
        assert!(matches!(
            Document::from_json(text),
            Err(ExportError::NotAnExport(_))
        ));
    }

    #[test]
    fn large_numbers_in_opaque_json_are_lossless() {
        let v: Json = serde_json::from_str(r#"{"n":123456789012345678901234567890}"#).unwrap();
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#"{"n":123456789012345678901234567890}"#
        );
    }
}
