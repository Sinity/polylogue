//! Keyed parser continuation, separate from the bounded provider-state cache.
use crate::{Json, MessageId, MessageRow};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Key {
    Response {
        namespace: String,
        session: String,
        key: String,
    },
    Alias {
        namespace: String,
        uuid: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Value {
    Response { row: Box<MessageRow> },
    Alias { target: Option<MessageId> },
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("read cancelled")]
    Cancelled,
    #[error("{0}")]
    Failed(String),
}
impl From<String> for Error {
    fn from(error: String) -> Self {
        Self::Failed(error)
    }
}
impl From<&str> for Error {
    fn from(error: &str) -> Self {
        Self::Failed(error.into())
    }
}

/// Implementations own the selected input/chain/reading/cursor view and its
/// speculative child. Provider parsers never choose those identities.
pub trait Context: Send {
    fn get(&mut self, key: &Key) -> Result<Option<Value>, Error>;
    fn put(&mut self, key: Key, value: Value) -> Result<(), Error>;
}

#[derive(Clone)]
pub struct Handle(Arc<Mutex<Access>>);
struct Access {
    backend: Box<dyn Context>,
    error: Option<Error>,
}
impl std::fmt::Debug for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ParserContext")
    }
}
impl Handle {
    pub fn new(backend: impl Context + 'static) -> Self {
        Self(Arc::new(Mutex::new(Access {
            backend: Box::new(backend),
            error: None,
        })))
    }
    pub fn get(&self, key: &Key) -> Option<Value> {
        let mut access = self.0.lock().expect("parser context lock");
        match access.backend.get(key) {
            Ok(value) => value,
            Err(error) => {
                access.error.get_or_insert(error);
                None
            }
        }
    }
    pub fn put(&self, key: Key, value: Value) {
        let mut access = self.0.lock().expect("parser context lock");
        if let Err(error) = access.backend.put(key, value) {
            access.error.get_or_insert(error);
        }
    }
    pub fn is_cancelled(&self) -> bool {
        matches!(
            self.0.lock().expect("parser context lock").error.as_ref(),
            Some(Error::Cancelled)
        )
    }
    /// Any backend failure rejects the entire parse result before publication.
    pub fn check(&self) -> Result<(), String> {
        match &self.0.lock().expect("parser context lock").error {
            Some(error) => Err(error.to_string()),
            None => Ok(()),
        }
    }
}

/// Small explicit fixtures and count-only tools can own their context directly.
/// The engine supplies Store's disk-backed adapter instead.
#[derive(Default)]
pub struct Memory(BTreeMap<Key, Value>);
impl Context for Memory {
    fn get(&mut self, key: &Key) -> Result<Option<Value>, Error> {
        Ok(self.0.get(key).cloned())
    }
    fn put(&mut self, key: Key, value: Value) -> Result<(), Error> {
        self.0.insert(key, value);
        Ok(())
    }
}
impl Default for Handle {
    fn default() -> Self {
        Self::new(Memory::default())
    }
}

/// Compact changes travel inside the existing opaque checkpoint state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Delta {
    pub version: u32,
    pub base: u64,
    pub through: u64,
    pub entries: Vec<Entry>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub value: Reference,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Reference {
    Response {
        message: MessageId,
        version: crate::BlobHash,
        offset: u64,
        end: u64,
    },
    Alias {
        target: Option<MessageId>,
    },
}
pub const FIELD: &str = "_polylogue_context";
pub fn delta(state: &Json) -> Result<Option<Delta>, serde_json::Error> {
    state
        .get(FIELD)
        .cloned()
        .map(serde_json::from_value)
        .transpose()
}

#[cfg(test)]
mod tests;
