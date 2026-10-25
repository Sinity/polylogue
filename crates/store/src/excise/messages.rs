//! The excised sessions' message IDs, kept after the index rows that
//! supplied them are deleted.
use crate::Result;
use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};
use std::collections::BTreeMap;

/// Each ID with how many times it was selected.
#[derive(Default)]
pub(super) struct SelectedMessages(BTreeMap<String, u64>);

impl SelectedMessages {
    pub(super) fn insert(&mut self, id: &str) -> Result<()> {
        crate::fault::check("excision_messages.insert", id)?;
        *self.0.entry(id.to_owned()).or_default() += 1;
        Ok(())
    }

    pub(super) fn count(&self) -> Result<u64> {
        Ok(self.0.values().sum())
    }

    pub(super) fn contains(&self, id: &str) -> Result<bool> {
        crate::fault::check("excision_messages.contains", id)?;
        Ok(self.0.contains_key(id))
    }
}

impl Serialize for SelectedMessages {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(None)?;
        for (id, times) in &self.0 {
            for _ in 0..*times {
                sequence.serialize_element(id)?;
            }
        }
        sequence.end()
    }
}

#[cfg(test)]
mod tests;
