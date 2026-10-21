//! In-memory state for individual captures and small parser fixtures.
use std::collections::HashMap;

#[derive(Default)]
pub struct MemoryState {
    first: HashMap<String, polylogue_model::SessionId>,
    previous: HashMap<polylogue_model::SessionId, polylogue_model::MessageId>,
}

impl crate::export::ExportState for MemoryState {
    fn claim(
        &mut self,
        native: &str,
        session: &polylogue_model::SessionId,
    ) -> Result<Option<polylogue_model::SessionId>, String> {
        use std::collections::hash_map::Entry;
        Ok(match self.first.entry(native.to_owned()) {
            Entry::Occupied(entry) => Some(entry.get().clone()),
            Entry::Vacant(entry) => {
                entry.insert(session.clone());
                None
            }
        })
    }
    fn last_message(
        &mut self,
        session: &polylogue_model::SessionId,
    ) -> Result<Option<polylogue_model::MessageId>, String> {
        Ok(self.previous.get(session).cloned())
    }
    fn set_last_message(
        &mut self,
        session: &polylogue_model::SessionId,
        message: &polylogue_model::MessageId,
    ) -> Result<(), String> {
        self.previous.insert(session.clone(), message.clone());
        Ok(())
    }
}
