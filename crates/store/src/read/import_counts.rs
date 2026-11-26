//! Counts of indexed objects attributed to a selected package's inputs.
use super::*;

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct ImportCounts {
    pub sessions: u64,
    pub documents: u64,
}

impl Reader<'_> {
    pub fn import_counts(&self, inputs: &[String]) -> Result<ImportCounts> {
        let ids = serde_json::to_string(inputs)?;
        let sessions: u64 = self.conn()?.scalar("SELECT COUNT(DISTINCT x.session_pk) FROM session_record s JOIN session_stated x ON x.stated_id=s.session_id JOIN input_ref i ON i.pk=s.input_pk WHERE i.id IN (SELECT value FROM json_each(?1))",[&ids])?;
        let documents: u64 = self.conn()?.scalar("SELECT COUNT(DISTINCT s.document_id) FROM document_parse s JOIN input_ref i ON i.pk=s.input_pk WHERE i.id IN (SELECT value FROM json_each(?1))",[&ids])?;
        Ok(ImportCounts {
            sessions,
            documents,
        })
    }
}
