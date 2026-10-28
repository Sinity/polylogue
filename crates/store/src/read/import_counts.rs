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
        let sessions=self.conn()?.query_row("SELECT COUNT(DISTINCT s.session_pk) FROM session_record s JOIN input_ref i ON i.pk=s.input_pk WHERE i.id IN (SELECT value FROM json_each(?1))",[&ids],|row|row.get(0))?;
        let documents=self.conn()?.query_row("SELECT COUNT(DISTINCT s.document_pk) FROM document_source s JOIN input_ref i ON i.pk=s.input_pk WHERE i.id IN (SELECT value FROM json_each(?1))",[&ids],|row|row.get(0))?;
        Ok(ImportCounts {
            sessions,
            documents,
        })
    }
}
