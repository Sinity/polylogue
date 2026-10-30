//! Exact predecessor run evidence never supplies current cursor authority.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum HistoricalRunReplay {
    Compatible { run: Box<QueryRun> },
    Unavailable { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HistoricalQueryRun {
    pub id: crate::RunId,
    pub question: QuestionId,
    pub source: crate::historical::SourceSchema,
    /// Exact available nested run encoding, not an acquisition-byte claim.
    pub encoded_run: Vec<u8>,
    /// Optional summary claims from the original payload, distinct from the
    /// actual fetched native record hash and projection evidence.
    pub declared_binding_hash: Option<crate::BlobHash>,
    pub declared_run_hash: Option<crate::BlobHash>,
    pub native: crate::historical::NativeProjectionHistory,
    #[serde(default)]
    pub imports: crate::authored::ImportHistory,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cached: Vec<crate::historical::CachedNativeContext>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub records: Vec<Vec<u8>>,
    pub replay: HistoricalRunReplay,
}
impl HistoricalQueryRun {
    pub fn validate(&self) -> Result<(), String> {
        let body: serde_json::Value =
            crate::historical::read_json(&self.encoded_run).map_err(|error| error.to_string())?;
        if body.get("id").and_then(serde_json::Value::as_str) != Some(self.id.as_str())
            || body
                .get("question")
                .cloned()
                .and_then(|value| serde_json::from_value::<QuestionId>(value).ok())
                != Some(self.question)
            || !matches!(
                body.pointer("/binding/kind")
                    .and_then(serde_json::Value::as_str),
                Some("original" | "continuation")
            )
        {
            return Err(
                "historical run disagrees with its recorded identity or source shape".into(),
            );
        }
        match &self.replay {
            HistoricalRunReplay::Compatible { run } => {
                if run.id != self.id
                    || run.question != Some(self.question)
                    || serde_json::to_value(run).map_err(|error| error.to_string())? != body
                {
                    return Err("compatible run changed the complete predecessor value".into());
                }
                if self
                    .declared_binding_hash
                    .is_some_and(|hash| run.binding_hash().ok() != Some(hash))
                    || self
                        .declared_run_hash
                        .is_some_and(|hash| run.record_hash().ok() != Some(hash))
                {
                    return Err(
                        "compatible run disagrees with its declared integrity claims".into(),
                    );
                }
            }
            HistoricalRunReplay::Unavailable { reason } if reason.is_empty() => {
                return Err("unavailable historical replay requires its reason".into());
            }
            HistoricalRunReplay::Unavailable { .. } => {}
        }
        Ok(())
    }
}
