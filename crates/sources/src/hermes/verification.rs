//! Hermes's verification ledger contains observations, not conversation turns.
use super::{Event, Json, Parse, Sink, key, lines};
use crate::Disposition;
use crate::export::{ExportParser, ExportState, Input, ItemSink, ScanError, ScanStats};
use std::path::Path;

pub(super) fn recognize(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| name == "verification_evidence.db.jsonl")
}

pub(super) fn open() -> Parse {
    Parse::Export(Box::new(Verification::default()))
}

#[derive(Default)]
struct Verification {
    version: Option<Json>,
}

impl ExportParser for Verification {
    fn scan(
        &mut self,
        _: &str,
        reader: &mut dyn Input,
        items: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        self.version = None;
        let stats = lines(reader, |_, value, _| {
            if value.get("t").and_then(Json::as_str) == Some("meta")
                && (value["r"]["key"] == "schema_version"
                    || value.get("k") == Some(&serde_json::json!(["schema_version"])))
            {
                self.version = (value.get("gone") != Some(&Json::Bool(true)))
                    .then(|| value["r"]["value"].clone());
            }
            Ok(())
        })?;
        reader
            .rewind()
            .map_err(|e| ScanError::Input(serde_json::Error::io(e)))?;
        lines(reader, |range, value, raw| {
            let row_key = key(value);
            items.push(range, Vec::new(), row_key.as_deref(), raw)
        })?;
        Ok(stats)
    }

    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let value: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let table = value.get("t").and_then(Json::as_str).unwrap_or("export");
        if self.version.as_ref().and_then(Json::as_str) != Some("1") {
            sink.records.record(table, &Disposition::unknown(
                "unsupported or absent Hermes verification schema version; original rows retained",
            ));
            return Ok(());
        }
        if value.get("gone") == Some(&Json::Bool(true)) {
            sink.records.record(table, &Disposition::unknown(
                "provider deleted verification row; deletion remains retained without inventing its prior value",
            ));
            return Ok(());
        }
        let (kind, time) = match table {
            "verification_events" => ("hermes_verification_event", Some("created_at")),
            "verification_state" => ("hermes_verification_state", Some("last_edit_at")),
            "meta" => {
                sink.records.record(table, &Disposition::Materialized);
                return Ok(());
            }
            _ if value.get("schema").is_some() => ("hermes_verification_export", None),
            _ => {
                sink.records.record(
                    table,
                    &Disposition::unknown("unrecognized Hermes verification table"),
                );
                return Ok(());
            }
        };
        if time.is_some() && !value.get("r").is_some_and(Json::is_object) {
            sink.records.record(
                table,
                &Disposition::refused("verification row is not an object"),
            );
            return Ok(());
        }
        let at = time
            .and_then(|field| value["r"].get(field))
            .and_then(crate::time::rfc3339);
        if let Some(field) = time
            && value["r"].get(field).is_some_and(|v| !v.is_null())
            && at.is_none()
        {
            sink.out.events.push(Event {
                session: None, kind: "hermes_import_diagnostic".into(), at: None, after: None, range: None,
                data: serde_json::json!({"field":field,"interpretation":"verification RFC3339 time unreadable; time remains unknown"}),
            });
        }
        // Preserve status, exit code and changed_paths_json as supplied. Missing
        // or future values do not become a tool outcome or an empty path list.
        sink.records.record(table, &Disposition::Materialized);
        sink.out.events.push(Event {
            session: None,
            kind: kind.into(),
            at,
            after: None,
            range: None,
            data: value,
        });
        Ok(())
    }
}
