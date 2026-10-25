//! Hermes's verification ledger contains observations, not conversation turns.
use super::{Event, Json, Parse, Sink, lines};
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
        lines(reader, |range, _, raw| {
            items.push(range.clone(), range.start.to_be_bytes().to_vec(), None, raw)
        })
    }

    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let value: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let table = value.get("t").and_then(Json::as_str).unwrap_or("export");
        if table == "meta" && value["r"]["key"] == "schema_version" {
            self.version = Some(value["r"]["value"].clone());
        }
        let deletion = value.get("gone") == Some(&Json::Bool(true));
        if deletion && table == "meta" {
            // Native schema 1 has only its schema-version meta row. A deleted
            // or opaque meta key cannot certify the schema of later rows.
            self.version = None;
        }
        let (kind, time) = if deletion {
            ("hermes_verification_row_deleted", None)
        } else if value.get("schema").is_some_and(Json::is_object) {
            // This is metadata about how the rows were read, not a provider record.
            ("hermes_verification_export", None)
        } else if self.version.as_ref().and_then(Json::as_str) != Some("1") {
            sink.records.record(
                table,
                &Disposition::unknown(
                    "unsupported or absent Hermes verification schema version; original rows kept",
                ),
            );
            return Ok(());
        } else {
            match table {
                "verification_events" => ("hermes_verification_event", Some("created_at")),
                "verification_state" => ("hermes_verification_state", Some("last_edit_at")),
                "meta" => {
                    sink.records.record(table, &Disposition::Materialized);
                    return Ok(());
                }
                _ => {
                    sink.records.record(
                        table,
                        &Disposition::unknown("unrecognized Hermes verification table"),
                    );
                    return Ok(());
                }
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
                data: serde_json::json!({"field":field,"reason":"verification RFC3339 time unreadable; time remains unknown"}),
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
