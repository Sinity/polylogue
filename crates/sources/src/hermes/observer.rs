//! ATOF runtime observations are not conversations. Native run/scope IDs stay
//! in the observation; it belongs to its input when no conversation is proved.
use super::{Event, Json, Parse, Sink, lines};
use crate::Disposition;
use crate::export::{ExportParser, ExportState, Input, ItemSink, ScanError, ScanStats};
use crate::fields::{Fields, Rule};
use std::path::Path;

pub(super) fn recognize(path: &Path, head: &[u8]) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "jsonl")
        && (crate::document::first_key(head) == Some("atof_version")
            || path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().ends_with(".atof.jsonl"))
            || (path.file_name().is_some_and(|name| name == "events.jsonl")
                && path
                    .components()
                    .any(|part| part.as_os_str() == "nemo-relay")))
}

pub(super) fn open(path: &Path) -> Option<Parse> {
    path.extension().filter(|extension| *extension == "jsonl")?;
    Some(Parse::Export(Box::new(Atof)))
}

struct Atof;

static FIELDS: Fields = Fields::new(&[
    ("atof_version", Rule::Read),
    ("kind", Rule::Read),
    ("uuid", Rule::Read),
    ("parent_uuid", Rule::Read),
    ("timestamp", Rule::Read),
    ("name", Rule::Read),
    ("scope_category", Rule::Read),
    ("category", Rule::Read),
    ("data", Rule::Read),
    ("data_schema", Rule::Kept("native observation schema")),
    (
        "metadata",
        Rule::Kept("native run metadata does not prove a conversation association"),
    ),
    ("attributes", Rule::Kept("native observation attributes")),
    ("category_profile", Rule::Kept("native observation profile")),
]);

fn kind(value: &Json) -> Option<&'static str> {
    match value.get("kind")?.as_str()? {
        "scope" => match (
            value.get("category")?.as_str()?,
            value.get("scope_category")?.as_str()?,
        ) {
            ("tool", "start") => Some("hermes_tool_start"),
            ("tool", "end") => Some("hermes_tool_end"),
            ("llm", "start") => Some("hermes_llm_start"),
            ("llm", "end") => Some("hermes_llm_end"),
            ("agent", "start") => Some("hermes_agent_start"),
            ("agent", "end") => Some("hermes_agent_end"),
            (_, "start" | "end") => Some("hermes_observer_scope"),
            _ => None,
        },
        "mark" => Some(match value.get("name").and_then(Json::as_str) {
            Some("hermes.approval.request") => "hermes_approval_request",
            Some("hermes.approval.response") => "hermes_approval_response",
            Some("hermes.subagent.start") => "hermes_subagent_start",
            Some("hermes.subagent.stop") => "hermes_subagent_end",
            Some("hermes.api.error") => "hermes_api_error",
            Some("hermes.api.response.unmatched" | "hermes.tool.response.unmatched") => {
                "hermes_unmatched_response"
            }
            Some("hermes.turn.start") => "hermes_turn_start",
            Some("hermes.turn.end") => "hermes_turn_end",
            Some("hermes.session.start") => "hermes_session_start",
            Some("hermes.session.end") => "hermes_session_end",
            _ => "hermes_observer_mark",
        }),
        _ => None,
    }
}

impl ExportParser for Atof {
    fn scan(
        &mut self,
        _: &str,
        reader: &mut dyn Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        lines(reader, |range, _, raw| {
            item.push(range, Vec::new(), None, raw)
        })
    }

    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let value: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if value.get("atof_version").and_then(Json::as_str) != Some("0.1") {
            sink.records.record(
                "atof",
                &Disposition::unknown("unsupported ATOF version; original record kept"),
            );
            return Ok(());
        }
        let Some(kind) = kind(&value).filter(|_| {
            value
                .get("uuid")
                .and_then(Json::as_str)
                .is_some_and(|id| !id.is_empty())
        }) else {
            sink.records.record(
                "atof",
                &Disposition::unknown(
                    "unsupported ATOF envelope or scope phase; original record kept",
                ),
            );
            return Ok(());
        };
        sink.records.fields("atof", &value, &FIELDS);
        let at = value.get("timestamp").and_then(crate::time::rfc3339);
        if value.get("timestamp").is_some_and(|time| !time.is_null()) && at.is_none() {
            sink.out.events.push(Event {
                session: None, kind: "hermes_import_diagnostic".into(), at: None, after: None, range: None,
                data: serde_json::json!({"field":"timestamp","reason":"ATOF RFC3339 time unreadable; time remains unknown"}),
            });
        }
        sink.out.events.push(Event {
            session: None,
            kind: kind.into(),
            at,
            after: None,
            range: None,
            data: value,
        });
        sink.records.record("atof", &Disposition::Materialized);
        Ok(())
    }
}
