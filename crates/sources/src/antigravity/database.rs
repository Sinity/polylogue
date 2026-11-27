//! Antigravity CLI summary rows, captured by the existing SQLite row exporter.
//! Summaries offer metadata, not messages or a guessed session start.
use crate::document::session_id;
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::lines::Record;
use crate::parse::{Output, Parse, Parser};
use crate::{RecordOutcome, Records};
use polylogue_model::{Event, Json, JsonExt, Link, LinkKind, Session, SessionKind, TitleSource};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const VERSION: u32 = 1;

parser_modules! {}
pub const LOCATIONS: &[&str] = &[".gemini/antigravity-cli/conversation_summaries.db"];

pub fn is_database(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| name == "conversation_summaries.db")
}

pub fn recognize(path: &Path, head: &crate::document::Head<'_>) -> bool {
    let head = head.bytes;
    path.file_name()
        .is_some_and(|name| name == "conversation_summaries.db.jsonl")
        && crate::document::starts_object_lines(head)
}

pub fn is_row_dump(path: &Path) -> bool {
    is_database(path) || recognize(path, &crate::document::Head::new(b"{"))
}
pub fn open(path: &Path) -> Option<Parse> {
    is_row_dump(path).then(|| Parse::Lines(Box::new(Summaries)))
}

struct Summaries;
#[derive(Clone, Default, Serialize, Deserialize)]
struct NoState;

static FIELDS: Fields = Fields::new(
    "summary",
    &[
        ("conversation_id", Rule::Read(T::String)),
        ("title", Rule::Read(T::String)),
        ("workspace_uris", Rule::Read(T::String)),
        ("parent_conversation_id", Rule::Read(T::String)),
        ("last_modified_time", Rule::Read(T::String)),
        ("preview", Rule::Kept("summary event")),
        ("step_count", Rule::Kept("summary event")),
        ("status", Rule::Kept("summary event")),
        ("source", Rule::Kept("summary event")),
        ("project_id", Rule::Kept("summary event")),
        ("agent_name", Rule::Kept("summary event")),
        ("nesting_depth", Rule::Kept("summary event")),
        ("battle_id", Rule::Kept("summary event")),
        ("winning_conversation_id", Rule::Kept("summary event")),
        ("not_fully_idle", Rule::Kept("summary event")),
        ("killed", Rule::Kept("summary event")),
        ("last_user_input_time", Rule::Kept("summary event")),
        ("last_user_input_step_index", Rule::Kept("summary event")),
        ("app_data_dir", Rule::Kept("summary event")),
    ],
);

impl Parser for Summaries {
    type ParserState = NoState;
    fn kind(&self, value: &Json) -> String {
        if value.get("schema").is_some() {
            "schema".into()
        } else {
            value.str("t").unwrap_or("?").into()
        }
    }
    fn parse_record(
        &self,
        _: &mut NoState,
        record: Record<'_>,
        value: &Json,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        if value.get("schema").is_some() {
            return RecordOutcome::Used;
        }
        let Some(table) = value.str("t") else {
            return RecordOutcome::unreadable("row dump without a table");
        };
        if value.get("gone").is_some() {
            return RecordOutcome::skipped(
                "row removed in the source; its captured history remains",
            );
        }
        let Some(row) = value.get("r").filter(|row| row.is_object()) else {
            return RecordOutcome::unreadable("row dump without row values");
        };
        if table != "conversation_summaries" {
            records.unknown_table(table, table, row);
            return RecordOutcome::skipped("unknown summary table retained in the row dump");
        }
        records.fields(table, row, &FIELDS);
        let Some(id) = row
            .str("conversation_id")
            .filter(|id| !id.is_empty())
            .and_then(|id| session_id(super::ORIGIN, id))
        else {
            return RecordOutcome::unreadable("summary without a conversation ID");
        };
        let parent = row
            .str("parent_conversation_id")
            .filter(|id| !id.is_empty())
            .and_then(|id| session_id(super::ORIGIN, id));
        let mut session = Session::new(
            id.clone(),
            if parent.is_some() {
                SessionKind::Subagent
            } else {
                SessionKind::Main
            },
        );
        if let Some(title) = row.str("title") {
            session.offer_title(title, TitleSource::Summary);
        }
        if let Some(uris) = row
            .str("workspace_uris")
            .and_then(|text| serde_json::from_str::<Vec<String>>(text).ok())
        {
            for uri in uris {
                if let Some(path) = uri
                    .strip_prefix("file://")
                    .filter(|path| path.starts_with('/'))
                {
                    session.offer_cwd(path);
                }
            }
        }
        out.sessions.push(session);
        if let Some(parent) = parent {
            out.links.push(Link {
                kind: LinkKind::Spawn,
                copies_prefix: false,
                reason: None,
                parent,
                child: id.clone(),
                branch_point: None,
                call_id: None,
                evidence: "parent_conversation_id".into(),
            });
        }
        let (at, evidence) =
            crate::time::parts(row.get("last_modified_time").and_then(crate::time::rfc3339));
        out.events.push(Event {
            session: Some(id),
            kind: "antigravity_summary".into(),
            at,
            at_quality: evidence.quality,
            at_precision: evidence.precision,
            at_until: evidence.until,
            after: None,
            range: Some((record.offset, record.offset + record.bytes.len() as u64)),
            data: row.clone(),
        });
        RecordOutcome::Used
    }
    fn flush(&self, _: &mut NoState, _: &mut Output) {}
}

#[cfg(test)]
mod tests;
