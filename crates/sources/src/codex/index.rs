//! Codex's two thread-wide files. `history.jsonl` keeps every prompt typed
//! into any thread (`{session_id, ts, text}`), including threads whose
//! rollout is gone; each becomes a `prompt` event of its thread.
//! `session_index.jsonl` names threads (`{id, thread_name, updated_at}`);
//! a later line renames.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

parser_modules! {}

use super::session_id;
use crate::lines::Record;
use crate::parse::{Output, Parser};
use crate::time;
use crate::{RecordOutcome, Records};
use polylogue_model::JsonExt;
use polylogue_model::{Event, Json, Session, SessionKind, TitleSource};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

static NAME_FIELDS: crate::fields::Fields = crate::fields::Fields::new(
    "name",
    &[
        (
            "id",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "thread_name",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "updated_at",
            crate::fields::Rule::Excluded("Codex index edit time; commit order chooses the name"),
        ),
    ],
);

pub struct History;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoState {}

impl Parser for History {
    type ParserState = NoState;

    fn kind(&self, _: &Json) -> String {
        "prompt".into()
    }

    fn parse_record(
        &self,
        _: &mut NoState,
        record: Record<'_>,
        v: &Json,
        out: &mut Output,
        _: &mut Records,
    ) -> RecordOutcome {
        let Some(session) = v
            .str("session_id")
            .filter(|s| !s.is_empty())
            .and_then(session_id)
        else {
            return RecordOutcome::unreadable("prompt without a session ID");
        };
        if v.str("text").is_none() {
            return RecordOutcome::skipped("prompt without text");
        }
        out.events.push({
            let (at, time_accuracy) = crate::time::parts(v.get("ts").and_then(time::epoch_seconds));
            Event {
                session: Some(session),
                kind: "prompt".into(),
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
                after: None,
                range: Some((record.offset, record.offset + record.bytes.len() as u64)),
                data: v.clone(),
            }
        });
        RecordOutcome::Used
    }

    fn flush(&self, _: &mut NoState, _: &mut Output) {}
}

pub struct SessionIndex;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Names {
    /// The latest name per thread, resent in full after every step so the
    /// writer's merge sees the newest.
    names: BTreeMap<String, String>,
}

impl Parser for SessionIndex {
    type ParserState = Names;

    fn kind(&self, _: &Json) -> String {
        "thread_name".into()
    }

    fn parse_record(
        &self,
        parser_state: &mut Names,
        _: Record<'_>,
        v: &Json,
        _: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        let (Some(id), Some(name)) = (
            v.str("id").filter(|s| !s.is_empty()),
            v.str("thread_name").filter(|s| !s.is_empty()),
        ) else {
            return RecordOutcome::skipped("index line without a thread ID or name");
        };
        if session_id(id).is_none() {
            return RecordOutcome::unreadable("thread ID is empty");
        }
        records.fields("thread_name", v, &NAME_FIELDS);
        parser_state.names.insert(id.to_owned(), name.to_owned());
        RecordOutcome::Used
    }

    fn flush(&self, parser_state: &mut Names, out: &mut Output) {
        for (id, name) in &parser_state.names {
            if let Some(sid) = session_id(id) {
                let mut s = Session::new(sid, SessionKind::Main);
                s.offer_title(name, TitleSource::Generated);
                out.sessions.push(s);
            }
        }
    }
}
