//! Codex's two thread-wide files. `history.jsonl` keeps every prompt typed
//! into any thread (`{session_id, ts, text}`), including threads whose
//! rollout is gone; each becomes a `prompt` event of its thread.
//! `session_index.jsonl` names threads (`{id, thread_name, updated_at}`);
//! a later line renames.

use super::{items::str_field, session_id};
use crate::lines::Record;
use crate::parse::{Output, Parser};
use crate::time;
use crate::{RecordOutcome, Records};
use polylogue_model::{Event, Json, Session, SessionKind, TitleSource};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

static NAME_FIELDS: crate::fields::Fields = crate::fields::Fields::new(&[
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
        crate::fields::Rule::Excluded("native index edit time; receipt order chooses the name"),
    ),
])
.for_kind("name");

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
        let Some(session) = str_field(v, "session_id").and_then(session_id) else {
            return RecordOutcome::unreadable("prompt without a session ID");
        };
        if v.get("text").and_then(Json::as_str).is_none() {
            return RecordOutcome::skipped("prompt without text");
        }
        out.events.push(Event {
            session: Some(session),
            kind: "prompt".into(),
            at: v.get("ts").and_then(time::epoch_seconds),
            after: None,
            range: Some((record.offset, record.offset + record.bytes.len() as u64)),
            data: v.clone(),
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
        let (Some(id), Some(name)) = (str_field(v, "id"), str_field(v, "thread_name")) else {
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
