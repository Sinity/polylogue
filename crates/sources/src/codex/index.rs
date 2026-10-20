//! Codex's two thread-wide files. `history.jsonl` keeps every prompt typed
//! into any thread (`{session_id, ts, text}`), including threads whose
//! rollout is gone; each becomes a `prompt` event of its thread.
//! `session_index.jsonl` names threads (`{id, thread_name, updated_at}`);
//! a later line renames.

use super::{items::str_field, session_id};
use crate::frame::Record;
use crate::parse::{Output, Parser};
use crate::time;
use crate::{Disposition, Records};
use polylogue_model::{Event, Json, Session, SessionKind, TitleSource};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub struct History;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoState {}

impl Parser for History {
    type Carry = NoState;

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
    ) -> Disposition {
        let Some(session) = str_field(v, "session_id").and_then(session_id) else {
            return Disposition::refused("prompt without a session ID");
        };
        if v.get("text").and_then(Json::as_str).is_none() {
            return Disposition::unknown("prompt without text");
        }
        out.events.push(Event {
            session,
            kind: "prompt".into(),
            at: v.get("ts").and_then(time::epoch_seconds),
            after: None,
            range: Some((record.offset, record.offset + record.bytes.len() as u64)),
            data: v.clone(),
        });
        Disposition::Materialized
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
    type Carry = Names;

    fn kind(&self, _: &Json) -> String {
        "thread_name".into()
    }

    fn parse_record(
        &self,
        carry: &mut Names,
        _: Record<'_>,
        v: &Json,
        _: &mut Output,
        _: &mut Records,
    ) -> Disposition {
        let (Some(id), Some(name)) = (str_field(v, "id"), str_field(v, "thread_name")) else {
            return Disposition::unknown("index line without a thread ID or name");
        };
        if session_id(id).is_none() {
            return Disposition::refused("thread ID is empty");
        }
        carry.names.insert(id.to_owned(), name.to_owned());
        Disposition::Materialized
    }

    fn flush(&self, carry: &mut Names, out: &mut Output) {
        for (id, name) in &carry.names {
            if let Some(sid) = session_id(id) {
                let mut s = Session::new(sid, SessionKind::Main);
                s.offer_title(name, TitleSource::Generated);
                out.sessions.push(s);
            }
        }
    }
}
