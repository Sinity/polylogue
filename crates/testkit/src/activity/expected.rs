//! The view `docs/sessions.md` says Polylogue shows for an [`Activity`],
//! derived from the description alone.
//!
//! - A session is everything that states its ID; a sidechain is a session of
//!   its own, kind `sidechain`, spawned by its main session.
//! - A message is stored once, in the session that originated it: a copy in
//!   a resume's file stores nothing new, and a read of the resume shows the
//!   copied messages first, in the file's order.
//! - Within one file a session's messages keep file order whatever their
//!   times (order times never decrease along a chain), and a copy restates
//!   the same order at the same times, so session order is file order.
//! - A response written as several records is one message named by its
//!   first record, holding every record's parts in order.
//! - A parent named by any record of a response, or by a compaction
//!   boundary, is the message it stands for.
//! - A call's outcome comes from its result, `unknown` without one.
//! - A copy without a stop reason is a response still in progress, and the
//!   finished original is shown.

use super::{Activity, Body, Part};
use crate::view::{BlockView, LinkView, MessageView, SessionView, View};
use polylogue_model::{CallOutcome, LinkKind, Role, SessionKind};
use std::collections::BTreeMap;

impl Activity {
    pub fn expected(&self) -> View {
        // What each record names: the message holding it.
        let mut message_of: BTreeMap<&str, &str> = BTreeMap::new();
        // A call's result, by call ID.
        let mut results: BTreeMap<&str, bool> = BTreeMap::new();
        for transcript in &self.sessions {
            for entry in &transcript.entries {
                for record in entry.record_uuids() {
                    message_of.insert(record, &entry.uuid);
                }
                if let Body::Compaction { boundary, .. } = &entry.body {
                    // The boundary stands for the message before it.
                    if let Some(parent) = &entry.parent {
                        message_of.insert(boundary, parent);
                    }
                }
                if let Body::Results(list) = &entry.body {
                    for result in list {
                        results.insert(&result.call, result.error);
                    }
                }
            }
        }
        let resolve = |record: &str| {
            let mut name = record;
            // A boundary leads to a record, which leads to its message.
            while let Some(next) = message_of.get(name).filter(|next| **next != name) {
                name = next;
            }
            format!("claude-code:{name}")
        };

        let mut sessions = Vec::new();
        let mut links = Vec::new();
        for (index, transcript) in self.sessions.iter().enumerate() {
            let main = format!("claude-code:{}", transcript.session);
            let side = format!("{main}/sidechain");
            let mut own = Vec::new();
            let mut side_own = Vec::new();
            for entry in &transcript.entries {
                let message = MessageView {
                    id: format!("claude-code:{}", entry.uuid),
                    role: match entry.body {
                        Body::Response { .. } => Role::Assistant,
                        _ => Role::User,
                    },
                    at: Some(entry.at),
                    parent: entry.parent.as_deref().map(resolve),
                    compaction_summary: matches!(entry.body, Body::Compaction { .. }),
                    blocks: blocks(&entry.body, &results),
                };
                if entry.sidechain {
                    side_own.push(message);
                } else {
                    own.push(message);
                }
            }
            let read = self
                .lines(index)
                .iter()
                .filter(|line| !line.entry.sidechain)
                .map(|line| format!("claude-code:{}", line.entry.uuid))
                .collect();
            sessions.push(SessionView {
                id: main.clone(),
                kind: SessionKind::Main,
                cwd: vec![transcript.cwd.clone()],
                messages: own,
                read,
            });
            if !side_own.is_empty() {
                sessions.push(SessionView {
                    id: side.clone(),
                    kind: SessionKind::Sidechain,
                    cwd: vec![transcript.cwd.clone()],
                    read: side_own.iter().map(|m| m.id.clone()).collect(),
                    messages: side_own,
                });
                links.push(LinkView {
                    kind: LinkKind::Spawn,
                    parent: main,
                    child: side,
                    copies_prefix: false,
                });
            }
        }
        View::sorted(sessions, links)
    }
}

fn blocks(body: &Body, results: &BTreeMap<&str, bool>) -> Vec<BlockView> {
    match body {
        Body::Prompt(text) => vec![BlockView::Text(text.clone())],
        Body::Compaction { summary, .. } => vec![BlockView::Text(summary.clone())],
        Body::Response { records, .. } => records
            .iter()
            .flat_map(|record| &record.parts)
            .map(|part| match part {
                Part::Text(text) => BlockView::Text(text.clone()),
                Part::Thinking(text) => BlockView::Thinking(text.clone()),
                Part::Call { id, command } => BlockView::Call {
                    id: id.clone(),
                    tool: "Bash".into(),
                    command: Some(command.clone()),
                    outcome: Some(match results.get(id.as_str()) {
                        Some(false) => CallOutcome::Ok,
                        Some(true) => CallOutcome::Error,
                        None => CallOutcome::Unknown,
                    }),
                },
            })
            .collect(),
        Body::Results(list) => list
            .iter()
            .map(|result| BlockView::Result {
                call: result.call.clone(),
                error: result.error,
                output: result.output.clone(),
            })
            .collect(),
    }
}
