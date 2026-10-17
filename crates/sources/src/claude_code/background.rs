//! Claude's structured background notification, never command-output prose.
use polylogue_model::{ErrorReason, ToolOutcome};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Leaf {
    Outcome,
    ExitCode,
    Status,
    OutputFile,
    TaskId,
}
pub struct Completion<'a> {
    pub task: &'a str,
    pub call: &'a str,
    pub status: &'a str,
    pub output_file: Option<&'a str>,
    pub exit: Option<i64>,
}
fn tag<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let (_, rest) = text.split_once(&open)?;
    let (value, tail) = rest.split_once(&close)?;
    (!tail.contains(&open) && !value.contains('<')).then_some(value)
}
pub fn parse(text: &str) -> Option<Completion<'_>> {
    let text = text
        .trim()
        .strip_prefix("<task-notification>")?
        .strip_suffix("</task-notification>")?;
    let status = tag(text, "status")?;
    if !matches!(status, "completed" | "failed" | "killed" | "cancelled") {
        return None;
    }
    let summary = tag(text, "summary");
    let exit = summary
        .and_then(|s| s.rsplit_once("exit code "))
        .and_then(|(_, n)| n.trim().parse().ok());
    Some(Completion {
        task: tag(text, "task-id")?,
        call: tag(text, "tool-use-id")?,
        status,
        output_file: tag(text, "output-file"),
        exit,
    })
}
impl Completion<'_> {
    pub fn outcome(&self) -> ToolOutcome {
        match (self.exit, self.status) {
            (Some(0), _) | (None, "completed") => ToolOutcome::Ok,
            (_, "killed" | "cancelled") => ToolOutcome::Error {
                reason: Some(ErrorReason::Interrupted),
            },
            _ => ToolOutcome::Error { reason: None },
        }
    }
}

use super::{
    KeyedRecords,
    keyed::*,
    projection::{Fact, Field},
    selection,
};
use crate::{frame::Record, selection::RecordRange};
use polylogue_model::{BlockContent, RetainedJson as Value, SessionKind};
impl<R: ReadView> KeyedRecords<R> {
    pub(super) fn remember_background(
        &mut self,
        task: &str,
        mut address: BackgroundAddress,
    ) -> Result<(), String> {
        let previous = match self.backgrounds.get(task) {
            Some(value) => Some(value.clone()),
            None => self.view.background(task).map_err(|e| e.to_string())?,
        };
        if let Some(old) = previous {
            if old != address {
                address.ambiguous = true;
            }
        }
        self.backgrounds.insert(task.into(), address);
        Ok(())
    }
    pub(super) fn complete_background(&mut self, record: &Record<'_>) -> Result<(), String> {
        let Some(text) = record.value.get("content").and_then(Value::as_str) else {
            return Ok(());
        };
        let Some(done) = parse(text) else {
            return Ok(());
        };
        let address = match self.backgrounds.get(done.task) {
            Some(value) => Some(value.clone()),
            None => self.view.background(done.task).map_err(|e| e.to_string())?,
        };
        let Some(address) = address.filter(|a| !a.ambiguous && a.call == done.call) else {
            return Ok(());
        };
        let mut response = self
            .response(&address.session, &address.response)?
            .ok_or("background response unavailable")?;
        if response.message.id != address.message {
            return Err("background response identity changed".into());
        }
        let mut contribution = self
            .contribution(&address.message, address.item)?
            .ok_or("background contribution unavailable")?;
        let raw = super::raw::canonical(record.bytes)?;
        if contribution
            .completion
            .as_ref()
            .is_some_and(|c| c.raw == raw)
        {
            return Ok(());
        }
        let Some(BlockContent::ToolResult(result)) =
            contribution.blocks.get_mut(address.local as usize)
        else {
            return Err("background result block unavailable".into());
        };
        if result.call_id.as_ref().map(|c| c.as_str()) != Some(done.call) {
            return Err("background invocation changed".into());
        }
        result.outcome = done.outcome();
        result.exit_code = done.exit;
        let data = result
            .data
            .get_or_insert_with(|| Value::Object(Default::default()))
            .as_object_mut()
            .ok_or("background result data must be an object")?;
        data.insert("background_task_id".into(), Value::from(done.task));
        data.insert(
            "background_completion_status".into(),
            Value::from(done.status),
        );
        if let Some(path) = done.output_file {
            data.insert("background_output_file".into(), Value::from(path));
        }
        let revision = response
            .revision
            .checked_add(1)
            .ok_or("background response revision overflow")?;
        let range = RecordRange {
            start: record.start,
            end: record.end,
        };
        contribution.projection.facts.retain(|fact| {
            !matches!(
                fact.selection.as_ref().map(|s| &s.transform),
                Some(selection::Transform::BackgroundV1 { .. })
            )
        });
        let mut add = |field, leaf| {
            contribution.projection.facts.push(Fact {
                field,
                source_path: vec!["content".into()],
                supplied_revision: revision,
                selection: Some(selection::background(range.clone(), leaf)),
            })
        };
        add(
            Field::Block {
                local: address.local,
                field: polylogue_model::merge::BlockField::Outcome,
            },
            Leaf::Outcome,
        );
        if done.exit.is_some() {
            add(
                Field::Block {
                    local: address.local,
                    field: polylogue_model::merge::BlockField::ExitCode,
                },
                Leaf::ExitCode,
            );
        }
        for (key, leaf) in [
            ("background_task_id", Leaf::TaskId),
            ("background_completion_status", Leaf::Status),
        ] {
            add(
                Field::ResultData {
                    local: address.local,
                    key: key.into(),
                },
                leaf,
            );
        }
        if done.output_file.is_some() {
            add(
                Field::ResultData {
                    local: address.local,
                    key: "background_output_file".into(),
                },
                Leaf::OutputFile,
            );
        }
        contribution.completion = Some(BackgroundCompletion { raw, range });
        response.revision = revision;
        response.message.version_hash = None;
        self.pending.insert(
            (
                response.message.id.session().as_str().into(),
                response.ordinal,
            ),
            PendingMessage {
                session_native: address.session.clone(),
                response: address.response.clone(),
                ordinal: response.ordinal,
                message: response.message.id.clone(),
                revision,
                items: response.items,
                incomplete: response.incomplete_bodies != 0,
            },
        );
        let index = self
            .session_index(&address.session, SessionKind::Main)?
            .ok_or("background Session unavailable")?;
        self.sessions[index].facts_pending = true;
        self.contributions
            .insert((address.message, address.item), contribution);
        self.responses
            .insert((address.session, address.response), response);
        Ok(())
    }
}

/// A metadata-only UUID supplement cannot erase a later named completion.
/// A replacement with a different invocation loses that old completion.
pub(super) fn carry(previous: Option<&Contribution>, current: &mut Contribution) {
    let Some(previous) = previous.filter(|c| c.completion.is_some()) else {
        return;
    };
    let mut carried = false;
    for (local, block) in current.blocks.iter_mut().enumerate() {
        let BlockContent::ToolResult(result) = block else {
            continue;
        };
        let Some(BlockContent::ToolResult(old)) = previous.blocks.get(local) else {
            continue;
        };
        if result.call_id != old.call_id {
            continue;
        }
        result.outcome = old.outcome.clone();
        result.exit_code = old.exit_code;
        if let Some(old_data) = old.data.as_ref().and_then(Value::as_object) {
            let data = result
                .data
                .get_or_insert_with(|| Value::Object(Default::default()))
                .as_object_mut();
            if let Some(data) = data {
                for (key, value) in old_data {
                    if matches!(
                        key.as_str(),
                        "background_task_id"
                            | "background_completion_status"
                            | "background_output_file"
                    ) {
                        data.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        carried = true;
    }
    if carried {
        current.completion = previous.completion.clone();
        current.projection.facts.extend(
            previous
                .projection
                .facts
                .iter()
                .filter(|fact| {
                    matches!(
                        fact.selection.as_ref().map(|s| &s.transform),
                        Some(selection::Transform::BackgroundV1 { .. })
                    )
                })
                .cloned(),
        );
    }
}
