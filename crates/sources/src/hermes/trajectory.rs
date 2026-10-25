//! ATIF is recorded interaction evidence. A run/trajectory identifier is not
//! a conversation identifier, and a source role does not establish a writer.
mod step;

use super::{Event, Json, Parse, Sink, identity};
use crate::Disposition;
use crate::export::{self, ExportParser, ExportState, Input, ItemSink, ScanError, ScanStats};
use crate::fields::{Fields, Rule};
use crate::parse::{Output, Step};
use polylogue_model::{Document, DocumentId, DocumentPurpose, DocumentRow, NativeId, Origin};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub(super) fn recognize(path: &Path, head: &[u8]) -> bool {
    path.extension().is_some_and(|s| s == "json")
        && ((crate::document::first_key(head) == Some("schema_version")
            && head
                .iter()
                .position(|b| *b == b':')
                .is_some_and(|i| head[i + 1..].trim_ascii_start().starts_with(b"\"ATIF-v")))
            || path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("trajectory-")))
}

pub(super) fn open() -> Parse {
    Parse::Export(Box::new(Trajectory::default()))
}

#[derive(Default)]
struct Trajectory {
    input: String,
    objects: BTreeMap<u64, Facts>,
    items: Vec<Item>,
    at: usize,
    seen_steps: BTreeMap<(u64, u64), u64>,
}
struct Facts {
    start: u64,
    end: u64,
    fields: Json,
    valid: bool,
    diagnostic: Option<&'static str>,
}
struct Item {
    parent: u64,
    start: u64,
}

static ROOT: Fields = Fields::new(&[
    ("schema_version", Rule::Read),
    ("session_id", Rule::Read),
    ("trajectory_id", Rule::Read),
    (
        "agent",
        Rule::Kept("native agent configuration; only name/version/model descriptors are projected"),
    ),
    ("steps", Rule::Read),
    ("subagent_trajectories", Rule::Read),
    ("notes", Rule::Kept("native annotation")),
    ("final_metrics", Rule::Read),
    ("continued_trajectory_ref", Rule::Read),
    ("extra", Rule::Kept("native trajectory metadata")),
]);
static STEP: Fields = Fields::new(&[
    ("step_id", Rule::Read),
    ("timestamp", Rule::Read),
    ("source", Rule::Read),
    ("message", Rule::Read),
    ("model_name", Rule::Read),
    ("reasoning_effort", Rule::Read),
    ("reasoning_content", Rule::Read),
    ("tool_calls", Rule::Read),
    ("observation", Rule::Read),
    ("metrics", Rule::Read),
    ("llm_call_count", Rule::Read),
    ("is_copied_context", Rule::Read),
    ("extra", Rule::Kept("native step extensions")),
]);

fn selected(value: &Json, names: &[&str]) -> Json {
    Json::Object(
        names
            .iter()
            .filter_map(|name| value.get(*name).cloned().map(|v| ((*name).into(), v)))
            .collect(),
    )
}
fn descriptors(value: &Json) -> Json {
    let mut out = selected(
        value,
        &[
            "schema_version",
            "session_id",
            "trajectory_id",
            "continued_trajectory_ref",
        ],
    );
    if let Some(agent) = value.get("agent") {
        out["agent"] = selected(agent, &["name", "version", "model_name"]);
    }
    out
}

impl ExportParser for Trajectory {
    fn scan(
        &mut self,
        input: &str,
        reader: &mut dyn Input,
        sink: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        *self = Self {
            input: input.into(),
            ..Self::default()
        };
        use export::FieldShape::{Array, Objects};
        let (objects, stats) = export::scan_objects(
            reader,
            false,
            &[("steps", Array), ("subagent_trajectories", Objects)],
            &[
                "schema_version",
                "session_id",
                "trajectory_id",
                "agent",
                "final_metrics",
                "continued_trajectory_ref",
            ],
            |parent, _, _, range, raw| {
                let ordinal = self.items.len() as u64;
                sink.push(
                    range.clone(),
                    ordinal.to_be_bytes().to_vec(),
                    None,
                    raw.get().as_bytes(),
                )?;
                self.items.push(Item {
                    parent,
                    start: range.start,
                });
                Ok(())
            },
        )?;
        let mut ids = BTreeSet::new();
        for object in objects {
            let native = object
                .fields
                .get("trajectory_id")
                .and_then(Json::as_str)
                .filter(|s| !s.is_empty());
            let unique = native.is_none_or(|id| ids.insert((object.parent, id.to_owned())));
            let diagnostic = if object.fields.get("schema_version").and_then(Json::as_str)
                != Some("ATIF-v1.7")
            {
                Some("unsupported ATIF version; original trajectory retained")
            } else if !object.fields.get("steps").is_some_and(Json::is_array)
                || !object.fields.get("agent").is_some_and(Json::is_object)
            {
                Some("ATIF requires agent and steps; original trajectory retained")
            } else if object.parent.is_some() && native.is_none() {
                Some(
                    "embedded ATIF trajectory has no native trajectory_id; original trajectory retained",
                )
            } else if !unique {
                Some(
                    "duplicate embedded ATIF trajectory_id; original trajectory retained without guessed reference binding",
                )
            } else {
                None
            };
            self.objects.insert(
                object.start,
                Facts {
                    start: object.start,
                    end: object.end,
                    fields: object.fields,
                    valid: diagnostic.is_none(),
                    diagnostic,
                },
            );
        }
        Ok(stats)
    }

    fn begin(&mut self, length: u64, _: Option<&[u8]>) -> Result<Option<Step>, String> {
        let mut records = crate::Records::default();
        let mut output = Output::default();
        for facts in self.objects.values() {
            records.fields("atif_trajectory", &facts.fields, &ROOT);
            let data = descriptors(&facts.fields);
            if facts.valid {
                let mut data = data;
                if let Some(metrics) = facts.fields.get("final_metrics") {
                    data["final_metrics"] = metrics.clone();
                }
                output.events.push(Event {
                    session: None,
                    kind: "hermes_trajectory".into(),
                    at: None,
                    after: None,
                    range: Some((facts.start, facts.end)),
                    data,
                });
                records.record("atif_trajectory", &Disposition::Materialized);
            } else {
                records.record(
                    "atif_trajectory",
                    &Disposition::unknown(
                        facts.diagnostic.unwrap_or("uninterpreted ATIF trajectory"),
                    ),
                );
            }
        }
        Ok(Some(Step {
            output,
            accounting: records.finish().map_err(|e| e.to_string())?,
            start: 0,
            consumed: length,
            restates: false,
        }))
    }

    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let item = self
            .items
            .get(self.at)
            .ok_or("missing ATIF frame context")?;
        self.at += 1;
        let facts = self
            .objects
            .get(&item.parent)
            .ok_or("missing ATIF trajectory context")?;
        if !facts.valid {
            sink.records.record(
                "atif_step",
                &Disposition::unknown(facts.diagnostic.unwrap_or("uninterpreted ATIF trajectory")),
            );
            return Ok(());
        }
        let value: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if !value.is_object() {
            sink.records.record(
                "atif_step",
                &Disposition::unknown("ATIF step is not an object; original value retained"),
            );
            return Ok(());
        }
        sink.records.fields("atif_step", &value, &STEP);
        let (blocks, mut limits) = step::blocks(&value);
        if !matches!(
            value.get("source").and_then(Json::as_str),
            Some("system" | "user" | "agent")
        ) {
            limits.push("unrecognized source role remains a native descriptor");
        }
        if let Some(count) = value.get("llm_call_count").filter(|v| !v.is_null()) {
            if count.as_u64().is_none() {
                limits.push(
                    "llm_call_count is not a nonnegative integer; no inference count assigned",
                );
            }
            if count.as_u64() == Some(0)
                && ["metrics", "reasoning_content"]
                    .iter()
                    .any(|field| value.get(*field).is_some_and(|v| !v.is_null()))
            {
                limits.push("deterministic dispatch carries conflicting inference fields; authorship stays unknown");
            }
        }
        let step_id = value
            .get("step_id")
            .and_then(Json::as_u64)
            .filter(|n| *n > 0);
        let suffix = if let Some(n) = step_id {
            let occurrence = self.seen_steps.entry((item.parent, n)).or_default();
            let suffix = format!("step-{n}:occurrence-{occurrence}");
            if *occurrence > 0 {
                limits.push("repeated step_id remains a distinct source occurrence");
            }
            *occurrence += 1;
            suffix
        } else {
            format!("~at-{}", item.start)
        };
        let trajectory = facts
            .fields
            .get("trajectory_id")
            .and_then(Json::as_str)
            .filter(|s| !s.is_empty())
            .map(identity::component)
            .unwrap_or_else(|| format!("~at-{}", facts.start));
        let id = DocumentId::new(
            Origin::new(super::ORIGIN).map_err(|e| e.to_string())?,
            NativeId::new(&format!(
                "~input:{}:trajectory-at-{}:{trajectory}:{suffix}",
                identity::component(&self.input),
                facts.start
            ))
            .map_err(|e| e.to_string())?,
        );
        let mut document = Document::new(id, DocumentPurpose::Transcript);
        document.source_time = value.get("timestamp").and_then(crate::time::rfc3339);
        if value.get("timestamp").is_some_and(|v| !v.is_null()) && document.source_time.is_none() {
            limits.push("ATIF RFC3339 time unreadable; time remains unknown");
        }
        if step_id.is_none() {
            limits.push("step_id is not a positive integer; identity is input/range scoped");
        }
        document.blocks = blocks;
        document.metadata = selected(
            &value,
            &[
                "step_id",
                "source",
                "timestamp",
                "model_name",
                "reasoning_effort",
                "llm_call_count",
                "is_copied_context",
                "metrics",
            ],
        );
        if let Some(extra) = value.get("extra") {
            for (key, names) in [
                (
                    "invocation",
                    &[
                        "invocation_id",
                        "framework",
                        "status",
                        "start_timestamp",
                        "end_timestamp",
                    ][..],
                ),
                (
                    "ancestry",
                    &["function_id", "function_name", "parent_id", "parent_name"][..],
                ),
            ] {
                if let Some(native) = extra.get(key).filter(|v| v.is_object()) {
                    document.metadata[key] = selected(native, names);
                }
            }
        }
        if let Some(calls) = value.get("tool_calls").and_then(Json::as_array) {
            let evidence:Vec<_> = calls.iter().filter_map(|call| {
                let provider = call.get("extra")?.get("provider_data")?;
                let ids = selected(provider,&["call_id","response_item_id"]);
                (!ids.as_object()?.is_empty()).then(||serde_json::json!({"tool_call_id":call.get("tool_call_id"),"provider_data":ids}))
            }).collect();
            if !evidence.is_empty() {
                document.metadata["tool_call_evidence"] = Json::Array(evidence);
            }
        }
        document.metadata["trajectory"] = descriptors(&facts.fields);
        document.metadata["identity"] =
            serde_json::json!({"kind":"input_scoped","input":self.input});
        sink.out.documents.push(DocumentRow {
            offset: 0,
            end: 0,
            document,
        });
        if !limits.is_empty() {
            sink.out.events.push(Event {
                session: None,
                kind: "hermes_import_diagnostic".into(),
                at: None,
                after: None,
                range: None,
                data: serde_json::json!({"interpretation":limits}),
            });
        }
        let disposition = if limits.is_empty() {
            Disposition::Materialized
        } else {
            Disposition::unknown("ATIF step has retained interpretation limits")
        };
        sink.records.record("atif_step", &disposition);
        Ok(())
    }
}
