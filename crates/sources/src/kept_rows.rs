//! Tool databases whose rows are captured but not yet read into sessions:
//! their logical export (architecture 4.2) is stored in full and each row
//! line is accounted as kept. Reading Hermes sessions and messages from
//! these rows is the Hermes parser's job.

use crate::account::{Accounting, Disposition};
use crate::frame::Record;
use crate::parse::{Output, Parser, StepParser};
use polylogue_model::Json;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const HERMES_LOCATIONS: &[&str] = &[".hermes/state.db"];

pub fn recognize_hermes(path: &Path, head: &[u8]) -> bool {
    path.file_name().is_some_and(|n| n == "state.db.jsonl")
        && head
            .iter()
            .find(|b| !b.is_ascii_whitespace())
            .is_none_or(|&b| b == b'{')
}

pub fn open(_: &Path) -> Option<Box<dyn StepParser>> {
    Some(Box::new(KeptRows))
}

pub struct KeptRows;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoCarry {}

impl Parser for KeptRows {
    type Carry = NoCarry;

    fn kind(&self, v: &Json) -> String {
        if v.get("schema").is_some() {
            "schema".into()
        } else {
            v.get("t").and_then(Json::as_str).unwrap_or("?").to_owned()
        }
    }

    fn record(
        &self,
        _: &mut NoCarry,
        _: Record<'_>,
        v: &Json,
        _: &mut Output,
        _: &mut Accounting,
    ) -> Disposition {
        if v.get("schema").is_some() {
            Disposition::Materialized
        } else if v.get("t").is_some() {
            Disposition::unknown("kept in the stored export; rows not read into sessions yet")
        } else {
            Disposition::refused("not an export line")
        }
    }

    fn flush(&self, _: &mut NoCarry, _: &mut Output) {}
}
