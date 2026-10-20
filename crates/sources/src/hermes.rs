//! Hermes: its `state.db`, captured as a logical export (architecture 4.2)
//! and stored in full. The rows are not read into sessions yet; each row
//! line is accounted as kept in the stored export.

use crate::frame::Record;
use crate::parse::{Output, Parse, Parser};
use crate::{Disposition, Records};
use polylogue_model::Json;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const LOCATIONS: &[&str] = &[".hermes/state.db"];

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    path.file_name().is_some_and(|n| n == "state.db.jsonl")
        && head
            .iter()
            .find(|b| !b.is_ascii_whitespace())
            .is_none_or(|&b| b == b'{')
}

pub fn open(_: &Path) -> Option<Parse> {
    Some(Parse::Lines(Box::new(HermesDatabase)))
}

pub struct HermesDatabase;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoState {}

impl Parser for HermesDatabase {
    type Carry = NoState;

    fn kind(&self, v: &Json) -> String {
        if v.get("schema").is_some() {
            "schema".into()
        } else {
            v.get("t").and_then(Json::as_str).unwrap_or("?").to_owned()
        }
    }

    fn parse_record(
        &self,
        _: &mut NoState,
        _: Record<'_>,
        v: &Json,
        _: &mut Output,
        _: &mut Records,
    ) -> Disposition {
        if v.get("schema").is_some() {
            Disposition::Materialized
        } else if v.get("t").is_some() {
            Disposition::unknown("kept in the stored export; rows not read into sessions yet")
        } else {
            Disposition::refused("not an export line")
        }
    }

    fn flush(&self, _: &mut NoState, _: &mut Output) {}
}
