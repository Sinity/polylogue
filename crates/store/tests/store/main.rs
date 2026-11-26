//! The store read through query text over archives the real engine built:
//! Morrow (`examples/morrow/`, its numbers in `FACTS.md`) and small homes
//! of hand-written Claude Code lines. Each test compares whole rows,
//! numbers or typed refusals with values worked out from those inputs and
//! the docs; none names a stored table or column.

mod aggregates;
mod complements;
mod count_cache;
mod facets;
mod fields;
mod measures;
mod saved;

use polylogue_store::Reader;
use polylogue_store::query::{Options, Program, execute::present};
use polylogue_store::selection::SelectionError;
use polylogue_testkit::short;
use serde_json::Value;

/// The Morrow examples' clock: 2026-10-08 12:00 in Europe/Warsaw, the
/// logical day starting at 04:00 (`FACTS.md`, times the examples use).
pub fn options() -> Options {
    Options {
        clock: query_language::time::Clock::new(
            "2026-10-08T10:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("Europe/Warsaw").unwrap(),
                jiff::civil::time(4, 0, 0, 0),
            )
            .unwrap(),
        ),
        origins: polylogue_sources::origin::ORIGINS
            .iter()
            .map(|origin| origin.name.to_owned())
            .collect(),
        page_size: Some(1000),
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        here: None,
        seen_marks_of: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        provider_catalog: None,
    }
}

/// `text` compiled as the daemon compiles it: bound against the reader's
/// saved definitions, then lowered, showing `fields` (the unit's default
/// columns when empty).
pub fn program(
    reader: &Reader<'_>,
    text: &str,
    fields: &[&str],
) -> Result<Program, SelectionError> {
    let mut options = options();
    options.fields = fields.iter().map(|field| (*field).to_owned()).collect();
    let plan = reader.prepare_query_plan(text, &options)?;
    reader.lower_plan(&plan, &options)
}

/// Every row `text` answers, as the reply shows it.
pub fn rows(reader: &Reader<'_>, text: &str, fields: &[&str]) -> Vec<Value> {
    let program = program(reader, text, fields).unwrap_or_else(|error| panic!("{text}: {error}"));
    let mut rows = reader
        .query_rows(&program)
        .unwrap_or_else(|error| panic!("{text}: {error}"));
    for row in &mut rows {
        present(&program, row).unwrap();
    }
    rows
}

/// The sessions `text` lists, in reply order, as `#n` for Morrow's sessions.
pub fn session_list(reader: &Reader<'_>, text: &str) -> Vec<String> {
    rows(reader, text, &["id"])
        .iter()
        .map(|row| short(row["id"].as_str().expect("a session ID")))
        .collect()
}
