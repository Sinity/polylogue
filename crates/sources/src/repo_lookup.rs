//! Polylogue's repository lookup record: one JSON document per repository directory, written
//! by the engine from what `.git` says on this host. Storing the lookup as
//! an input keeps the join to sessions a function of stored bytes, so a
//! rebuild gives every session the same repository.

use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 3;

use crate::parse::{DocumentParser, Output, Parse};
use crate::{RecordOutcome, Records};
use polylogue_model::{Json, RepoLookup};
use std::path::Path;

pub const ORIGIN: &str = "polylogue-repo-lookup";

pub const LOCATIONS: &[&str] = &["polylogue/repo-lookup/*.json"];

/// The document Polylogue writes: `{"path": …, "remote": …, …}`.
pub fn recognize(path: &Path, head: &crate::document::Head<'_>) -> bool {
    let head = head.bytes;
    path.extension().is_some_and(|e| e == "json")
        && crate::document::first_key(head) == Some("path")
}

pub fn open(_path: &Path) -> Option<Parse> {
    Some(Parse::Document(Box::new(Lookup)))
}

pub struct Lookup;

impl DocumentParser for Lookup {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut Records) {
        let v: Json = match serde_json::from_slice(bytes) {
            Ok(v) => v,
            Err(e) => {
                records.record("repo_lookup", &RecordOutcome::unreadable(e.to_string()));
                return;
            }
        };
        let Some(path) = v.str("path").filter(|s| !s.is_empty()).map(str::to_owned) else {
            records.record("repo_lookup", &RecordOutcome::unreadable("no path"));
            return;
        };
        // An empty list means no root commits were found, so a value that is
        // not a list of strings is unreadable rather than empty.
        let roots = match v.get("root_commits") {
            None | Some(Json::Null) => Some(Vec::new()),
            Some(Json::Array(roots)) => roots
                .iter()
                .map(|root| root.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>(),
            Some(_) => None,
        };
        let Some(root_commits) = roots else {
            records.record(
                "repo_lookup",
                &RecordOutcome::unreadable("root_commits is not a list of strings"),
            );
            return;
        };
        // `read_at` says when the lookup was taken; the row needs only what
        // it found.
        out.repo_lookups.push(RepoLookup {
            path,
            remote: v.str("remote").filter(|s| !s.is_empty()).map(str::to_owned),
            root_commits,
        });
        records.record("repo_lookup", &RecordOutcome::Used);
    }
}

#[cfg(test)]
mod tests;
