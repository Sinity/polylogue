//! Polylogue's repository lookup record: one JSON document per repository directory, written
//! by the engine from what `.git` says on this host. Storing the lookup as
//! an input keeps the join to sessions a function of stored bytes, so a
//! rebuild gives every session the same repository.

use crate::document::{str_of, string_of};
use crate::parse::{DocumentParser, Output, Parse};
use crate::{Disposition, Records};
use polylogue_model::{Json, RepoLookup};
use std::path::Path;

pub const ORIGIN: &str = "polylogue-repo-lookup";

pub const LOCATIONS: &[&str] = &["polylogue/repo-lookup/*.json"];

/// The document Polylogue writes: `{"path": …, "remote": …, …}`.
pub fn recognize(path: &Path, head: &[u8]) -> bool {
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
                records.record("repo_lookup", &Disposition::refused(e.to_string()));
                return;
            }
        };
        let Some(path) = string_of(&v, "path") else {
            records.record("repo_lookup", &Disposition::refused("no path"));
            return;
        };
        // `read_at` says when the lookup was taken; the row needs only what
        // it found.
        out.repo_lookups.push(RepoLookup {
            path,
            remote: str_of(&v, "remote").map(str::to_owned),
            root_commit: str_of(&v, "root_commit").map(str::to_owned),
        });
        records.record("repo_lookup", &Disposition::Materialized);
    }
}

#[cfg(test)]
mod tests;
