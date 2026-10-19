//! Polylogue's repository lookup record (spec sources/README.md, the
//! repo-lookup record): one JSON document per repository directory, written
//! by the engine from what `.git` says on this host. Storing the lookup as
//! an input keeps the join to sessions a function of stored bytes, so a
//! rebuild gives every session the same repository.

use crate::account::{Accounting, Disposition};
use crate::document::{Sink, str_of, string_of};
use crate::origin::Root;
use crate::parse::{DocumentParser, Output, StepParser, Whole};
use polylogue_model::{Json, RepoLookup};
use std::path::Path;

pub const ORIGIN: &str = "polylogue-repo-lookup";

pub const ROOT: Root = Root {
    env: Some("XDG_STATE_HOME"),
    default: ".local/state",
};

pub const LOCATIONS: &[&str] = &["polylogue/repo-lookup/*.json"];

/// The document Polylogue writes: `{"path": …, "remote": …, …}`.
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    path.extension().is_some_and(|e| e == "json")
        && crate::document::first_key(head) == Some("path")
}

pub fn open(_path: &Path) -> Option<Box<dyn StepParser>> {
    Some(Box::new(Whole(Lookup)))
}

pub struct Lookup;

impl DocumentParser for Lookup {
    fn document(&self, bytes: &[u8], out: &mut Output, acc: &mut Accounting) {
        let mut sink = Sink::new(out, acc);
        let v: Json = match serde_json::from_slice(bytes) {
            Ok(v) => v,
            Err(e) => {
                sink.record("repo_lookup", &Disposition::refused(e.to_string()));
                return;
            }
        };
        let Some(path) = string_of(&v, "path") else {
            sink.record("repo_lookup", &Disposition::refused("no path"));
            return;
        };
        // `read_at` says when the lookup was taken; the row needs only what
        // it found.
        sink.out.repo_lookups.push(RepoLookup {
            path,
            remote: str_of(&v, "remote").map(str::to_owned),
            root_commit: str_of(&v, "root_commit").map(str::to_owned),
        });
        sink.record("repo_lookup", &Disposition::Materialized);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lookup_document_becomes_one_row() {
        let doc = br#"{"path":"/work/repo","remote":"https://github.com/o/r.git","root_commit":"abc123","read_at":"2026-10-19T00:00:00Z"}"#;
        assert!(recognize(Path::new("/s/polylogue/repo-lookup/x.json"), doc));
        let mut out = Output::default();
        let mut acc = Accounting::default();
        Lookup.document(doc, &mut out, &mut acc);
        assert_eq!(
            out.repo_lookups,
            [RepoLookup {
                path: "/work/repo".into(),
                remote: Some("https://github.com/o/r.git".into()),
                root_commit: Some("abc123".into()),
            }]
        );
        assert_eq!(acc.framed, 1);
    }

    #[test]
    fn a_lookup_without_a_remote_keeps_its_root() {
        let doc = br#"{"path":"/work/local","remote":null,"root_commit":"def","read_at":"x"}"#;
        let mut out = Output::default();
        Lookup.document(doc, &mut out, &mut Accounting::default());
        assert_eq!(out.repo_lookups[0].remote, None);
        assert_eq!(out.repo_lookups[0].root_commit.as_deref(), Some("def"));
    }
}
