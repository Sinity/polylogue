//! The generator reaches every case it is written from.

use super::delivered_activity;
use crate::File;
use crate::activity::{Activity, Body, ChangeHow, Part};
use crate::schedule::{Schedule, Step};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use std::collections::{BTreeMap, BTreeSet};

/// The cases one generated activity and schedule hold.
fn cases(activity: &Activity, files: &[File], schedule: &Schedule) -> BTreeSet<&'static str> {
    let mut found = BTreeSet::new();
    for (index, transcript) in activity.sessions.iter().enumerate() {
        let mut replies: BTreeMap<&str, usize> = BTreeMap::new();
        let mut times: BTreeMap<bool, Vec<i64>> = BTreeMap::new();
        let mut previous_record: Option<&str> = None;
        for line in activity.lines(index) {
            let entry = line.entry;
            if line.owner == transcript.session {
                if let Some(parent) = &entry.parent {
                    found.insert("a parent link");
                    *replies.entry(parent).or_default() += 1;
                }
                if entry.sidechain {
                    found.insert("a sidechain");
                }
                let session = times.entry(entry.sidechain).or_default();
                if session.contains(&entry.at) {
                    found.insert("two records with equal times");
                }
                if session.iter().any(|at| *at > entry.at) {
                    found.insert("times out of file order");
                }
                session.push(entry.at);
            }
            match &entry.body {
                Body::Response { records, .. } if records.len() > 1 => {
                    found.insert("a split response");
                }
                Body::Results(results) => {
                    if results.iter().any(|r| r.error) {
                        found.insert("an error result");
                    }
                    if previous_record == entry.parent.as_deref()
                        && let Some(Body::Response { records, .. }) = activity
                            .sessions
                            .iter()
                            .flat_map(|t| &t.entries)
                            .find(|e| e.last_record() == previous_record.unwrap_or_default())
                            .map(|e| &e.body)
                        && records.last().is_some_and(|r| {
                            r.parts.iter().any(|p| {
                                matches!(p, Part::Call { id, .. } if results.iter().any(|res| &res.call == id))
                            })
                        })
                    {
                        found.insert("a tool result in the next record");
                    }
                }
                _ => {}
            }
            let texts = match &entry.body {
                Body::Prompt(text) => vec![text.as_str()],
                Body::Response { records, .. } => records
                    .iter()
                    .flat_map(|r| &r.parts)
                    .filter_map(|p| match p {
                        Part::Text(t) | Part::Thinking(t) => Some(t.as_str()),
                        Part::Call { .. } => None,
                    })
                    .collect(),
                _ => Vec::new(),
            };
            if texts.iter().any(|t| !t.is_ascii()) {
                found.insert("non-ASCII text");
            }
            previous_record = Some(entry.last_record());
        }
        if replies.values().any(|n| *n > 1) {
            found.insert("a fork");
        }
        if let Some(copy) = &transcript.copy {
            match copy.change.as_ref().map(|c| c.how) {
                None => found.insert("a copied prefix, exact"),
                Some(ChangeHow::Edited) => found.insert("a copied prefix, edited"),
                Some(ChangeHow::CutShort) => found.insert("a copied prefix, cut short"),
            };
            if activity.sessions[copy.from].copy.is_some() {
                found.insert("a chained resume");
            }
        }
    }
    if files.iter().any(|f| f.bytes.len() as u64 > schedule.chunk) {
        found.insert("a file above the chunk size");
    }
    for step in &schedule.steps {
        match *step {
            Step::Write { file, end } => {
                let bytes = &files[file].bytes;
                if end > 0 && end < bytes.len() && bytes[end - 1] != b'\n' {
                    found.insert("a cut inside a line");
                    if bytes[end] & 0xC0 == 0x80 {
                        found.insert("a cut inside a multi-byte character");
                    }
                }
            }
            Step::Restart => {
                found.insert("a restart between steps");
            }
            Step::Rebuild => {
                found.insert("a rebuild between steps");
            }
            Step::Store => {}
        }
    }
    found
}

/// Sixty-four cases from a fixed seed hold every listed case at least once.
#[test]
fn the_generator_reaches_every_listed_case() {
    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::from_seed(RngAlgorithm::ChaCha, &[7; 32]),
    );
    let strategy = delivered_activity();
    let mut found = BTreeSet::new();
    for _ in 0..64 {
        let (activity, files, schedule) = strategy.new_tree(&mut runner).expect("a case").current();
        found.extend(cases(&activity, &files, &schedule));
    }
    let listed: BTreeSet<&str> = [
        "a parent link",
        "a fork",
        "a sidechain",
        "two records with equal times",
        "times out of file order",
        "a split response",
        "a tool result in the next record",
        "an error result",
        "a copied prefix, exact",
        "a copied prefix, edited",
        "a copied prefix, cut short",
        "a chained resume",
        "non-ASCII text",
        "a file above the chunk size",
        "a cut inside a line",
        "a cut inside a multi-byte character",
        "a restart between steps",
        "a rebuild between steps",
    ]
    .into();
    assert_eq!(found, listed);
}
