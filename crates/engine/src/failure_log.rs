//! What fails is written to the log once, when it starts, and once more when
//! it clears; not on every pass while it lasts (docs/operations.md, the
//! log). Status carries the conditions until a pass sees them clear.
use polylogue_model::{Cause, Code, Failure};
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

/// A failing condition: what failed, what ends it and its code. The same
/// subject failing for the same cause and code is one condition, whatever
/// counts or offsets its sentence carries.
type Condition = (String, Cause, Code);

#[derive(Default)]
pub(crate) struct FailureLog {
    /// Conditions logged and not yet seen to clear, with whether each was
    /// a whole turn's failure.
    failing: BTreeMap<Condition, (bool, Failure)>,
}

/// What a pass looked at, which decides what of the failing it can clear.
pub(crate) struct Looked<'a> {
    /// The files the pass planned to read: a file's failure that did not
    /// recur among them cleared. A file left alone (unchanged since it
    /// failed) clears nothing.
    pub files: &'a HashSet<PathBuf>,
    /// The pass walked every source: a failing step (a collector, the
    /// watch, coverage) that did not fail again cleared.
    pub walked: bool,
    /// The turn completed: a whole turn's failure cleared.
    pub completed: bool,
}

impl FailureLog {
    pub(crate) fn errors(&self) -> Vec<Failure> {
        self.failing
            .values()
            .map(|(_, failure)| failure.clone())
            .collect()
    }

    /// Logs the failures a pass ended with that are not already failing, and
    /// the clearing of failing conditions the pass looked at and did not
    /// meet again. A failure of one file or step is a warning; a whole
    /// turn's is an error.
    pub(crate) fn pass_ended(&mut self, failures: &[Failure], looked: Looked<'_>) {
        let now: BTreeMap<Condition, &Failure> = failures
            .iter()
            .map(|failure| {
                let (subject, cause, code) = failure.key();
                ((subject.to_owned(), cause, code), failure)
            })
            .collect();
        let files: HashSet<String> = looked
            .files
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        self.failing.retain(|condition, (turn, failure)| {
            if let Some(current) = now.get(condition) {
                *failure = (*current).clone();
                return true;
            }
            let cleared = if *turn {
                looked.completed
            } else if std::path::Path::new(&condition.0).is_absolute() {
                files.contains(&condition.0)
            } else {
                looked.walked
            };
            if cleared {
                let (subject, _, code) = condition;
                if *turn {
                    polylogue_store::log!(info, "passes complete again ({})", code.as_str());
                } else {
                    polylogue_store::log!(info, "{subject}: no longer failing ({})", code.as_str());
                }
            }
            !cleared
        });
        for (condition, failure) in now {
            if self.failing.contains_key(&condition) {
                continue;
            }
            let turn = failure.subject.is_empty();
            if turn {
                polylogue_store::log!(
                    error,
                    "a pass failed: {failure} ({}, {})",
                    failure.cause,
                    failure.code.as_str()
                );
            } else {
                polylogue_store::log!(
                    warning,
                    "{failure} ({}, {})",
                    failure.cause,
                    failure.code.as_str()
                );
            }
            self.failing.insert(condition, (turn, failure.clone()));
        }
    }
}

#[cfg(test)]
mod tests;
