//! Every framed record ends in exactly one disposition, and the counts must
//! add up to the records framed. The writer refuses a step whose counts do
//! not.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Disposition {
    /// The record became rows: messages, events, links, session facts.
    Materialized,
    /// Understood well enough to name, not turned into rows. The reason is
    /// shown with the count, so a reader knows what was set aside and why.
    Unknown { reason: String },
    /// Not a readable record: malformed JSON, a value of the wrong shape.
    /// The bytes stay stored.
    Refused { reason: String },
}

impl Disposition {
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self::Unknown {
            reason: reason.into(),
        }
    }

    pub fn refused(reason: impl Into<String>) -> Self {
        Self::Refused {
            reason: reason.into(),
        }
    }
}

/// Counts per record kind and disposition for one parse step.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accounting {
    pub framed: u64,
    pub materialized: BTreeMap<String, u64>,
    /// Keyed by `record kind: reason`.
    pub unknown: BTreeMap<String, u64>,
    pub refused: BTreeMap<String, u64>,
    /// Wire fields neither read nor excluded, by `record kind.path`. Not a
    /// failure: it shows where a parser falls behind its format.
    pub unread_fields: BTreeMap<String, u64>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("{framed} records framed but {accounted} accounted for")]
pub struct CountMismatch {
    pub framed: u64,
    pub accounted: u64,
}

impl Accounting {
    pub fn add(&mut self, kind: &str, disposition: &Disposition) {
        let bump = |m: &mut BTreeMap<_, u64>, k| *m.entry(k).or_default() += 1;
        match disposition {
            Disposition::Materialized => bump(&mut self.materialized, kind.to_owned()),
            Disposition::Unknown { reason } => bump(&mut self.unknown, format!("{kind}: {reason}")),
            Disposition::Refused { reason } => bump(&mut self.refused, format!("{kind}: {reason}")),
        }
    }

    pub fn unread(&mut self, kind: &str, path: &str) {
        *self
            .unread_fields
            .entry(format!("{kind}.{path}"))
            .or_default() += 1;
    }

    pub fn accounted(&self) -> u64 {
        self.materialized.values().sum::<u64>()
            + self.unknown.values().sum::<u64>()
            + self.refused.values().sum::<u64>()
    }

    pub fn check(&self) -> Result<(), CountMismatch> {
        let accounted = self.accounted();
        if accounted == self.framed {
            Ok(())
        } else {
            Err(CountMismatch {
                framed: self.framed,
                accounted,
            })
        }
    }

    pub fn merge(&mut self, other: &Accounting) {
        self.framed += other.framed;
        for (k, v) in &other.materialized {
            *self.materialized.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &other.unknown {
            *self.unknown.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &other.refused {
            *self.refused.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &other.unread_fields {
            *self.unread_fields.entry(k.clone()).or_default() += v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_must_add_up() {
        let mut a = Accounting {
            framed: 3,
            ..Accounting::default()
        };
        a.add("user", &Disposition::Materialized);
        a.add("progress", &Disposition::unknown("tick"));
        assert_eq!(
            a.check(),
            Err(CountMismatch {
                framed: 3,
                accounted: 2
            })
        );
        a.add("?", &Disposition::refused("malformed JSON"));
        assert_eq!(a.check(), Ok(()));
    }
}
