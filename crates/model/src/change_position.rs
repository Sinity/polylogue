//! A change feed position, `GENERATION:INDEX:ARCHIVE`: the index generation
//! and the index and archive write counters read at that point.
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangePosition {
    pub generation: String,
    pub index: i64,
    pub archive: i64,
}

impl ChangePosition {
    pub fn parse(text: &str) -> Option<Self> {
        let mut parts = text.split(':');
        let generation = parts.next()?;
        let index = parts.next()?.parse::<i64>().ok()?;
        let archive = parts.next()?.parse::<i64>().ok()?;
        if generation.is_empty()
            || generation.chars().any(char::is_whitespace)
            || index < 0
            || archive < 0
            || parts.next().is_some()
        {
            return None;
        }
        Some(Self {
            generation: generation.into(),
            index,
            archive,
        })
    }
}

impl fmt::Display for ChangePosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.generation, self.index, self.archive)
    }
}

#[cfg(test)]
mod tests;
