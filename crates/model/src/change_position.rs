//! A change feed position, `INDEX_INSTANCE:INDEX:ARCHIVE`: the index instance
//! and the index and archive write counters read at that point.
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ChangePosition {
    pub index_instance: String,
    pub index: i64,
    pub archive: i64,
}

impl ChangePosition {
    pub fn parse(text: &str) -> Option<Self> {
        let mut parts = text.split(':');
        let index_instance = parts.next()?;
        let index = parts.next()?.parse::<i64>().ok()?;
        let archive = parts.next()?.parse::<i64>().ok()?;
        if index_instance.is_empty()
            || index_instance.chars().any(char::is_whitespace)
            || index < 0
            || archive < 0
            || parts.next().is_some()
        {
            return None;
        }
        Some(Self {
            index_instance: index_instance.into(),
            index,
            archive,
        })
    }
}

impl fmt::Display for ChangePosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.index_instance, self.index, self.archive)
    }
}

#[cfg(test)]
mod tests;
