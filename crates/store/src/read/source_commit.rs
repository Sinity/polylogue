//! Source admission is owned by the selected source store, independently of
//! the read snapshot and the message's display order.
use super::Reader;
use crate::Result;

impl Reader<'_> {
    pub fn source_commit(
        &self,
        input: &str,
        chain: i64,
        through: Option<u64>,
    ) -> Result<Option<u64>> {
        let Some(through) = through else {
            return Ok(None);
        };
        crate::source::Source::admitted_through(self.conn()?, input, chain, through)
    }
}
