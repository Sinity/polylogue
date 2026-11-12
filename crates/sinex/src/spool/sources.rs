//! Pending source heads come from durable producer queue metadata.
use super::*;
use polylogue_store::primary::{
    SourceSelector,
    queue::{Lookup, Order},
};

#[derive(Default)]
pub(crate) struct Heads {
    pub head: Option<Prepared>,
    pub raw: Option<Prepared>,
    pub control: Option<Prepared>,
    pub unattributed: bool,
}

pub(super) fn get(spool: &Spool, source: &SourceSelector, version: &str) -> Result<Heads> {
    let source = serde_json::to_vec(source)?;
    let latest = |raw, reading_version| -> Result<Option<Prepared>> {
        spool
            .queue
            .lookup(
                Lookup::Source {
                    source: &source,
                    raw,
                    reading_version,
                },
                None,
                None,
                Order::NewestFirst,
                1,
            )?
            .into_iter()
            .next()
            .map(|entry| spool.prepared(entry))
            .transpose()
    };
    Ok(Heads {
        head: latest(false, None)?,
        raw: latest(true, None)?,
        control: latest(false, Some(version))?,
        unattributed: !spool
            .queue
            .lookup(Lookup::Unattributed, None, None, Order::OldestFirst, 1)?
            .is_empty(),
    })
}
