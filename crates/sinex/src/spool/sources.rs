//! A source's latest pending operations come from the producer spool's metadata.
use super::*;
use polylogue_store::primary::{
    SourceSelector,
    queue::{Lookup, Order},
};

#[derive(Default)]
pub(crate) struct LatestPending {
    pub latest: Option<Prepared>,
    pub raw: Option<Prepared>,
    pub unattributed: bool,
}

pub(super) fn get(spool: &Spool, source: &SourceSelector) -> Result<LatestPending> {
    let source = serde_json::to_vec(source)?;
    let latest = |raw| -> Result<Option<Prepared>> {
        spool
            .queue
            .lookup(
                Lookup::Source {
                    source: &source,
                    raw,
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
    Ok(LatestPending {
        latest: latest(false)?,
        raw: latest(true)?,
        unattributed: !spool
            .queue
            .lookup(Lookup::Unattributed, None, None, Order::OldestFirst, 1)?
            .is_empty(),
    })
}
