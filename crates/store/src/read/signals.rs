use super::Reader;
use crate::row::Rows as _;
use crate::{Result, source::Source};
use polylogue_model::signal::{Checkpoint, StoredPendingDelivery};

impl Reader<'_> {
    pub fn signal_checkpoint(&self, evaluator: &str) -> Result<Option<Checkpoint>> {
        crate::signals::checkpoint(self.conn()?, Source::Cache, evaluator)
    }

    /// Older records stay readable but never trigger effects.
    pub fn visit_signals(
        &self,
        mut visit: impl FnMut(StoredPendingDelivery) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn()?;
        let ids: Vec<String> = conn.values(
            "SELECT id FROM source_delivery WHERE signal IS NOT NULL ORDER BY at,id",
            [],
        )?;
        for id in ids {
            visit(crate::authored_records::pending_delivery(conn, &id)?)?;
        }
        Ok(())
    }

    /// When each signal rule, by name, last fired: the evaluator's latest
    /// firing time in the stored deliveries.
    pub fn last_firings(&self) -> Result<std::collections::BTreeMap<String, i64>> {
        let conn = self.conn()?;
        let ids: Vec<String> = conn.values(
            "SELECT id FROM source_delivery WHERE signal IS NOT NULL",
            [],
        )?;
        let mut last = std::collections::BTreeMap::new();
        for id in ids {
            let Some(metadata) = crate::signals::metadata(conn, Source::Cache, &id)? else {
                continue;
            };
            let fired = metadata.delivery.fired;
            let at = last.entry(fired.rule_name).or_insert(fired.at);
            *at = (*at).max(fired.at);
        }
        Ok(last)
    }
}
