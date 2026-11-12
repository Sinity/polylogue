use super::{Cause, Class, Derivation, Output, Version};
use crate::Result;
use polylogue_model::read::SessionLineage;
use rusqlite::{Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "lineage",
    class: Class::Fold,
    products: &[polylogue_model::derived::SESSION_LINEAGE],
    version: Version::Rules(1),
    after: &[],
    dirtied_by: &[Cause::LineageChanged],
    compute: |conn, pk, _| {
        Ok(Output::Lineage(
            crate::lineage::compute(conn, pk)?.expect("derived session exists"),
        ))
    },
};

pub(super) fn apply(tx: &Transaction<'_>, pk: i64, value: &SessionLineage) -> Result<()> {
    tx.prepare_cached("INSERT INTO session_lineage(session_pk,root,depth,root_pending,detail)
      VALUES(?1,?2,?3,?4,?5) ON CONFLICT(session_pk) DO UPDATE SET
      root=excluded.root,depth=excluded.depth,root_pending=excluded.root_pending,detail=excluded.detail")?
      .execute(params![pk,value.root.as_ref().map(ToString::to_string),value.depth,value.root_pending,serde_json::to_string(value)?])?;
    Ok(())
}
