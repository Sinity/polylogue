//! The session lineage row (root, depth, pending root and issue), computed by
//! [`crate::lineage`] and written to `session_lineage`.

use super::{Kind, Summary, Version, When};
use crate::Result;
use polylogue_model::read::SessionLineage;
use rusqlite::{Transaction, params};

pub(super) const SUMMARY: Summary = Summary {
    name: "lineage",
    products: &[polylogue_model::derived::SESSION_LINEAGE],
    version: Version::Rules(1),
    kind: Kind::Marks {
        compute: |conn, pk, out| {
            out.lineage = Some(
                crate::lineage::compute_lineage(conn, crate::row::SessionPk(pk))?
                    .ok_or_else(|| crate::StoreError::internal("a due session has no row"))?,
            );
            Ok(())
        },
        when: When::Lineage,
    },
};

pub(super) fn apply(tx: &Transaction<'_>, pk: i64, value: &SessionLineage) -> Result<()> {
    tx.prepare_cached("INSERT INTO session_lineage(session_pk,root,depth,root_pending,issue)
      VALUES(?1,?2,?3,?4,?5) ON CONFLICT(session_pk) DO UPDATE SET
      root=excluded.root,depth=excluded.depth,root_pending=excluded.root_pending,issue=excluded.issue")?
      .execute(params![pk,value.root.as_ref().map(ToString::to_string),value.depth,value.root_pending,value.issue.as_ref().map(serde_json::to_string).transpose()?])?;
    Ok(())
}
