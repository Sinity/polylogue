//! Physical import ingress is separate from an assertion's immutable authorship.
use super::*;
use polylogue_model::authored::ImportAttribution;

pub(crate) fn contains(conn: &Connection, id: &str, ingress: &ImportAttribution) -> Result<bool> {
    let source = Source::reading(conn)?;
    Ok(conn.prepare_cached(&source.sql("SELECT 1 FROM @source.assertion_import WHERE assertion=?1 AND input=?2 AND chain=?3 AND machine=?4 AND actor=?5"))?.exists(params![id,ingress.input,ingress.chain,ingress.machine,ingress.actor.to_string()])?)
}

pub(crate) fn record(
    conn: &Connection,
    id: &str,
    ingress: &ImportAttribution,
    counter: i64,
) -> Result<()> {
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT OR IGNORE INTO @source.assertion_import(assertion,input,chain,machine,actor,at,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7)"),params![id,ingress.input,ingress.chain,ingress.machine,ingress.actor.to_string(),ingress.at.to_string(),counter])?;
    Ok(())
}

impl crate::Reader<'_> {
    /// Locally recorded source memberships never supply an original authored time.
    pub fn assertion_imports(
        &self,
        reference: &RevisionReference,
    ) -> Result<Option<Vec<ImportAttribution>>> {
        let Some(assertion) = self.assertion(reference)? else {
            return Ok(None);
        };
        let conn = self.conn()?;
        let source = Source::reading(conn)?;
        let mut query = conn.prepare_cached(&source.sql("SELECT input,chain,machine,actor,at FROM @source.assertion_import WHERE assertion=?1 ORDER BY machine,input,chain,actor"))?;
        let mut rows = query.query([assertion.fact.id])?;
        let mut imports = Vec::new();
        while let Some(row) = rows.next()? {
            let at: String = row.get(4)?;
            imports.push(ImportAttribution {
                input: row.get(0)?,
                chain: row.get(1)?,
                machine: row.get(2)?,
                actor: row.id(3)?,
                at: Timestamp::parse_rfc3339(&at)
                    .ok_or_else(|| StoreError::stored(4, "import time is not RFC 3339"))?,
            });
        }
        self.check()?;
        Ok(Some(imports))
    }
}
