//! Receiver-observed ingress and sender claims stay separate from authored
//! facts. Neither is native publication proof or a privilege grant.
use super::*;
use polylogue_model::authored::{ImportAttribution, ImportHistory};

/// The caller owns the transaction. Current attribution is computed at the
/// receiving boundary; sender claims cannot populate observed ingress.
pub fn restore(
    conn: &Connection,
    fact: &AssertionFact,
    declared_claims: &[ImportAttribution],
    current: &ImportAttribution,
    counter: i64,
) -> Result<()> {
    super::apply(conn, std::slice::from_ref(fact), counter)?;
    let mut row =
        super::get(conn, &fact.id)?.ok_or_else(|| StoreError::NotFound(fact.id.clone()))?;
    row.imports.merge(declared_claims, current);
    save(conn, &fact.id, &row.imports)
}
pub(super) fn save(conn: &Connection, id: &str, history: &ImportHistory) -> Result<()> {
    let source = Source::of(conn)?;
    let history = serde_json::to_string(history)?;
    conn.execute(
        &source.sql("UPDATE @source.assertion SET imports=?2 WHERE id=?1 AND imports<>?2"),
        params![id, history],
    )?;
    Ok(())
}
impl crate::Reader<'_> {
    pub fn assertion_record(
        &self,
        reference: &RevisionReference,
    ) -> Result<Option<polylogue_model::assertion::AssertionRecord>> {
        let Some(id) = &reference.revision else {
            return Err(invalid(
                "assertion restoration record needs an exact revision",
            ));
        };
        let Some(row) = super::get(self.conn()?, id)?.filter(|row| row.fact.root == reference.root)
        else {
            return Ok(None);
        };
        Ok(Some(polylogue_model::assertion::AssertionRecord {
            reference: super::exact(&row.fact.root, &row.fact.id).to_string(),
            assertion: row.fact,
            imports: row
                .imports
                .observed
                .into_iter()
                .chain(row.imports.declared)
                .collect(),
        }))
    }
}
