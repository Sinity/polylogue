use super::*;
impl Reader<'_> {
    /// Original local ingress context, retained in the source observation.
    pub fn import_attribution(
        &self,
        input: &str,
        chain: i64,
    ) -> Result<polylogue_model::authored::ImportAttribution> {
        let conn = self.conn()?;
        let source = crate::source::Source::of(conn)?;
        let (machine,seen_at,details):(String,i64,Option<String>)=conn.query_row(&source.sql("SELECT o.machine,o.seen_at,o.details_json FROM @source.observation o LEFT JOIN @source.revision r ON r.input=o.input AND r.number=o.revision WHERE o.input=?1 AND coalesce(o.chain,r.chain)=?2 ORDER BY o.commit_seq DESC,o.rowid DESC LIMIT 1"),params![input,chain],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?;
        let details: polylogue_model::Json =
            serde_json::from_str(details.as_deref().ok_or_else(|| {
                StoreError::Invalid("bundle source has no recorded local ingress".into())
            })?)?;
        let member: crate::ImportMember = serde_json::from_value(details["import"].clone())?;
        let actor = member
            .authorship
            .ok_or_else(|| {
                StoreError::Invalid("bundle source has no computed local import actor".into())
            })?
            .author;
        Ok(polylogue_model::authored::ImportAttribution {
            machine,
            input: input.into(),
            chain,
            actor,
            at: Timestamp::from_unix_ms(seen_at).ok_or_else(|| {
                StoreError::Invalid("import source receipt time is outside supported range".into())
            })?,
        })
    }
}
