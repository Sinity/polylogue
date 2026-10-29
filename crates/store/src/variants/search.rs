use super::*;

/// Derived text only. Lifecycle and normal derive transactions maintain it
/// alongside their source position, using the same canonical source rows.
pub const SCHEMA: &str = include_str!("../schema/variants-index.sql");

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct VariantMatch {
    pub variant: Variant,
    pub part: VariantPart,
}

pub fn project(conn: &Connection) -> Result<()> {
    conn.execute("DELETE FROM variant_text WHERE NOT EXISTS(SELECT 1 FROM source_variant v WHERE v.id=variant_text.variant)",[])?;
    conn.execute("DELETE FROM variant_indexed WHERE NOT EXISTS(SELECT 1 FROM source_variant v WHERE v.id=variant_indexed.variant)",[])?;
    let mut statement=conn.prepare("SELECT v.id,v.commit_seq FROM source_variant v LEFT JOIN variant_indexed i ON i.variant=v.id WHERE i.commit_seq IS NULL OR i.commit_seq!=v.commit_seq ORDER BY v.id")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let counter: i64 = row.get(1)?;
        conn.execute("DELETE FROM variant_text WHERE variant=?1", [&id])?;
        super::rows::parts(conn, &id, |part| {
            conn.execute(
                "INSERT INTO variant_text(variant,n,body) VALUES(?1,?2,?3)",
                params![
                    id,
                    i64::try_from(part.n).map_err(|_| StoreError::Invalid(
                        "variant part number is out of range".into()
                    ))?,
                    part.body
                ],
            )?;
            Ok(())
        })?;
        conn.execute("INSERT INTO variant_indexed(variant,commit_seq) VALUES(?1,?2) ON CONFLICT(variant) DO UPDATE SET commit_seq=excluded.commit_seq",params![id,counter])?;
    }
    Ok(())
}

impl Reader<'_> {
    /// Explicit variant search. Ordinary captured-content searches do not
    /// read this index. Language restricts the rendition's destination.
    pub fn visit_variant_search(
        &self,
        text: &str,
        language: Option<&str>,
        mut visit: impl FnMut(VariantMatch) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn()?;
        let mut statement=conn.prepare("SELECT t.variant,t.n FROM variant_fts JOIN variant_text t ON t.rowid=variant_fts.rowid JOIN source_variant v ON v.id=t.variant WHERE variant_fts MATCH ?1 AND (?2 IS NULL OR v.to_lang=?2) ORDER BY rank,t.variant,t.n")?;
        let mut rows = statement.query(params![text, language])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let n: i64 = row.get(1)?;
            let variant = super::rows::get(conn, &id)?.ok_or_else(|| {
                StoreError::Invalid("variant search refers to an absent rendition".into())
            })?;
            if !heads(conn, &variant.root)?.iter().any(|v| v.id == id) {
                continue;
            }
            super::rows::parts(conn, &id, |part| {
                if i64::try_from(part.n).ok() == Some(n) {
                    visit(VariantMatch {
                        variant: variant.clone(),
                        part,
                    })?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}
