use super::*;
use crate::archive::writer::Position;

/// Derived text only. Lifecycle and normal derive transactions maintain it
/// alongside their source position, using the same source rows.
pub const SCHEMA: &str = include_str!("../schema/alternates-index.sql");

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct AlternateMatch {
    pub alternate: Alternate,
    pub part: AlternatePart,
}

pub fn project(conn: &Connection) -> Result<()> {
    project_through(conn, None)
}

pub(crate) fn project_through(conn: &Connection, cut: Option<Position>) -> Result<()> {
    let through = cut.map(|position| position.0);
    conn.execute(
        "DELETE FROM alternate_text WHERE NOT EXISTS(
             SELECT 1 FROM source_alternate v WHERE v.id=alternate_text.alternate
             AND (?1 IS NULL OR v.commit_seq<=?1))",
        [through],
    )?;
    conn.execute(
        "DELETE FROM alternate_indexed WHERE NOT EXISTS(
             SELECT 1 FROM source_alternate v WHERE v.id=alternate_indexed.alternate
             AND (?1 IS NULL OR v.commit_seq<=?1))",
        [through],
    )?;
    let mut statement = conn.prepare(
        "SELECT v.id,v.commit_seq FROM source_alternate v
         LEFT JOIN alternate_indexed i ON i.alternate=v.id
         WHERE (i.commit_seq IS NULL OR i.commit_seq!=v.commit_seq)
         AND (?1 IS NULL OR v.commit_seq<=?1) ORDER BY v.id",
    )?;
    let mut rows = statement.query([through])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let counter: i64 = row.get(1)?;
        conn.execute("DELETE FROM alternate_text WHERE alternate=?1", [&id])?;
        super::rows::parts(conn, &id, |part| {
            conn.execute(
                "INSERT INTO alternate_text(alternate,position,body) VALUES(?1,?2,?3)",
                params![
                    id,
                    crate::to_i64(part.n, "alternate part number is out of range")?,
                    part.body
                ],
            )?;
            Ok(())
        })?;
        conn.execute("INSERT INTO alternate_indexed(alternate,commit_seq) VALUES(?1,?2) ON CONFLICT(alternate) DO UPDATE SET commit_seq=excluded.commit_seq",params![id,counter])?;
    }
    Ok(())
}

impl Reader<'_> {
    /// Explicit alternate search. Captured-content searches do not
    /// read this index. Language restricts the alternate's destination.
    pub fn visit_alternate_search(
        &self,
        text: &str,
        language: Option<&str>,
        mut visit: impl FnMut(AlternateMatch) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn()?;
        let mut statement=conn.prepare("SELECT t.alternate,t.position FROM alternate_fts JOIN alternate_text t ON t.rowid=alternate_fts.rowid JOIN source_alternate v ON v.id=t.alternate WHERE alternate_fts MATCH ?1 AND (?2 IS NULL OR v.to_lang=?2) ORDER BY rank,t.alternate,t.position")?;
        let mut rows = statement.query(params![text, language])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let n: i64 = row.get(1)?;
            let alternate = super::rows::get(conn, &id)?.ok_or_else(|| {
                StoreError::stored(0, "alternate search refers to an absent alternate")
            })?;
            if !heads(conn, &alternate.root)?.iter().any(|v| v.id == id) {
                continue;
            }
            let n = u64::try_from(n).map_err(|_| {
                crate::StoreError::stored(0, "negative indexed alternate part number")
            })?;
            let part = super::rows::get_part(conn, &id, n)?.ok_or_else(|| {
                StoreError::stored(1, "alternate search refers to an absent part")
            })?;
            visit(AlternateMatch { alternate, part })?;
        }
        Ok(())
    }
}
