//! Revision chains whose index lags their stored bytes: what the loop
//! parses again from the archive, for watched and imported inputs alike.
use super::*;

/// One revision chain the index has not parsed to its stored length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehindChain {
    pub input: String,
    /// The chain holds the input's latest revision.
    pub current: bool,
    /// For an imported input, the format and path it was imported as.
    pub imported: Option<ImportedAs>,
    pub progress: ChainProgress,
    pub reading_changed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedAs {
    pub format: String,
    /// The path the parser reads the bytes as, never a file to open.
    pub path: Option<String>,
}

impl Reader<'_> {
    /// Every chain of every input whose index is behind its stored bytes,
    /// by input and chain. A chain that ends in an incomplete record the
    /// parser is waiting on is not behind.
    pub fn chains_behind_index(&self, versions: &[(String, String)]) -> Result<Vec<BehindChain>> {
        let mut stmt = self.conn()?.prepare(
            "WITH inputs AS MATERIALIZED (
                SELECT input, max(number) AS number, max(produced = 'imported') AS imported
                FROM a.revision GROUP BY input
            ), chains AS (
                SELECT input, chain, max(length) AS length
                FROM a.revision GROUP BY input, chain
            ), first_paths AS MATERIALIZED (
                SELECT input, min(rowid) AS observation
                FROM a.observation GROUP BY input
            ), formats AS (
                SELECT o.input, o.path, json_extract(o.details_json, '$.import_format') AS format,
                    row_number() OVER (PARTITION BY o.input
                        ORDER BY (json_extract(o.details_json, '$.import_format') = 'export-file'),
                                 o.rowid DESC) AS choice
                FROM a.observation o WHERE o.class = 'imported' AND json_valid(o.details_json)
                    AND json_type(o.details_json, '$.import_format') = 'text'
            )
            SELECT c.input, c.chain, c.length, l.chain = c.chain,
                s.imported,
                COALESCE(f.format, i.format),
                COALESCE(f.path, o.path),
                COALESCE(p.parsed_upto, 0), p.input_pk IS NOT NULL, COALESCE(p.status = 'waiting' AND p.detail = CAST(c.length AS TEXT), 0), p.reading_version
            FROM chains c JOIN a.input i ON i.id = c.input
            JOIN inputs s ON s.input = c.input
            JOIN a.revision l ON l.input = c.input AND l.number = s.number
            LEFT JOIN formats f ON f.input = c.input AND f.choice = 1
            LEFT JOIN first_paths fp ON fp.input = c.input
            LEFT JOIN a.observation o ON o.rowid = fp.observation
            LEFT JOIN input_ref ir ON ir.id = c.input
            LEFT JOIN parse_state p ON p.input_pk = ir.pk AND p.chain = c.chain
            WHERE c.length > 0
              AND (p.input_pk IS NULL
                OR (NOT COALESCE(p.status = 'waiting' AND p.detail = CAST(c.length AS TEXT), 0) AND p.parsed_upto < c.length)
                OR EXISTS(SELECT 1 FROM json_each(?1) expected
                    WHERE json_extract(expected.value, '$[0]') = COALESCE(f.format, i.format)
                      AND json_extract(expected.value, '$[1]') IS NOT p.reading_version))
            ORDER BY 5,(SELECT max(o.seen_at) FROM a.observation o WHERE o.input=c.input) DESC,c.input,c.chain DESC",
        )?;
        let versions_json = serde_json::to_string(versions)?;
        Ok(stmt
            .query_map([versions_json], |r| {
                let imported: bool = r.get(4)?;
                Ok(BehindChain {
                    input: r.get(0)?,
                    current: r.get(3)?,
                    imported: imported
                        .then(|| -> rusqlite::Result<_> {
                            Ok(ImportedAs {
                                format: r.get(5)?,
                                path: r.get(6)?,
                            })
                        })
                        .transpose()?,
                    reading_changed: {
                        let format: String = r.get(5)?;
                        let previous: Option<String> = r.get(10)?;
                        versions
                            .iter()
                            .find(|(name, _)| name == &format)
                            .is_some_and(|(_, version)| previous.as_ref() != Some(version))
                    },
                    progress: ChainProgress {
                        chain: r.get(1)?,
                        length: r.get::<_, i64>(2)? as u64,
                        parsed_upto: r.get::<_, i64>(7)? as u64,
                        parsed: r.get(8)?,
                        waiting: r.get(9)?,
                        arrival_rank: if imported { 2 } else { 0 },
                    },
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }
}

#[cfg(test)]
mod tests;
