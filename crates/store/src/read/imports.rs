//! Immutable acquisition identity is independent of its observation pathname.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainedImport {
    pub id: String,
    pub format: String,
    /// Captured parser context, never a provider file to reopen.
    pub path: Option<String>,
    /// This format/path pair was recorded by the acquiring importer.
    pub format_observed: bool,
    /// Earlier acquisitions lack a recorded interpretation and can supply legacy path context.
    pub legacy_context: bool,
    pub state: RetainedInput,
}

impl Reader<'_> {
    /// All retained immutable imports, even when several used the same path or
    /// were captured on another machine. Only metadata is decoded here.
    pub fn retained_imports(&self) -> Result<Vec<RetainedImport>> {
        let mut stmt=self.conn()?.prepare("WITH interpreted AS (
            SELECT o.input,o.path,json_extract(o.details_json,'$.import.format') AS format,
                row_number() OVER (PARTITION BY o.input ORDER BY (json_extract(o.details_json,'$.import.format')='export-file'),o.rowid DESC) AS choice
            FROM a.observation o WHERE o.class='imported' AND json_valid(o.details_json)
                AND json_type(o.details_json,'$.import.format')='text'
            ) SELECT i.id,COALESCE(selected.format,i.format),
            COALESCE(selected.path,(SELECT o.path FROM a.observation o WHERE o.input=i.id AND o.class<>'excised' ORDER BY o.rowid LIMIT 1)),
            r.chain,r.length,COALESCE(p.parsed_upto,0),p.parser_version,COALESCE(p.status='waiting',0),selected.input IS NOT NULL,
            EXISTS(SELECT 1 FROM a.observation legacy WHERE legacy.input=i.id AND legacy.class='imported'
                AND CASE WHEN json_valid(legacy.details_json) THEN json_type(legacy.details_json,'$.import.format') END IS NULL)
            FROM a.input i JOIN a.revision r ON r.input=i.id
            LEFT JOIN interpreted selected ON selected.input=i.id AND selected.choice=1
            LEFT JOIN input_ref ir ON ir.id=i.id LEFT JOIN parse_state p ON p.input_pk=ir.pk AND p.chain=r.chain
            WHERE r.number=(SELECT max(number) FROM a.revision latest WHERE latest.input=i.id)
              AND EXISTS(SELECT 1 FROM a.revision acquisition WHERE acquisition.input=i.id AND acquisition.produced='imported')
            ORDER BY i.id")?;
        Ok(stmt
            .query_map([], |r| {
                Ok(RetainedImport {
                    id: r.get(0)?,
                    format: r.get(1)?,
                    path: r.get(2)?,
                    format_observed: r.get(8)?,
                    legacy_context: r.get(9)?,
                    state: RetainedInput {
                        chain: r.get(3)?,
                        length: r.get::<_, i64>(4)? as u64,
                        parsed_upto: r.get::<_, i64>(5)? as u64,
                        parser_version: r.get::<_, Option<i64>>(6)?.map(|v| v as u32),
                        waiting: r.get(7)?,
                        route_rank: 2,
                    },
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    /// Captured legacy acquisition paths, without probing the current filesystem.
    pub fn retained_import_paths(&self, input: &str) -> Result<Vec<String>> {
        Ok(self.conn()?.prepare("SELECT path FROM a.observation WHERE input=?1 AND class='imported' AND CASE WHEN json_valid(details_json) THEN json_type(details_json,'$.import.format') END IS NULL GROUP BY path ORDER BY min(rowid)")?
            .query_map([input], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?)
    }
}
