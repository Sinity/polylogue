//! Immutable acquisition identity is independent of its observation pathname.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainedImport {
    pub id: String,
    pub format: String,
    /// Captured parser context, never a provider file to reopen.
    pub path: Option<String>,
    pub state: RetainedInput,
}

impl Reader<'_> {
    /// All retained immutable imports, even when several used the same path or
    /// were captured on another machine. Only metadata is decoded here.
    pub fn retained_imports(&self) -> Result<Vec<RetainedImport>> {
        let mut stmt=self.conn()?.prepare("SELECT i.id,i.format,
            (SELECT o.path FROM a.observation o WHERE o.input=i.id AND o.class<>'excised' ORDER BY o.rowid DESC LIMIT 1),
            r.chain,r.length,COALESCE(p.parsed_upto,0),p.parser_version,COALESCE(p.status='waiting',0)
            FROM a.input i JOIN a.revision r ON r.input=i.id
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
}
