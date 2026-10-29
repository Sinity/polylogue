//! Read-only selection of exact input revisions for offline inspection.
use crate::{Reader, Result, StoreError};
use rusqlite::{OptionalExtension, params};

#[derive(Debug, Clone)]
pub struct InputRevision {
    pub input: String,
    pub format: String,
    pub revision: i64,
    pub chain: i64,
    pub length: u64,
    /// Used only to choose the origin's framing, never emitted in a schema.
    pub path: String,
}

impl Reader<'_> {
    /// `None` selects the latest revision of this exact input, never another input.
    pub fn schema_input(
        &self,
        input: &str,
        revision: Option<i64>,
    ) -> Result<Option<InputRevision>> {
        let conn = self.conn()?;
        let row = conn.query_row(
            "SELECT i.format,r.number,r.chain,r.length,
             coalesce((SELECT o.path FROM source_observation o WHERE o.input=i.id AND o.revision=r.number ORDER BY o.rowid DESC LIMIT 1),'')
             FROM source_input i JOIN source_revision r ON r.input=i.id
             WHERE i.id=?1 AND (?2 IS NULL OR r.number=?2) ORDER BY r.number DESC LIMIT 1",
            params![input, revision], |r| Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?,r.get::<_,String>(4)?))
        ).optional()?;
        row.map(|(format, revision, chain, length, path)| {
            Ok(InputRevision {
                input: input.into(),
                format,
                revision,
                chain,
                length: u64::try_from(length)
                    .map_err(|_| StoreError::Invalid("negative input length".into()))?,
                path,
            })
        })
        .transpose()
    }
}

/// One internal scan page. Pass `next` as `after` unchanged. Pages bound
/// transfers; callers continue until `next` is absent.
#[derive(Debug, Clone)]
pub struct SchemaInputsPage {
    pub inputs: Vec<InputRevision>,
    /// Inputs present in the origin whose requested revision is absent.
    pub missing_revisions: Vec<String>,
    pub next: Option<String>,
    /// Whole-selection count, independent of page. With `since_ms`, these
    /// inputs cannot establish whether they are inside its time bound.
    pub unknown_time_inputs: u64,
}
impl Reader<'_> {
    /// Select all inputs, or one origin, using receipt observation time.
    /// Missing times are excluded only when a time bound is requested.
    pub fn schema_inputs_page(
        &self,
        origin: Option<&str>,
        revision: Option<i64>,
        since_ms: Option<i64>,
        after: Option<&str>,
        page_size: usize,
    ) -> Result<SchemaInputsPage> {
        if page_size == 0 {
            return Err(StoreError::Invalid(
                "a schema input page must be positive".into(),
            ));
        }
        let conn = self.conn()?;
        let base = "WITH selected AS (
          SELECT i.id,i.format,r.number,r.chain,r.length,
            coalesce((SELECT o.path FROM source_observation o WHERE o.input=i.id AND o.revision=r.number ORDER BY o.commit_seq DESC,o.rowid DESC LIMIT 1),'') AS path,
            (SELECT o.seen_at FROM source_observation o WHERE o.input=i.id AND o.revision=r.number ORDER BY o.commit_seq DESC,o.rowid DESC LIMIT 1) AS observed_at
          FROM source_input i LEFT JOIN source_revision r ON r.input=i.id
            AND r.number=coalesce(?2,(SELECT max(newer.number) FROM source_revision newer WHERE newer.input=i.id))
          WHERE (?1 IS NULL OR i.format=?1))";
        let unknown: i64 = conn.query_row(
            &format!("{base} SELECT count(*) FROM selected WHERE observed_at IS NULL"),
            params![origin, revision],
            |row| row.get(0),
        )?;
        let limit = i64::try_from(page_size.saturating_add(1)).unwrap_or(i64::MAX);
        let mut statement = conn.prepare_cached(&format!(
            "{base}
          SELECT id,format,number,chain,length,path FROM selected
          WHERE (?3 IS NULL OR observed_at>=?3) AND (?4 IS NULL OR id>?4)
          ORDER BY id LIMIT ?5"
        ))?;
        let mut rows = statement.query(params![origin, revision, since_ms, after, limit])?;
        let mut inputs = Vec::new();
        let mut missing_revisions = Vec::new();
        let mut last = None;
        let mut count = 0;
        let mut next = None;
        while let Some(row) = rows.next()? {
            if count == page_size {
                next = last;
                break;
            }
            let input: String = row.get(0)?;
            last = Some(input.clone());
            count += 1;
            if let Some(number) = row.get::<_, Option<i64>>(2)? {
                let length: i64 = row.get(4)?;
                inputs.push(InputRevision {
                    input,
                    format: row.get(1)?,
                    revision: number,
                    chain: row.get(3)?,
                    length: u64::try_from(length)
                        .map_err(|_| StoreError::Invalid("negative input length".into()))?,
                    path: row.get(5)?,
                });
            } else {
                missing_revisions.push(input);
            }
        }
        Ok(SchemaInputsPage {
            inputs,
            missing_revisions,
            next,
            unknown_time_inputs: u64::try_from(unknown)
                .map_err(|_| StoreError::Invalid("negative unknown-time input count".into()))?,
        })
    }
}

#[cfg(test)]
mod tests;
