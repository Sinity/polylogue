//! Source-envelope checks shared by admission, recovery and publication.
use super::*;

pub(crate) fn key_sql(row: &str) -> String {
    format!(
        "json_array({row}.input,{row}.chain,{row}.source_start,{row}.source_end,lower(hex({row}.source_hash)),lower(hex({row}.blob)))"
    )
}

pub(crate) fn check_source(
    conn: &Connection,
    store: &hashed_blobs::Store,
    input: &str,
    chain: i64,
    source: &Extraction,
) -> Result<()> {
    if source.range.start >= source.range.end {
        return Err(StoreError::Invalid(
            "decoded bytes have no source envelope".into(),
        ));
    }
    let end = i64::try_from(source.range.end)
        .map_err(|_| StoreError::Invalid("decoded source exceeds archive address space".into()))?;
    let exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM revision WHERE input=?1 AND chain=?2 AND length>=?3)",
        rusqlite::params![input, chain, end],
        |r| r.get(0),
    )?;
    if !exists {
        return Err(StoreError::Invalid(
            "decoded bytes name no retained source range".into(),
        ));
    }
    let retained = crate::blobs::StoredChain::load_range(
        conn,
        "",
        input,
        chain,
        source.range.start,
        source.range.end,
    )?;
    if retained.range_hash(store, source.range.start, source.range.end)? != source.source_hash {
        return Err(StoreError::Conflict(
            "decoded bytes refer to a changed retained source envelope".into(),
        ));
    }
    Ok(())
}

pub(crate) type SourceKey = (String, i64, u64, u64, BlobHash);

pub(crate) fn check_once(
    conn: &Connection,
    store: &hashed_blobs::Store,
    input: &str,
    chain: i64,
    source: &Extraction,
    checked: &mut std::collections::BTreeSet<SourceKey>,
) -> Result<()> {
    let key = (
        input.to_owned(),
        chain,
        source.range.start,
        source.range.end,
        source.source_hash,
    );
    if !checked.contains(&key) {
        check_source(conn, store, input, chain, source)?;
        checked.insert(key);
    }
    Ok(())
}

pub(crate) struct Key {
    pub input: String,
    pub chain: i64,
    pub start: i64,
    pub end: i64,
    pub source: BlobHash,
    pub blob: BlobHash,
}
impl Key {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        let (input, chain, start, end, source, blob): (String, i64, i64, i64, String, String) =
            serde_json::from_str(value)?;
        if input.is_empty() || chain < 1 || start < 0 || end <= start {
            return Err(StoreError::Invalid(
                "invalid extracted source unit key".into(),
            ));
        }
        Ok(Self {
            input,
            chain,
            start,
            end,
            source: BlobHash::parse(&source).map_err(|e| StoreError::Invalid(e.to_string()))?,
            blob: BlobHash::parse(&blob).map_err(|e| StoreError::Invalid(e.to_string()))?,
        })
    }
    pub(crate) fn args(&self) -> [rusqlite::types::Value; 6] {
        use rusqlite::types::Value;
        [
            Value::Text(self.input.clone()),
            Value::Integer(self.chain),
            Value::Integer(self.start),
            Value::Integer(self.end),
            Value::Blob(self.source.0.to_vec()),
            Value::Blob(self.blob.0.to_vec()),
        ]
    }
    pub(crate) fn extraction(&self) -> Extraction {
        Extraction {
            range: self.start as u64..self.end as u64,
            source_hash: self.source,
            blob: self.blob,
        }
    }
}
pub(crate) const MATCH: &str =
    "input=?1 AND chain=?2 AND source_start=?3 AND source_end=?4 AND source_hash=?5 AND blob=?6";

pub(crate) fn candidates(
    conn: &Connection,
    input: Option<(&str, i64, u64, u64)>,
    blob: Option<&BlobHash>,
) -> Result<Vec<Key>> {
    let (id, chain, start, end) = input.unwrap_or(("", 0, 0, 0));
    let hash = blob.map(|h| h.0.as_slice());
    let sql = format!(
        "SELECT DISTINCT {} FROM extracted_blob x JOIN extent e ON e.input=x.input AND e.chain=x.chain AND e.offset<x.source_end AND e.offset+e.length>x.source_start WHERE (?1='' OR (x.input=?1 AND x.chain=?2 AND x.source_start<?4 AND x.source_end>?3)) AND (?5 IS NULL OR e.blob=?5)",
        key_sql("x")
    );
    let mut stmt = conn.prepare(&sql)?;
    stmt.query_map(
        rusqlite::params![id, chain, start as i64, end as i64, hash],
        |r| r.get::<_, String>(0),
    )?
    .map(|row| Key::parse(&row?))
    .collect()
}

pub(crate) fn invalidate(
    conn: &Connection,
    store: &hashed_blobs::Store,
    keys: Vec<Key>,
) -> Result<Vec<BlobHash>> {
    let mut retired = std::collections::BTreeSet::new();
    let mut checked = std::collections::BTreeMap::new();
    for key in keys {
        let source = (key.input.clone(), key.chain, key.start, key.end, key.source);
        let changed = match checked.get(&source) {
            Some(changed) => *changed,
            None => {
                let changed =
                    match check_source(conn, store, &key.input, key.chain, &key.extraction()) {
                        Ok(()) => false,
                        Err(StoreError::Conflict(_)) => true,
                        Err(error) => return Err(error),
                    };
                checked.insert(source, changed);
                changed
            }
        };
        match changed {
            false => {}
            true => {
                conn.execute(
                    &format!("DELETE FROM extracted_blob WHERE {MATCH}"),
                    rusqlite::params_from_iter(key.args().iter()),
                )?;
                retired.insert(key.blob);
            }
        }
    }
    Ok(retired.into_iter().collect())
}
