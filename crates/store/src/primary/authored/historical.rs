//! Retired accepted actions are restored as evidence, never reaccepted as a
//! current request. Their source encoding and original outcome stay intact.
use super::*;
use polylogue_model::historical::HistoricalAcceptedRequest;

#[cfg(test)]
mod tests;

fn check(record: &HistoricalAcceptedRequest) -> Result<()> {
    record
        .validate()
        .map_err(|error| StoreError::Invalid(error.to_string()))?;
    crate::assertions::conversion::validate_historical_effects(record)
}

pub fn get(conn: &Connection, id: &str) -> Result<Option<HistoricalAcceptedRequest>> {
    let source = Source::reading(conn)?;
    let stored = conn.query_row(
        &source.sql("SELECT content_hash,historical,imports FROM @source.client_request WHERE id=?1 AND historical IS NOT NULL"),
        [id],
        |row| Ok((row.get::<_,Vec<u8>>(0)?, row.get::<_,Vec<u8>>(1)?, row.get::<_,String>(2)?)),
    ).optional()?;
    let Some((hash, body, imports)) = stored else {
        return Ok(None);
    };
    let mut record: HistoricalAcceptedRequest = crate::encoding::unpack_json(&body)?;
    check(&record)?;
    if record.id != id || record.content_hash.0.as_slice() != hash {
        return Err(StoreError::Invalid(
            "historical accepted identity disagrees with its row".into(),
        ));
    }
    record.imports = serde_json::from_str(&imports)?;
    Ok(Some(record))
}

/// Caller owns the transaction. Original metadata can be unknown; receiver
/// attribution is computed at ingress and cannot certify the original actor.
pub fn restore(
    tx: &Transaction<'_>,
    record: &HistoricalAcceptedRequest,
    claims: &[ImportAttribution],
    current: &ImportAttribution,
    counter: i64,
) -> Result<()> {
    check(record)?;
    if matches!(
        record.source,
        polylogue_model::historical::SourceSchema::RetainedPublication { .. }
    ) {
        return Err(StoreError::Invalid(
            "queued publication is not canonical accepted evidence".into(),
        ));
    }
    if counter < 0 {
        return Err(StoreError::Invalid(
            "negative historical request counter".into(),
        ));
    }
    let source = Source::of(tx)?;
    let mut incoming = record.clone();
    let observed = std::mem::take(&mut incoming.native.observed);
    incoming.native.declare(observed);
    let mut imports = ImportHistory::default();
    incoming.imports = ImportHistory::default();
    if let Some(mut known) = get(tx, &record.id)? {
        imports = std::mem::take(&mut known.imports);
        if !super::conserve::same_historical(&known, &incoming)
            || (known.source != incoming.source && incoming.native.is_empty())
        {
            return Err(StoreError::Conflict(record.id.clone()));
        }
        if let (Some(known), Some(incoming)) = (&mut known.original, &incoming.original) {
            super::conserve::merge_original(known, incoming);
        }
        known.native.declare(incoming.native.declared.clone());
        known.imports = ImportHistory::default();
        tx.execute(
            &source.sql("UPDATE @source.client_request SET historical=?2 WHERE id=?1"),
            params![record.id, pack(&known)?],
        )?;
    } else {
        let exists = tx
            .query_row(
                &source.sql("SELECT 1 FROM @source.client_request WHERE id=?1"),
                [&record.id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if exists {
            return Err(StoreError::Conflict(record.id.clone()));
        }
        if let Some(effects) = &record.conserved_effects {
            super::records::apply(tx, &effects.current, counter, true)?;
            for fact in &effects.historical_assertions {
                crate::assertions::apply_historical(tx, fact, counter)?;
            }
        }
        tx.execute(
            &source.sql("INSERT INTO @source.client_request(id,content_hash,created_at,historical,commit_seq) VALUES(?1,?2,?3,?4,?5)"),
            params![record.id,record.content_hash.0.as_slice(),record.at.map(|at|at.unix_ms()),pack(&incoming)?,counter],
        )?;
    }
    // Every carried observation becomes a declared claim at this receiver.
    let declared: Vec<_> = record
        .imports
        .observed
        .iter()
        .chain(&record.imports.declared)
        .chain(claims)
        .cloned()
        .collect();
    imports.merge(&declared, current);
    tx.execute(
        &source.sql("UPDATE @source.client_request SET imports=?2 WHERE id=?1"),
        params![record.id, serde_json::to_string(&imports)?],
    )?;
    Ok(())
}

pub fn visit(
    conn: &Connection,
    accept: &mut dyn FnMut(HistoricalAcceptedRequest) -> Result<()>,
) -> Result<()> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare_cached(&source.sql(
        "SELECT id FROM @source.client_request WHERE historical IS NOT NULL ORDER BY commit_seq,id",
    ))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        accept(get(conn, &id)?.ok_or(StoreError::NotFound(id))?)?;
    }
    Ok(())
}

impl crate::Reader<'_> {
    pub fn historical_accepted_request(
        &self,
        id: &str,
    ) -> Result<Option<HistoricalAcceptedRequest>> {
        let result = get(self.conn()?, id);
        self.check()?;
        result
    }
    pub fn visit_historical_accepted_requests(
        &self,
        accept: &mut dyn FnMut(HistoricalAcceptedRequest) -> Result<()>,
    ) -> Result<()> {
        let result = visit(self.conn()?, &mut |record| {
            self.check()?;
            accept(record)
        });
        self.check()?;
        result
    }
}

pub(crate) fn restore_record(
    tx: &Transaction<'_>,
    evidence: &polylogue_model::historical::HistoricalRecord,
    claims: &[ImportAttribution],
    current: &ImportAttribution,
    counter: i64,
) -> Result<()> {
    evidence
        .validate()
        .map_err(|error| StoreError::Invalid(error.to_string()))?;
    match evidence {
        polylogue_model::historical::HistoricalRecord::Accepted(record) => {
            restore(tx, record, claims, current, counter)
        }
        polylogue_model::historical::HistoricalRecord::Assertion(fact) => {
            crate::assertions::restore_historical(tx, fact, claims, current, counter)
        }
        polylogue_model::historical::HistoricalRecord::Run(run) => {
            crate::query_objects::restore_historical_run(tx, run, claims, current, counter)
        }
    }
}
