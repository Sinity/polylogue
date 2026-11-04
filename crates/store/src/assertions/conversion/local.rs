//! Numbered source conversion conserves rows inside the opening transaction.
use super::*;
use crate::source::Source;
use rusqlite::{OptionalExtension, Transaction, params};

fn schema(tx: &Transaction<'_>, source: Source, sql: &str) -> Result<()> {
    let sql = match source {
        Source::Ledger => sql.to_owned(),
        Source::Cache => crate::source::from_archive(sql)?.ddl,
    };
    tx.execute_batch(&sql)?;
    Ok(())
}

fn timestamp(millis: i64) -> Result<Timestamp> {
    Timestamp::from_unix_ms(millis)
        .ok_or_else(|| invalid("recorded predecessor time is out of range"))
}

fn pairs(tx: &Transaction<'_>, source: Source, id: &str) -> Result<Vec<(String, String)>> {
    Ok(tx
        .prepare_cached(&source.sql(
            "SELECT kind,other FROM @source.note_relation WHERE note_id=?1 ORDER BY kind,other",
        ))?
        .query_map([id], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?)
}

fn apply_effect(
    tx: &Transaction<'_>,
    source_schema: &SourceSchema,
    value: serde_json::Value,
    counter: i64,
) -> Result<()> {
    let effects = legacy::effects(source_schema, None, &serde_json::to_vec(&value)?)?;
    if effects.current != AuthoredEffect::default() {
        return Err(invalid(
            "standalone predecessor row produced another authored effect",
        ));
    }
    for fact in effects.historical_assertions {
        apply_projection(tx, &fact, counter)?;
    }
    Ok(())
}

/// SQL stored millisecond times and relation sets; a retained frozen effect can
/// preserve greater precision and original ordering. Verify that exact old
/// projection, while retaining the original row's actual local admission.
fn apply_projection(
    tx: &Transaction<'_>,
    fact: &polylogue_model::historical::HistoricalAssertionEvidence,
    counter: i64,
) -> Result<()> {
    if let Some(polylogue_model::historical::AssertionEvidence::Historical {
        fact: known, ..
    }) = crate::assertions::get_evidence(tx, &fact.id)?
    {
        let millis = |time: Option<Timestamp>| time.map(|time| time.unix_ms());
        let targets = |fact: &polylogue_model::historical::HistoricalAssertionEvidence| {
            fact.targets
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
        };
        let relations = |fact: &polylogue_model::historical::HistoricalAssertionEvidence| {
            fact.relations
                .iter()
                .map(serde_json::to_vec)
                .collect::<std::result::Result<std::collections::BTreeSet<_>, _>>()
        };
        let mut projected = fact.clone();
        projected.created_at = known.created_at;
        projected.valid_until = known.valid_until;
        projected.targets = known.targets.clone();
        projected.relations = known.relations.clone();
        if projected != known
            || millis(fact.created_at) != millis(known.created_at)
            || millis(fact.valid_until) != millis(known.valid_until)
            || targets(fact) != targets(&known)
            || relations(fact)? != relations(&known)?
        {
            return Err(StoreError::Conflict(format!(
                "assertion:{}@{}",
                fact.root, fact.id
            )));
        }
        if counter < 0 {
            return Err(invalid("negative predecessor assertion admission"));
        }
        let source = Source::of(tx)?;
        tx.execute(
            &source.sql("UPDATE @source.assertion SET commit_seq=?2 WHERE id=?1"),
            params![fact.id, counter],
        )?;
        return Ok(());
    }
    crate::assertions::apply_historical(tx, fact, counter)
}

fn notes(tx: &Transaction<'_>, source: Source, source_schema: &SourceSchema) -> Result<()> {
    let mut statement=tx.prepare_cached(&source.sql("SELECT id,first_note,kind,key,value,author,machine,created_at,valid_until,commit_seq FROM @source.note ORDER BY commit_seq,id"))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let root: String = row.get(1)?;
        let kind: String = row.get(2)?;
        let key: Option<String> = row.get(3)?;
        let value: Option<String> = row.get(4)?;
        let author: String = row.get(5)?;
        let machine: String = row.get(6)?;
        let created_at = timestamp(row.get(7)?)?;
        let valid_until = row.get::<_, Option<i64>>(8)?.map(timestamp).transpose()?;
        let counter: i64 = row.get(9)?;
        let relations = pairs(tx, source, &id)?;
        let mut effect = serde_json::to_value(AuthoredEffect::default())?;
        if kind == "withdrawal" {
            let targets: i64 = tx.query_row(
                &source.sql("SELECT count(*) FROM @source.note_target WHERE note_id=?1"),
                [&id],
                |row| row.get(0),
            )?;
            if targets != 0 {
                return Err(invalid(
                    "a predecessor withdrawal has additional unmapped targets",
                ));
            }

            let withdraws: Vec<_> = relations
                .iter()
                .filter(|(role, _)| role == "withdraws")
                .map(|(_, target)| target.clone())
                .collect();
            effect["withdrawals"] = serde_json::json!([{"id":id,"first_note":root,"machine":machine,"author":author,"created_at":created_at,"reason":value,"withdraws":withdraws}]);
            // Every stored relation is retained, including unexpected historical
            // roles; those cannot silently become a different frozen effect.
            if relations.iter().any(|(role, _)| role != "withdraws") {
                return Err(invalid(
                    "a predecessor withdrawal has additional unmapped relations",
                ));
            }
        } else {
            let targets: Vec<String> = tx
                .prepare_cached(
                    &source
                        .sql("SELECT ref FROM @source.note_target WHERE note_id=?1 ORDER BY ref"),
                )?
                .query_map([&id], |row| row.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            let revises: Vec<_> = relations
                .iter()
                .filter(|(role, _)| role == "revises")
                .map(|(_, target)| target.clone())
                .collect();
            let others: Vec<_> = relations
                .into_iter()
                .filter(|(role, _)| role != "revises")
                .collect();
            effect["notes"] = serde_json::json!([{"note":{"id":id,"first_note":root,"kind":kind,"key":key,"value":value,"author":author,"created_at":created_at,"valid_until":valid_until,"targets":targets,"relations":others},"machine":machine,"revises":revises}]);
        }
        apply_effect(tx, source_schema, effect, counter)?;
    }
    Ok(())
}

fn settings(tx: &Transaction<'_>, source: Source, source_schema: &SourceSchema) -> Result<()> {
    let mut statement=tx.prepare_cached(&source.sql("SELECT id,machine,key,value,author,set_at,commit_seq FROM @source.setting ORDER BY commit_seq,id"))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let machine: String = row.get(1)?;
        let key: String = row.get(2)?;
        let value: String = row.get(3)?;
        let author: String = row.get(4)?;
        let set_at = timestamp(row.get(5)?)?;
        let counter: i64 = row.get(6)?;
        let body: Option<Vec<u8>> = tx
            .query_row(
                &source.sql("SELECT request FROM @source.client_request WHERE id=?1"),
                [&id],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        let request = body
            .as_deref()
            .map(crate::encoding::unpack_json::<legacy::Request>)
            .transpose()?;
        let mut effect = serde_json::to_value(AuthoredEffect::default())?;
        effect["settings"] = serde_json::json!([{"id":id,"machine":machine,"key":key,"value":value,"author":author,"set_at":set_at}]);
        let effects = legacy::effects(
            source_schema,
            request.as_ref(),
            &serde_json::to_vec(&effect)?,
        )?;
        for fact in effects.historical_assertions {
            apply_projection(tx, &fact, counter)?;
        }
    }
    Ok(())
}

/// Called only by Source5 or its Index15 primary-source counterpart. Opening
/// owns the transaction, so a single unrepresentable row leaves the old schema,
/// accepted bodies, counters and schema version unchanged.
pub(crate) fn migrate_source(tx: &Transaction<'_>) -> Result<()> {
    let source = Source::of(tx)?;
    let source_schema = match source {
        Source::Ledger => SourceSchema::PolylogueArchive { version: 4 },
        Source::Cache => SourceSchema::LocalProjection {
            index_version: 14,
            source_schema_version: 4,
        },
    };
    migrate_query_runs(tx, source)?;
    schema(tx, source, crate::assertions::SCHEMA)?;
    tx.execute_batch(&source.sql("DROP INDEX @source.client_request_commit_seq; ALTER TABLE @source.client_request RENAME TO @source.client_request_before_assertions;"))?;
    schema(tx, source, crate::schema::CLIENT_REQUESTS)?;
    let mut statement=tx.prepare_cached(&source.sql("SELECT id,content_hash,created_at,request,outcome,effect,imports,commit_seq FROM @source.client_request_before_assertions ORDER BY commit_seq,id"))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let hash: Vec<u8> = row.get(1)?;
        let hash = BlobHash(
            hash.try_into()
                .map_err(|_| invalid("recorded accepted hash is not a BLAKE3 value"))?,
        );
        let at = row.get::<_, Option<i64>>(2)?.map(timestamp).transpose()?;
        let request: Option<Vec<u8>> = row.get(3)?;
        let outcome: Option<Vec<u8>> = row.get(4)?;
        let effect: Option<Vec<u8>> = row.get(5)?;
        let imports: String = row.get(6)?;
        let original_imports: polylogue_model::authored::ImportHistory =
            serde_json::from_str(&imports)?;
        let counter: i64 = row.get(7)?;
        let request_bytes = request
            .as_deref()
            .map(crate::encoding::unpack_bytes)
            .transpose()?;
        let outcome_bytes = outcome
            .as_deref()
            .map(crate::encoding::unpack_bytes)
            .transpose()?;
        let effect_bytes = effect
            .as_deref()
            .map(crate::encoding::unpack_bytes)
            .transpose()?;
        let mut accepted = match request_bytes.as_deref() {
            Some(request) => conserve_request(
                source_schema.clone(),
                request,
                outcome_bytes.as_deref(),
                effect_bytes.as_deref(),
                Some(hash),
                None,
            )?,
            None => conserve_without_request(
                source_schema.clone(),
                id.clone(),
                hash,
                at,
                outcome_bytes.as_deref(),
                effect_bytes.as_deref(),
            )?,
        };
        let body_time = match &accepted {
            ConservedAccepted::Compatible(prepared) => Some(prepared.request.at),
            ConservedAccepted::Historical(record) => record.at,
        };
        if matches!((body_time,at),(Some(body),Some(stored)) if body.unix_ms()!=stored.unix_ms()) {
            return Err(invalid(
                "the original request time disagrees with its stored source row",
            ));
        }
        if let ConservedAccepted::Historical(record) = &mut accepted
            && record.at.is_none()
        {
            record.at = at;
        }
        match &mut accepted {
            ConservedAccepted::Compatible(prepared) => {
                prepared
                    .original
                    .as_mut()
                    .expect("converted encoding")
                    .original
                    .imports = Some(original_imports)
            }
            ConservedAccepted::Historical(record) => {
                record
                    .original
                    .as_mut()
                    .expect("converted encoding")
                    .imports = Some(original_imports)
            }
        }
        if accepted.id() != id {
            return Err(StoreError::Conflict(id));
        }
        crate::primary::authored::conserve_local(tx, &accepted, &[], counter)?;
        tx.execute(
            &source.sql("UPDATE @source.client_request SET imports=?2,created_at=?3 WHERE id=?1"),
            params![id, imports, at.map(|time| time.unix_ms())],
        )?;
        if matches!(accepted, ConservedAccepted::Historical(_)) {
            tx.execute(
                &source.sql(
                    "UPDATE @source.client_request SET request=?2,outcome=?3,effect=?4 WHERE id=?1",
                ),
                params![id, request, outcome, effect],
            )?;
        }
    }
    drop(rows);
    drop(statement);
    notes(tx, source, &source_schema)?;
    settings(tx, source, &source_schema)?;
    tx.execute_batch(&source.sql("DROP TABLE @source.client_request_before_assertions; DROP TABLE @source.note_relation; DROP TABLE @source.note_target; DROP TABLE @source.note; DROP TABLE @source.setting;"))?;
    Ok(())
}

/// Existing run custody gains an explicit object alternative without changing
/// any original run or completion encoding or admitting arbitrary null queries.
pub(crate) fn migrate_query_runs(tx: &Transaction<'_>, source: Source) -> Result<()> {
    tx.execute_batch(&source.sql("DROP INDEX @source.query_run_query; ALTER TABLE @source.query_run RENAME TO @source.query_run_before_object;"))?;
    schema(tx, source, crate::schema::QUERY_RUNS)?;
    tx.execute_batch(&source.sql("INSERT INTO @source.query_run(id,binding_hash,run_hash,query,at,read_at,author,via,outcome,count,exact,finished_at,commit_seq,source_order,run,completion) SELECT id,binding_hash,run_hash,query,at,read_at,author,via,outcome,count,exact,finished_at,commit_seq,source_order,run,completion FROM @source.query_run_before_object; DROP TABLE @source.query_run_before_object;"))?;
    let rows = tx.prepare_cached(&source.sql("SELECT id,run,binding_hash,run_hash FROM @source.query_run WHERE run IS NOT NULL ORDER BY id"))?
        .query_map([], |row| Ok((row.get::<_,String>(0)?,row.get::<_,Vec<u8>>(1)?,row.get::<_,String>(2)?,row.get::<_,String>(3)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (id, body, binding, hash) in rows {
        let bytes = crate::encoding::unpack_bytes(&body)?;
        let schema = match source {
            Source::Ledger => SourceSchema::PolylogueArchive { version: 4 },
            Source::Cache => SourceSchema::LocalProjection {
                index_version: 14,
                source_schema_version: 4,
            },
        };
        let record = crate::query_objects::conserve_historical_run(
            schema,
            &bytes,
            Some(
                BlobHash::parse(&binding)
                    .map_err(|_| invalid("stored run binding hash is invalid"))?,
            ),
            Some(BlobHash::parse(&hash).map_err(|_| invalid("stored run hash is invalid"))?),
            Default::default(),
        )?;
        if record.id.as_str() != id {
            return Err(invalid("stored predecessor run identity disagrees"));
        }
        tx.execute(
            &source.sql("UPDATE @source.query_run SET historical=?2 WHERE id=?1"),
            params![id, crate::encoding::pack_json(&record)?],
        )?;
    }
    Ok(())
}
