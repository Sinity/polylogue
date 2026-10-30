//! One-way predecessor run conservation shares the ordinary run row.
use super::*;
use polylogue_model::{
    historical::{NativeProjectionHistory, SourceSchema},
    query_objects::{HistoricalQueryRun, HistoricalRunReplay},
};

fn validate_compatible_run(run: &QueryRun) -> Result<()> {
    validate_run(run)?;
    if let Some(execution) = run.binding.original() {
        if matches!(
            execution.query_options.direct_reference.as_ref(),
            Some(polylogue_model::Reference::Request(_))
        ) {
            return Err(invalid(
                "the predecessor source did not define request-object reads",
            ));
        }
        let mut catalog = polylogue_query::catalog::catalog(&[], &[]);
        catalog.source_outputs = execution.replay.source_outputs.clone();
        let rebound = query_language::bind::bind(&execution.replay.bound_ast, &catalog)
            .map_err(|error| invalid(error.to_string()))?;
        let before = query_language::identity::normal_form(&execution.replay.bound_ast, &catalog)
            .map_err(|error| invalid(error.to_string()))?;
        let after = query_language::identity::normal_form(rebound.query(), &catalog)
            .map_err(|error| invalid(error.to_string()))?;
        if before != after {
            return Err(invalid(
                "the predecessor query changes meaning under the current catalog",
            ));
        }
    }
    Ok(())
}

pub fn conserve_historical_run(
    source: SourceSchema,
    encoded_run: &[u8],
    declared_binding_hash: Option<polylogue_model::BlobHash>,
    declared_run_hash: Option<polylogue_model::BlobHash>,
    native: NativeProjectionHistory,
) -> Result<HistoricalQueryRun> {
    let value: serde_json::Value = crate::encoding::read_json(encoded_run)?;
    let id = value
        .get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid("historical run lacks its recorded ID"))?
        .parse()
        .map_err(|_| invalid("historical run ID is invalid"))?;
    let question: QuestionId = serde_json::from_value(
        value
            .get("question")
            .cloned()
            .ok_or_else(|| invalid("historical run lacks its recorded question"))?,
    )?;
    let replay = match crate::encoding::read_json::<QueryRun>(encoded_run) {
        Ok(run) => match validate_compatible_run(&run) {
            Ok(())
                if serde_json::to_value(&run)? == value
                    && declared_binding_hash
                        .is_none_or(|hash| run.binding_hash().ok() == Some(hash))
                    && declared_run_hash
                        .is_none_or(|hash| run.record_hash().ok() == Some(hash)) =>
            {
                HistoricalRunReplay::Compatible { run: Box::new(run) }
            }
            Ok(()) => HistoricalRunReplay::Unavailable {
                reason:
                    "predecessor run encoding or declared integrity cannot be replayed unchanged"
                        .into(),
            },
            Err(error) => HistoricalRunReplay::Unavailable {
                reason: error.to_string(),
            },
        },
        Err(error) => HistoricalRunReplay::Unavailable {
            reason: error.to_string(),
        },
    };
    let record = HistoricalQueryRun {
        id,
        question,
        source,
        encoded_run: encoded_run.to_vec(),
        declared_binding_hash,
        declared_run_hash,
        native,
        imports: Default::default(),
        cached: vec![],
        records: vec![],
        replay,
    };
    record.validate().map_err(invalid)?;
    Ok(record)
}

pub(crate) fn merge_historical_run(
    known: &mut HistoricalQueryRun,
    incoming: &HistoricalQueryRun,
) -> Result<()> {
    known.validate().map_err(invalid)?;
    incoming.validate().map_err(invalid)?;
    let a: serde_json::Value = crate::encoding::read_json(&known.encoded_run)?;
    let b: serde_json::Value = crate::encoding::read_json(&incoming.encoded_run)?;
    if known.id != incoming.id
        || known.question != incoming.question
        || a != b
        || known
            .declared_binding_hash
            .zip(incoming.declared_binding_hash)
            .is_some_and(|(a, b)| a != b)
        || known
            .declared_run_hash
            .zip(incoming.declared_run_hash)
            .is_some_and(|(a, b)| a != b)
    {
        return Err(StoreError::Conflict(format!("run:{}", known.id)));
    }
    known.declared_binding_hash = known
        .declared_binding_hash
        .or(incoming.declared_binding_hash);
    known.declared_run_hash = known.declared_run_hash.or(incoming.declared_run_hash);
    for observation in &incoming.native.observed {
        known
            .native
            .observe(observation.clone())
            .map_err(|error| invalid(error.to_string()))?;
    }
    known.native.declare(incoming.native.declared.clone());
    for cached in &incoming.cached {
        if !known.cached.contains(cached) {
            known.cached.push(cached.clone());
        }
    }
    for record in &incoming.records {
        if !known.records.contains(record) {
            known.records.push(record.clone());
        }
    }
    for observed in &incoming.imports.observed {
        known.imports.merge(&incoming.imports.declared, observed);
    }
    for declared in &incoming.imports.declared {
        if !known.imports.declared.contains(declared) {
            known.imports.declared.push(declared.clone());
        }
    }
    if known.encoded_run != incoming.encoded_run && !known.records.contains(&incoming.encoded_run) {
        known.records.push(incoming.encoded_run.clone());
    }
    Ok(())
}

pub(crate) fn apply_historical_run(
    conn: &Connection,
    record: &HistoricalQueryRun,
    counter: i64,
    source_order: Option<u64>,
) -> Result<()> {
    record.validate().map_err(invalid)?;
    if let HistoricalRunReplay::Compatible { run } = &record.replay {
        validate_compatible_run(run)?;
    }
    let source = Source::of(conn)?;
    let stored: Option<Vec<u8>> = conn
        .query_row(
            &source.sql(
                "SELECT historical FROM @source.query_run WHERE id=?1 AND historical IS NOT NULL",
            ),
            [record.id.as_str()],
            |row| row.get(0),
        )
        .optional()?;
    let old_body: Option<Vec<u8>> = conn
        .query_row(
            &source.sql("SELECT run FROM @source.query_run WHERE id=?1 AND run IS NOT NULL"),
            [record.id.as_str()],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(body) = old_body {
        let known: serde_json::Value = crate::encoding::unpack_json(&body)?;
        let incoming: serde_json::Value = crate::encoding::read_json(&record.encoded_run)?;
        if known != incoming {
            return Err(StoreError::Conflict(format!("run:{}", record.id)));
        }
    }
    let mut record = record.clone();
    if let Some(stored) = stored {
        let mut known: HistoricalQueryRun = crate::encoding::unpack_json(&stored)?;
        merge_historical_run(&mut known, &record)?;
        record = known;
    }

    if let HistoricalRunReplay::Compatible { run } = &record.replay {
        apply_run(
            conn,
            run,
            counter,
            source_order.ok_or_else(|| {
                invalid("compatible historical projection requires its actual source order")
            })?,
        )?;
    } else {
        let order = source_order
            .map(i64::try_from)
            .transpose()
            .map_err(|_| invalid("historical run source order exceeds SQLite integer range"))?;
        let hashes: Option<(Option<String>, Option<String>)> = conn
            .query_row(
                &source.sql("SELECT binding_hash,run_hash FROM @source.query_run WHERE id=?1"),
                [record.id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((binding, run)) = hashes
            && (binding
                .as_ref()
                .zip(record.declared_binding_hash)
                .is_some_and(|(known, claim)| known != &claim.to_string())
                || run
                    .as_ref()
                    .zip(record.declared_run_hash)
                    .is_some_and(|(known, claim)| known != &claim.to_string()))
        {
            return Err(StoreError::Conflict(format!("run:{}", record.id)));
        }
        conn.execute(&source.sql("INSERT INTO @source.query_run(id,binding_hash,run_hash,query,commit_seq,source_order,historical) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(id) DO UPDATE SET historical=excluded.historical"),
            params![record.id.as_str(),record.declared_binding_hash.map(|hash|hash.to_string()),record.declared_run_hash.map(|hash|hash.to_string()),record.question.to_string(),counter,order,pack(&record)?])?;
    }
    conn.execute(
        &source.sql("UPDATE @source.query_run SET historical=?2 WHERE id=?1"),
        params![record.id.as_str(), pack(&record)?],
    )?;
    Ok(())
}

pub(crate) fn historical_run_on(
    conn: &Connection,
    id: &polylogue_model::RunId,
) -> Result<Option<(HistoricalQueryRun, u64)>> {
    let source = Source::reading(conn)?;
    let stored = conn.query_row(&source.sql("SELECT historical,commit_seq FROM @source.query_run WHERE id=?1 AND historical IS NOT NULL"), [id.as_str()], |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, i64>(1)?))).optional()?;
    stored
        .map(|(body, commit)| {
            let record: HistoricalQueryRun = crate::encoding::unpack_json(&body)?;
            record.validate().map_err(invalid)?;
            if &record.id != id {
                return Err(invalid("historical run row identity disagrees"));
            }
            let commit =
                u64::try_from(commit).map_err(|_| invalid("negative historical run admission"))?;
            Ok((record, commit))
        })
        .transpose()
}

/// Imported metadata is a declared old execution, never a new Begin or a
/// renewed grant to replay it. Existing locally retained authority survives.
pub(crate) fn prepare_historical_run_restore(
    conn: &Connection,
    record: &HistoricalQueryRun,
    claims: &[polylogue_model::authored::ImportAttribution],
    current: &polylogue_model::authored::ImportAttribution,
) -> Result<HistoricalQueryRun> {
    record.validate().map_err(invalid)?;
    let mut incoming = record.clone();
    let observed = std::mem::take(&mut incoming.native.observed);
    incoming.native.declare(observed);
    let mut declared = claims.to_vec();
    declared.extend(
        record
            .imports
            .observed
            .iter()
            .chain(&record.imports.declared)
            .cloned(),
    );
    incoming.imports = Default::default();
    if matches!(incoming.replay, HistoricalRunReplay::Compatible { .. }) {
        incoming.replay = HistoricalRunReplay::Unavailable {
            reason:
                "imported historical execution metadata does not renew original replay authority"
                    .into(),
        };
    }
    if let Some((mut known, _)) = historical_run_on(conn, &record.id)? {
        merge_historical_run(&mut known, &incoming)?;
        incoming = known;
    }
    incoming.imports.merge(&declared, current);
    Ok(incoming)
}

pub(crate) fn restore_historical_run(
    conn: &Connection,
    record: &HistoricalQueryRun,
    claims: &[polylogue_model::authored::ImportAttribution],
    current: &polylogue_model::authored::ImportAttribution,
    counter: i64,
) -> Result<()> {
    let restored = prepare_historical_run_restore(conn, record, claims, current)?;
    let source = Source::of(conn)?;
    let order: Option<i64> = conn
        .query_row(
            &source.sql("SELECT source_order FROM @source.query_run WHERE id=?1"),
            [record.id.as_str()],
            |row| row.get(0),
        )
        .optional()?
        .flatten();
    let order = order
        .map(u64::try_from)
        .transpose()
        .map_err(|_| invalid("negative historical run source order"))?;
    apply_historical_run(conn, &restored, counter, order)
}

impl Reader<'_> {
    pub fn historical_query_run(
        &self,
        id: &polylogue_model::RunId,
    ) -> Result<Option<HistoricalQueryRun>> {
        let result =
            historical_run_on(self.conn()?, id).map(|stored| stored.map(|(record, _)| record));
        self.check()?;
        result
    }
}
