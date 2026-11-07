//! Questions and their executions, shared by the writer and the confirmed stream.

use crate::{Reader, Result, StoreError, primary::Product, source::Source};
use polylogue_model::query_objects::{
    ExecutionBinding, QueryCustody, QueryEdge, QueryLookup, QueryObject, QueryRun, QueryRunOutcome,
    RunBinding,
};
use query_language::identity::{QuestionId, question_id_from_normal_form};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::BTreeMap;
pub(crate) mod evolution;

fn invalid(message: impl Into<String>) -> StoreError {
    StoreError::Invalid(message.into())
}

fn hash(
    normal_form: &serde_json::Value,
    parameters: &BTreeMap<String, String>,
) -> Result<QuestionId> {
    question_id_from_normal_form(normal_form, parameters)
        .map_err(|error| invalid(error.to_string()))
}

pub(crate) fn validate_object(object: &QueryObject) -> Result<()> {
    polylogue_query::plan::validate_object(object).map_err(|error| invalid(error.to_string()))?;
    validate_integrity(object)
}

pub(crate) fn validate_integrity(object: &QueryObject) -> Result<()> {
    if hash(&object.normal_form, &object.parameters)? != object.id {
        return Err(invalid("query hash does not identify its normal form"));
    }
    let mut definitions = BTreeMap::new();
    for definition in &object.definitions {
        if hash(&definition.normal_form, &definition.parameters)? != definition.question {
            return Err(invalid(
                "query definition hash does not identify its normal form",
            ));
        }
        if let Some(prior) = definitions.insert(definition.question.to_string(), definition)
            && prior != definition
        {
            return Err(invalid(
                "one query definition has conflicting frozen bodies",
            ));
        }
    }
    for dependency in &object.dependencies {
        if !definitions.contains_key(&dependency.to_string()) {
            return Err(invalid("a query dependency has no frozen definition"));
        }
    }
    for edge in &object.edges {
        match edge {
            QueryEdge::Saved { query } | QueryEdge::Combine { query } | QueryEdge::In { query }
                if !object.dependencies.contains(query) =>
            {
                return Err(invalid("a composition edge has no frozen query dependency"));
            }
            QueryEdge::Result { result } if result.is_empty() => {
                return Err(invalid(
                    "a query composition edge has an empty result identity",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Frozen result schemas must agree with the same canonical metadata readers bind.
pub(crate) fn validate_result_bindings(
    conn: &Connection,
    source: Source,
    object: &QueryObject,
) -> Result<()> {
    let outputs = object.source_outputs.iter().chain(
        object
            .definitions
            .iter()
            .flat_map(|definition| &definition.source_outputs),
    );
    let result_outputs: Vec<_> = outputs
        .filter(|output| {
            matches!(
                output.source,
                query_language::catalog::OutputSource::Result { .. }
            )
        })
        .collect();
    if result_outputs.is_empty() {
        return Ok(());
    }
    let kept = crate::selection::kept_results(conn, source)?;
    let bindings = polylogue_query::Bindings {
        kept,
        ..Default::default()
    };
    let actual = bindings.catalog();
    for output in result_outputs {
        let query_language::catalog::OutputSource::Result { reference, .. } = &output.source else {
            unreachable!()
        };
        let Some(canonical) = actual
            .source_outputs
            .iter()
            .find(|candidate| candidate.source == output.source)
        else {
            return Err(StoreError::NotFound(format!(
                "canonical result schema {reference}"
            )));
        };
        let query_language::catalog::OutputSource::Result { part, .. } = &output.source else {
            unreachable!()
        };
        let id = reference.strip_prefix("result:").unwrap_or(reference);
        let original = &bindings.kept[id];
        let proof = match part {
            query_language::ast::ResultPart::Rows => &original.output_metadata.rows,
            query_language::ast::ResultPart::Matches => &original.output_metadata.matches,
        };
        let unproven_fields = match proof {
            polylogue_model::kept::SchemaProof::Unproven => None,
            polylogue_model::kept::SchemaProof::Captured {
                unproven_fields, ..
            } => Some(unproven_fields),
        };
        if canonical.schema != output.schema || unproven_fields != output.unproven_fields.as_ref() {
            return Err(invalid(format!(
                "frozen result schema {reference} differs from its canonical column metadata"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_run(run: &QueryRun) -> Result<()> {
    run.validate_evolution().map_err(invalid)?;
    if run.machine.trim().is_empty() {
        return Err(invalid("a query run requires its accepting machine"));
    }
    if polylogue_model::ChangePosition::parse(&run.read_at.to_string()).as_ref()
        != Some(&run.read_at)
    {
        return Err(invalid("a query run has an invalid read position"));
    }
    match &run.binding {
        RunBinding::RetiredRead { .. } | RunBinding::Unavailable { .. } => {
            if run.question.is_none() {
                return Err(invalid(
                    "a retained unavailable invocation requires its original question",
                ));
            }
            Ok(())
        }
        RunBinding::Original { execution } => validate_execution(run, execution),
        RunBinding::OriginalObject { execution } => {
            if run.question.is_some() {
                return Err(invalid("an object read has no semantic query question"));
            }
            execution
                .validate()
                .map_err(|error| invalid(error.to_string()))
        }
    }
}
fn validate_execution(run: &QueryRun, execution: &ExecutionBinding) -> Result<()> {
    if run.historical_source.is_some()
        && execution
            .replay
            .context
            .versions
            .get("catalog")
            .map(String::as_str)
            != Some(polylogue_query::plan::catalog_fingerprint())
    {
        return Err(invalid(
            "migrated execution requires current native declarations",
        ));
    }
    if Some(execution.replay.context.question) != run.question {
        return Err(invalid(
            "a query run's replay context names another question",
        ));
    }
    for (key, expected) in [
        ("generation", run.read_at.generation.clone()),
        ("index", run.read_at.index.to_string()),
        ("archive", run.read_at.archive.to_string()),
    ] {
        if execution
            .replay
            .context
            .read_positions
            .get(key)
            .is_some_and(|actual| actual != &expected)
        {
            return Err(invalid(
                "a query run's replay context contradicts its read position",
            ));
        }
    }
    if let Some(scope) = execution.replay.context.versions.get("scope")
        && execution.replay.scope_fingerprint.as_ref() != Some(scope)
    {
        return Err(invalid(
            "a query run's replay context contradicts its grant scope",
        ));
    }
    execution
        .validate_presentation()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(())
}

pub(crate) fn equivalent(left: &QueryObject, right: &QueryObject) -> bool {
    // Alias labels and source spans do not create a second semantic question.
    left.id == right.id
        && left.normal_form == right.normal_form
        && left.parameters == right.parameters
}

pub(crate) fn known_object(conn: &Connection, object: &QueryObject) -> Result<bool> {
    let source = Source::of(conn)?;
    let bytes: Option<Vec<u8>> = conn
        .query_row(
            &source.sql("SELECT object FROM @source.query WHERE hash=?1"),
            [object.id.to_string()],
            |row| row.get(0),
        )
        .optional()?;
    let Some(bytes) = bytes else {
        return Ok(false);
    };
    let existing: QueryObject = crate::encoding::unpack_json(&bytes)?;
    if !equivalent(&existing, object) {
        return Err(StoreError::Conflict(format!("query:{}", object.id)));
    }
    Ok(true)
}

pub(crate) fn known_run(conn: &Connection, run: &QueryRun) -> Result<bool> {
    let source = Source::of(conn)?;
    let integrity: Option<(String, String)> = conn
        .query_row(
            &source.sql("SELECT binding_hash,run_hash FROM @source.query_run WHERE id=?1"),
            [run.id.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    if let Some((action, intent)) = integrity
        && (run.binding_hash()?.to_string() != action || run.record_hash()?.to_string() != intent)
    {
        return Err(StoreError::Conflict(format!("run:{}", run.id)));
    }
    let Some(existing) = lookup_run(conn, source, &run.id, None)? else {
        return Ok(false);
    };
    if existing.value != *run {
        return Err(StoreError::Conflict(format!("run:{}", run.id)));
    }
    Ok(true)
}

pub(crate) fn known_run_outcome(conn: &Connection, outcome: &QueryRunOutcome) -> Result<bool> {
    let source = Source::of(conn)?;
    let stored:Option<(Vec<u8>,String,String)>=conn.query_row(
        &source.sql("SELECT completion,binding_hash,run_hash FROM @source.query_run WHERE id=?1 AND completion IS NOT NULL"),
        [outcome.run.as_str()],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?)),
    ).optional()?;
    let Some((bytes, action, intent)) = stored else {
        return Ok(false);
    };
    let existing: QueryRunOutcome = crate::encoding::unpack_json(&bytes)?;
    check_outcome_integrity(&existing, &action, &intent)?;
    if existing != *outcome {
        return Err(StoreError::Conflict(format!("run outcome:{}", outcome.run)));
    }
    Ok(true)
}

fn check_outcome_integrity(value: &QueryRunOutcome, action: &str, intent: &str) -> Result<()> {
    if value.binding_hash.to_string() != action || value.run_hash.to_string() != intent {
        return Err(invalid(
            "retained run outcome failed its action or invocation integrity check",
        ));
    }
    Ok(())
}

fn pack(value: &impl serde::Serialize) -> Result<Vec<u8>> {
    Ok(crate::encoding::pack_bytes(&serde_json::to_vec(value)?)?)
}

fn first_run(conn: &Connection, source: Source, question: &str) -> Result<()> {
    conn.execute(
        &source.sql("UPDATE @source.query SET first_run=(SELECT id FROM @source.query_run WHERE query=?1 ORDER BY commit_seq,source_order,id LIMIT 1) WHERE hash=?1"),
        [question],
    )?;
    Ok(())
}

/// The caller's ordinary source transaction owns the complete question and edges.
pub(crate) fn apply_object(conn: &Connection, object: &QueryObject) -> Result<()> {
    validate_integrity(object)?;
    let source = Source::of(conn)?;
    let id = object.id.to_string();
    let existing: Option<Vec<u8>> = conn
        .query_row(
            &source.sql("SELECT object FROM @source.query WHERE hash=?1"),
            [&id],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(existing) = existing {
        let existing: QueryObject = crate::encoding::unpack_json(&existing)?;
        if !equivalent(&existing, object) {
            return Err(StoreError::Conflict(format!("query:{id}")));
        }
        return Ok(());
    }
    conn.execute(
        &source.sql("INSERT INTO @source.query(hash,normal_form,object) VALUES (?1,?2,?3)"),
        params![
            id,
            serde_json::to_string(&object.normal_form)?,
            pack(object)?
        ],
    )?;
    for edge in &object.edges {
        let (uses, how) = match edge {
            QueryEdge::Saved { query } => (format!("query:{query}"), "saved"),
            QueryEdge::Combine { query } => (format!("query:{query}"), "combine"),
            QueryEdge::In { query } => (format!("query:{query}"), "in"),
            QueryEdge::Result { result } => (format!("result:{result}"), "result"),
        };
        conn.execute(
            &source
                .sql("INSERT OR IGNORE INTO @source.query_edge(query,uses,how,edge) VALUES (?1,?2,?3,?4)"),
            params![id, uses, how, pack(edge)?],
        )?;
    }
    first_run(conn, source, &id)
}

struct StoredExecution {
    binding_hash: Option<String>,
    run_hash: Option<String>,
    body: Option<Vec<u8>>,
}

/// Source order breaks ties inside one confirmed position; timestamps never order runs.
pub(crate) fn apply_run(
    conn: &Connection,
    run: &QueryRun,
    commit_seq: i64,
    source_order: u64,
) -> Result<()> {
    validate_run(run)?;
    let source = Source::of(conn)?;
    let binding_hash = run.binding_hash()?.to_string();
    let run_hash = run.record_hash()?.to_string();
    let existing: Option<StoredExecution> = conn
        .query_row(
            &source.sql("SELECT binding_hash,run_hash,run FROM @source.query_run WHERE id=?1"),
            [run.id.as_str()],
            |row| {
                Ok(StoredExecution {
                    binding_hash: row.get(0)?,
                    run_hash: row.get(1)?,
                    body: row.get(2)?,
                })
            },
        )
        .optional()?;
    if let Some(StoredExecution {
        binding_hash: hash,
        run_hash: intent_hash,
        body,
    }) = existing
    {
        if hash.as_ref().is_some_and(|hash| hash != &binding_hash)
            || intent_hash.as_ref().is_some_and(|hash| hash != &run_hash)
        {
            return Err(StoreError::Conflict(format!("run:{}", run.id)));
        }
        if let Some(body) = body {
            let existing: QueryRun = crate::encoding::unpack_json(&body)?;
            return if existing == *run {
                Ok(())
            } else {
                Err(StoreError::Conflict(format!("run:{}", run.id)))
            };
        }
    }
    let order = i64::try_from(source_order)
        .map_err(|_| invalid("query run order exceeds SQLite's integer range"))?;
    let object_ref = match &run.binding {
        RunBinding::OriginalObject { execution } => Some(execution.reference.to_string()),
        _ => None,
    };
    conn.execute(
        &source.sql("INSERT INTO @source.query_run(id,binding_hash,run_hash,query,at,read_at,author,via,commit_seq,source_order,run,object_ref) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12) ON CONFLICT(id) DO UPDATE SET binding_hash=excluded.binding_hash,run_hash=excluded.run_hash,query=excluded.query,object_ref=excluded.object_ref,at=excluded.at,read_at=excluded.read_at,author=excluded.author,via=excluded.via,commit_seq=excluded.commit_seq,source_order=excluded.source_order,run=excluded.run"),
        params![run.id.as_str(),binding_hash,run_hash,run.question.map(|question| question.to_string()),run.at.unix_ns().to_string(),run.read_at.to_string(),serde_json::to_string(&run.authorship.author)?,run.authorship.via.as_str(),commit_seq,order,pack(run)?,object_ref],
    )?;
    if let Some(question) = run.question {
        first_run(conn, source, &question.to_string())?;
    }
    Ok(())
}

pub(crate) fn validate_run_outcome(outcome: &QueryRunOutcome) -> Result<()> {
    if outcome.exact && outcome.count.is_none() {
        return Err(invalid(
            "a run outcome requires its run and a known exact count",
        ));
    }
    if let polylogue_model::query_objects::RunOutcome::Error { kind } = &outcome.outcome
        && kind.is_empty()
    {
        return Err(invalid("a failed query run requires its error kind"));
    }
    Ok(())
}

pub(crate) fn apply_run_outcome(conn: &Connection, outcome: &QueryRunOutcome) -> Result<()> {
    validate_run_outcome(outcome)?;
    let source = Source::of(conn)?;
    let hash = outcome.binding_hash.to_string();
    let existing: Option<StoredExecution> = conn
        .query_row(
            &source
                .sql("SELECT binding_hash,run_hash,completion FROM @source.query_run WHERE id=?1"),
            [outcome.run.as_str()],
            |row| {
                Ok(StoredExecution {
                    binding_hash: row.get(0)?,
                    run_hash: row.get(1)?,
                    body: row.get(2)?,
                })
            },
        )
        .optional()?;
    if let Some(StoredExecution {
        binding_hash: existing_hash,
        run_hash: intent_hash,
        body,
    }) = existing
    {
        if existing_hash.as_ref().is_some_and(|known| known != &hash)
            || intent_hash
                .as_ref()
                .is_some_and(|known| known != &outcome.run_hash.to_string())
        {
            return Err(StoreError::Conflict(format!("run:{}", outcome.run)));
        }
        if let Some(body) = body {
            let existing: QueryRunOutcome = crate::encoding::unpack_json(&body)?;
            return if existing == *outcome {
                Ok(())
            } else {
                Err(StoreError::Conflict(format!("run outcome:{}", outcome.run)))
            };
        }
    }
    conn.execute(
        &source.sql("INSERT INTO @source.query_run(id,binding_hash,run_hash,outcome,count,exact,finished_at,completion) VALUES (?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(id) DO UPDATE SET binding_hash=coalesce(binding_hash,excluded.binding_hash),run_hash=coalesce(run_hash,excluded.run_hash),outcome=excluded.outcome,count=excluded.count,exact=excluded.exact,finished_at=excluded.finished_at,completion=excluded.completion"),
        params![outcome.run.as_str(),hash,outcome.run_hash.to_string(),serde_json::to_string(&outcome.outcome)?,outcome.count.map(|count|count.to_string()),outcome.exact,outcome.at.unix_ns().to_string(),pack(outcome)?],
    )?;
    Ok(())
}

pub(crate) fn clear(conn: &Connection) -> Result<()> {
    let source = Source::of(conn)?;
    for table in ["query_run", "query_edge", "query"] {
        conn.execute(&source.sql(&format!("DELETE FROM @source.{table}")), [])?;
    }
    Ok(())
}

fn prefix(prefix: &str) -> Result<String> {
    if !(12..=64).contains(&prefix.len()) || !prefix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(
            "a query hash requires 12 to 64 hexadecimal characters",
        ));
    }
    Ok(prefix.to_ascii_lowercase())
}

pub(crate) fn lookup_run(
    conn: &Connection,
    source: Source,
    id: &polylogue_model::RunId,
    pending: Option<&dyn crate::primary::Sinex>,
) -> Result<Option<QueryLookup<QueryRun>>> {
    let stored:Option<(Vec<u8>,String,String)>=conn.query_row(
        &source.sql("SELECT run,binding_hash,run_hash FROM @source.query_run WHERE id=?1 AND run IS NOT NULL"),
        [id.as_str()],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?)),
    ).optional()?;
    let mut found = stored
        .map(|(body, action, intent)| -> Result<QueryLookup<QueryRun>> {
            let value: QueryRun = crate::encoding::unpack_json(&body)?;
            if &value.id != id
                || value.binding_hash()?.to_string() != action
                || value.record_hash()?.to_string() != intent
            {
                return Err(invalid(
                    "retained run beginning failed its action or invocation integrity check",
                ));
            }
            Ok(QueryLookup {
                value,
                custody: QueryCustody::Canonical,
            })
        })
        .transpose()?;
    if let Some(pending) = pending {
        pending.visit_named_products(
            &crate::primary::NamedProductSelector::Exact(format!("query-run:{id}")),
            &mut |product| {
                if let Product::QueryRun { run } = product
                    && &run.id == id
                {
                    validate_run(run)?;
                    if let Some(existing) = &found {
                        if existing.value != **run {
                            return Err(StoreError::Conflict(format!("run:{id}")));
                        }
                    } else {
                        found = Some(QueryLookup {
                            value: *run.clone(),
                            custody: QueryCustody::Queued,
                        });
                    }
                }
                Ok(())
            },
        )?;
    }
    Ok(found)
}

/// The same canonical and durable queued custody lookup serves readers and
/// accepting original/scratch preparation. The caller retains its scope guard.
pub(crate) fn query_object_on(
    conn: &Connection,
    source: Source,
    key: &str,
    pending: Option<&dyn crate::primary::Sinex>,
) -> Result<Option<QueryLookup<QueryObject>>> {
    let key = prefix(key)?;
    let mut found = BTreeMap::<String, QueryLookup<QueryObject>>::new();
    let mut statement = conn.prepare(&source.sql(
        "SELECT hash,object FROM @source.query WHERE hash LIKE ?1 || '%' ORDER BY hash LIMIT 2",
    ))?;
    let bodies = statement.query_map([&key], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
    })?;
    for body in bodies {
        let (stored, body) = body?;
        let object: QueryObject = crate::encoding::unpack_json(&body)?;
        validate_integrity(&object)?;
        if object.id.to_string() != stored {
            return Err(invalid(
                "retained query body disagrees with its stored identity",
            ));
        }
        found.insert(
            object.id.to_string(),
            QueryLookup {
                value: object,
                custody: QueryCustody::Canonical,
            },
        );
    }
    if let Some(pending) = pending {
        let name = format!("query:{key}");
        let selector = if key.len() == 64 {
            crate::primary::NamedProductSelector::Exact(name)
        } else {
            crate::primary::NamedProductSelector::Prefix(name)
        };
        pending.visit_named_products(&selector, &mut |product| {
            if let Product::Query { object } = product
                && object.id.to_string().starts_with(&key)
            {
                validate_integrity(object)?;
                let id = object.id.to_string();
                if let Some(existing) = found.get(&id) {
                    if !equivalent(&existing.value, object) {
                        return Err(StoreError::Conflict(format!("query:{id}")));
                    }
                } else if found.len() < 2 {
                    found.insert(
                        id,
                        QueryLookup {
                            value: *object.clone(),
                            custody: QueryCustody::Queued,
                        },
                    );
                }
            }
            Ok(())
        })?;
    }
    if found.len() > 1 {
        return Err(invalid(format!("query hash prefix {key} is ambiguous")));
    }
    Ok(found.into_values().next())
}

impl Reader<'_> {
    /// A pending primary body is already durable, but is not a Sinex admission.
    pub fn query_object(
        &self,
        key: &str,
        pending: Option<&dyn crate::primary::Sinex>,
    ) -> Result<Option<QueryLookup<QueryObject>>> {
        let found = query_object_on(self.conn()?, Source::Cache, key, pending)?;
        self.conn()?;
        Ok(found)
    }

    pub fn query_run(
        &self,
        id: &polylogue_model::RunId,
        pending: Option<&dyn crate::primary::Sinex>,
    ) -> Result<Option<QueryLookup<QueryRun>>> {
        let found = lookup_run(self.conn()?, Source::Cache, id, pending)?;
        self.conn()?;
        Ok(found)
    }

    pub fn query_run_outcome(
        &self,
        id: &polylogue_model::RunId,
        pending: Option<&dyn crate::primary::Sinex>,
    ) -> Result<Option<QueryLookup<QueryRunOutcome>>> {
        let stored:Option<(Vec<u8>,String,String)>=self.conn()?.query_row(
            "SELECT completion,binding_hash,run_hash FROM source_query_run WHERE id=?1 AND completion IS NOT NULL",
            [id.as_str()],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?)),
        ).optional()?;
        let mut found = stored
            .map(
                |(body, action, intent)| -> Result<QueryLookup<QueryRunOutcome>> {
                    let value: QueryRunOutcome = crate::encoding::unpack_json(&body)?;
                    check_outcome_integrity(&value, &action, &intent)?;
                    Ok(QueryLookup {
                        value,
                        custody: QueryCustody::Canonical,
                    })
                },
            )
            .transpose()?;
        if let Some(pending) = pending {
            pending.visit_named_products(
                &crate::primary::NamedProductSelector::Exact(format!("query-run-outcome:{id}")),
                &mut |product| {
                    if let Product::QueryRunOutcome { outcome } = product
                        && &outcome.run == id
                    {
                        validate_run_outcome(outcome)?;
                        if let Some(existing) = &found {
                            if existing.value != **outcome {
                                return Err(StoreError::Conflict(format!("run outcome:{id}")));
                            }
                        } else {
                            found = Some(QueryLookup {
                                value: *outcome.clone(),
                                custody: QueryCustody::Queued,
                            });
                        }
                    }
                    Ok(())
                },
            )?;
        }
        self.conn()?;
        Ok(found)
    }

    pub fn query_uses(&self, question: QuestionId) -> Result<Vec<QueryEdge>> {
        let mut statement = self
            .conn()?
            .prepare("SELECT edge FROM source_query_edge WHERE query=?1 ORDER BY how,uses")?;
        let rows = statement.query_map([question.to_string()], |row| row.get::<_, Vec<u8>>(0))?;
        let mut edges = Vec::new();
        for row in rows {
            edges.push(crate::encoding::unpack_json(&row?)?);
        }
        self.conn()?;
        Ok(edges)
    }

    pub fn query_first_run(&self, question: QuestionId) -> Result<Option<String>> {
        Ok(self
            .conn()?
            .query_row(
                "SELECT first_run FROM source_query WHERE hash=?1",
                [question.to_string()],
                |row| row.get(0),
            )
            .optional()?
            .flatten())
    }
}

#[cfg(test)]
mod tests;
