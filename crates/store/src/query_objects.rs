//! Questions and their executions, shared by the writer and the confirmed stream.

use crate::ConnExt as _;
use crate::{Reader, Result, StoreError, source::Source};
use polylogue_model::query_objects::{
    ExecutionBinding, QueryEdge, QueryObject, QueryRun, QueryRunOutcome, RunBinding,
};
use query_language::identity::{QuestionId, question_id_from_normal_form};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::BTreeMap;

fn question_id(
    normal_form: &serde_json::Value,
    parameters: &BTreeMap<String, String>,
) -> Result<QuestionId> {
    question_id_from_normal_form(normal_form, parameters)
        .map_err(|error| crate::StoreError::internal(error.to_string()))
}

pub(crate) fn validate_object(object: &QueryObject) -> Result<()> {
    polylogue_query::plan::validate_object(object)
        .map_err(|error| crate::StoreError::internal(error.to_string()))?;
    validate_integrity(object)
}

pub(crate) fn validate_integrity(object: &QueryObject) -> Result<()> {
    if question_id(&object.normal_form, &object.parameters)? != object.id {
        return Err(crate::StoreError::internal(
            "query hash does not identify its normal form",
        ));
    }
    let mut definitions = BTreeMap::new();
    for definition in &object.definitions {
        if question_id(&definition.normal_form, &definition.parameters)? != definition.query_id {
            return Err(crate::StoreError::internal(
                "query definition hash does not identify its normal form",
            ));
        }
        if let Some(prior) = definitions.insert(definition.query_id.to_string(), definition)
            && prior != definition
        {
            return Err(crate::StoreError::internal(
                "one query definition has conflicting saved bodies",
            ));
        }
    }
    for dependency in &object.dependencies {
        if !definitions.contains_key(&dependency.to_string()) {
            return Err(crate::StoreError::internal(
                "a query dependency has no saved definition",
            ));
        }
    }
    for edge in &object.edges {
        match edge {
            QueryEdge::Saved { query } | QueryEdge::Combine { query } | QueryEdge::In { query }
                if !object.dependencies.contains(query) =>
            {
                return Err(crate::StoreError::internal(
                    "a composition edge has no saved query dependency",
                ));
            }
            QueryEdge::Result { result } if result.is_empty() => {
                return Err(crate::StoreError::internal(
                    "a query composition edge has an empty result identity",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Saved result schemas must agree with the column metadata readers bind.
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
    // Only the kept results these outputs name are read.
    let ids: Vec<&str> = result_outputs
        .iter()
        .filter_map(|output| match &output.source {
            query_language::catalog::OutputSource::Result { reference, .. } => {
                Some(reference.strip_prefix("result:").unwrap_or(reference))
            }
            _ => None,
        })
        .collect();
    let kept = crate::selection::kept_results_named(conn, source, ids)?;
    let bindings = polylogue_query::Bindings {
        kept,
        ..Default::default()
    };
    let catalog = bindings.catalog();
    for output in result_outputs {
        let query_language::catalog::OutputSource::Result { reference, .. } = &output.source else {
            unreachable!()
        };
        let Some(bound) = catalog
            .source_outputs
            .iter()
            .find(|candidate| candidate.source == output.source)
        else {
            return Err(StoreError::NotFound(format!(
                "stored result schema {reference}"
            )));
        };
        if bound.schema != output.schema {
            return Err(crate::StoreError::internal(format!(
                "saved result schema {reference} differs from its bound column metadata"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_run(run: &QueryRun) -> Result<()> {
    if run.machine.trim().is_empty() {
        return Err(crate::StoreError::internal(
            "a query run requires its accepting machine",
        ));
    }
    if polylogue_model::ChangePosition::parse(&run.read_position.to_string()).as_ref()
        != Some(&run.read_position)
    {
        return Err(crate::StoreError::internal(
            "a query run has an invalid read position",
        ));
    }
    match &run.binding {
        RunBinding::Original { execution } => validate_execution(run, execution),
        RunBinding::OriginalObject { execution } => {
            if run.query_id.is_some() {
                return Err(crate::StoreError::internal(
                    "an object read has no query question",
                ));
            }
            execution
                .validate()
                .map_err(|error| crate::StoreError::internal(error.to_string()))
        }
    }
}
fn validate_execution(run: &QueryRun, execution: &ExecutionBinding) -> Result<()> {
    if Some(execution.replay.context.question) != run.query_id {
        return Err(crate::StoreError::internal(
            "a query run's replay context names another question",
        ));
    }
    for (key, expected) in [
        ("index_instance", run.read_position.index_instance.clone()),
        ("index", run.read_position.index.to_string()),
        ("archive", run.read_position.archive.to_string()),
    ] {
        if execution
            .replay
            .context
            .read_positions
            .get(key)
            .is_some_and(|recorded| recorded != &expected)
        {
            return Err(crate::StoreError::internal(
                "a query run's replay context contradicts its read position",
            ));
        }
    }
    if let Some(access) = execution.replay.context.versions.get("access")
        && execution.replay.access_fingerprint.as_ref() != Some(access)
    {
        return Err(crate::StoreError::internal(
            "a query run's replay context contradicts its read access",
        ));
    }
    execution
        .validate_presentation()
        .map_err(|error| crate::StoreError::internal(error.to_string()))?;
    Ok(())
}

pub(crate) fn equivalent(left: &QueryObject, right: &QueryObject) -> bool {
    // Alias labels and source spans do not create a second question.
    left.id == right.id
        && left.normal_form == right.normal_form
        && left.parameters == right.parameters
}

pub(crate) fn known_object(conn: &Connection, object: &QueryObject) -> Result<bool> {
    let source = Source::of(conn)?;
    let bytes: Option<Vec<u8>> = conn.scalar_optional(
        &source.sql("SELECT object FROM @source.query WHERE hash=?1"),
        [object.id.to_string()],
    )?;
    let Some(bytes) = bytes else {
        return Ok(false);
    };
    let existing: QueryObject = crate::encoding::unpack_json(&bytes)?;
    if !equivalent(&existing, object) {
        return Err(StoreError::Conflict(format!("query:{}", object.id)));
    }
    Ok(true)
}

/// The outcome stored for a run in `source`, checked against its recorded
/// action and invocation hashes.
fn stored_run_outcome(
    conn: &Connection,
    source: Source,
    id: &polylogue_model::RunId,
) -> Result<Option<QueryRunOutcome>> {
    let stored:Option<(Vec<u8>,String,String)>=conn.query_row(
        &source.sql("SELECT completion,binding_hash,run_hash FROM @source.query_run WHERE id=?1 AND completion IS NOT NULL"),
        [id.as_str()],|row| <(_, _, _)>::try_from(row),
    ).optional()?;
    stored
        .map(|(body, action, intent)| -> Result<QueryRunOutcome> {
            let value: QueryRunOutcome = crate::encoding::unpack_json(&body)?;
            check_outcome_integrity(&value, &action, &intent)?;
            Ok(value)
        })
        .transpose()
}

fn check_outcome_integrity(value: &QueryRunOutcome, action: &str, intent: &str) -> Result<()> {
    if value.binding_hash.to_string() != action || value.run_hash.to_string() != intent {
        return Err(crate::StoreError::stored_data(
            "stored run outcome failed its action or invocation integrity check",
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

/// The caller's source transaction holds the whole question and its edges.
pub(crate) fn apply_object(conn: &Connection, object: &QueryObject, commit_seq: i64) -> Result<()> {
    validate_integrity(object)?;
    if commit_seq <= 0 {
        return Err(StoreError::internal(
            "query commit position must be positive",
        ));
    }
    if known_object(conn, object)? {
        return Ok(());
    }
    let source = Source::of(conn)?;
    let id = object.id.to_string();
    conn.execute(
        &source.sql(
            "INSERT INTO @source.query(hash,normal_form,object,commit_seq) VALUES (?1,?2,?3,?4)",
        ),
        params![
            id,
            serde_json::to_string(&object.normal_form)?,
            pack(object)?,
            commit_seq,
        ],
    )?;
    for edge in &object.edges {
        let uses = match edge {
            QueryEdge::Saved { query } | QueryEdge::Combine { query } | QueryEdge::In { query } => {
                format!("query:{query}")
            }
            QueryEdge::Result { result } => format!("result:{result}"),
        };
        let kind = edge.kind().as_str();
        conn.execute(
            &source
                .sql("INSERT OR IGNORE INTO @source.query_edge(query,uses,kind,edge) VALUES (?1,?2,?3,?4)"),
            params![id, uses, kind, pack(edge)?],
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
    let order = i64::try_from(source_order).map_err(|_| {
        crate::StoreError::internal("query run order exceeds SQLite's integer range")
    })?;
    let object_ref = match &run.binding {
        RunBinding::OriginalObject { execution } => Some(execution.reference.to_string()),
        _ => None,
    };
    conn.execute(
        &source.sql("INSERT INTO @source.query_run(id,binding_hash,run_hash,query,at,read_position,author,via,commit_seq,source_order,run,object_ref) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12) ON CONFLICT(id) DO UPDATE SET binding_hash=excluded.binding_hash,run_hash=excluded.run_hash,query=excluded.query,object_ref=excluded.object_ref,at=excluded.at,read_position=excluded.read_position,author=excluded.author,via=excluded.via,commit_seq=excluded.commit_seq,source_order=excluded.source_order,run=excluded.run"),
        params![run.id.as_str(),binding_hash,run_hash,run.query_id.map(|query_id| query_id.to_string()),run.at.unix_ns().to_string(),run.read_position.to_string(),serde_json::to_string(&run.authorship.author)?,run.authorship.via.as_str(),commit_seq,order,pack(run)?,object_ref],
    )?;
    if let Some(question) = run.query_id {
        first_run(conn, source, &question.to_string())?;
    }
    Ok(())
}

pub(crate) fn validate_run_outcome(outcome: &QueryRunOutcome) -> Result<()> {
    if outcome.exact && outcome.count.is_none() {
        return Err(crate::StoreError::internal(
            "an exact run outcome requires a known count",
        ));
    }
    if let polylogue_model::query_objects::RunOutcome::Error { kind } = &outcome.outcome
        && kind.is_empty()
    {
        return Err(crate::StoreError::internal(
            "a failed query run requires its error kind",
        ));
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

fn prefix(prefix: &str) -> Result<String> {
    if !(8..=64).contains(&prefix.len()) || !prefix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(polylogue_model::Reason::BadQueryHash {
            prefix: prefix.into(),
        }
        .into());
    }
    Ok(prefix.to_ascii_lowercase())
}

pub(crate) fn lookup_run(
    conn: &Connection,
    source: Source,
    id: &polylogue_model::RunId,
) -> Result<Option<QueryRun>> {
    let stored:Option<(Vec<u8>,String,String)>=conn.query_row(
        &source.sql("SELECT run,binding_hash,run_hash FROM @source.query_run WHERE id=?1 AND run IS NOT NULL"),
        [id.as_str()],|row| <(_, _, _)>::try_from(row),
    ).optional()?;
    let found = stored
        .map(|(body, action, intent)| -> Result<QueryRun> {
            let value: QueryRun = crate::encoding::unpack_json(&body)?;
            if &value.id != id
                || value.binding_hash()?.to_string() != action
                || value.record_hash()?.to_string() != intent
            {
                return Err(crate::StoreError::stored_data(
                    "stored run beginning failed its action or invocation integrity check",
                ));
            }
            Ok(value)
        })
        .transpose()?;
    Ok(found)
}

/// One lookup over the stored query objects serves readers and
/// accepting original/scratch preparation. The caller keeps its scope guard.
pub(crate) fn query_object_on(
    conn: &Connection,
    source: Source,
    key: &str,
) -> Result<Option<QueryObject>> {
    let key = prefix(key)?;
    let mut found = BTreeMap::<String, QueryObject>::new();
    let mut statement = conn.prepare(&source.sql(
        "SELECT hash,object FROM @source.query WHERE hash LIKE ?1 || '%' ORDER BY hash LIMIT 2",
    ))?;
    let bodies = statement.query_map([&key], |row| <(String, Vec<u8>)>::try_from(row))?;
    for body in bodies {
        let (stored, body) = body?;
        let object: QueryObject = crate::encoding::unpack_json(&body)?;
        validate_integrity(&object)?;
        if object.id.to_string() != stored {
            return Err(crate::StoreError::stored_data(
                "saved query body disagrees with its stored identity",
            ));
        }
        found.insert(object.id.to_string(), object);
    }
    if found.len() > 1 {
        return Err(crate::StoreError::Ambiguous {
            reference: format!("query:{key}"),
            candidates: found.keys().map(|id| format!("query:{id}")).collect(),
        });
    }
    Ok(found.into_values().next())
}

impl Reader<'_> {
    pub fn query_object(&self, key: &str) -> Result<Option<QueryObject>> {
        let found = query_object_on(self.conn()?, Source::Cache, key)?;
        self.conn()?;
        Ok(found)
    }

    pub fn query_run(&self, id: &polylogue_model::RunId) -> Result<Option<QueryRun>> {
        let found = lookup_run(self.conn()?, Source::Cache, id)?;
        self.conn()?;
        Ok(found)
    }

    pub fn query_run_outcome(
        &self,
        id: &polylogue_model::RunId,
    ) -> Result<Option<QueryRunOutcome>> {
        let found = stored_run_outcome(self.conn()?, Source::Cache, id)?;
        self.conn()?;
        Ok(found)
    }
}

#[cfg(test)]
mod tests;
