//! Canonical questions and executions, shared by the writer and confirmed replay.
use crate::{Reader, Result, StoreError, primary::Product, source::Source};
use polylogue_model::query_objects::{QueryCustody, QueryEdge, QueryLookup, QueryObject, QueryRun};
use query_language::identity::{QuestionId, question_id_from_normal_form};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::BTreeMap;

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

pub(crate) fn validate_run(run: &QueryRun) -> Result<()> {
    if run.id.is_empty() || run.machine.trim().is_empty() {
        return Err(invalid(
            "a query run requires its identity and accepting machine",
        ));
    }
    if run.replay.context.question != run.question {
        return Err(invalid(
            "a query run's replay context names another question",
        ));
    }
    if run.exact && run.count.is_none() {
        return Err(invalid("an exact query run requires a known count"));
    }
    if let polylogue_model::query_objects::RunOutcome::Error { kind } = &run.outcome
        && kind.is_empty()
    {
        return Err(invalid("a failed query run requires its error kind"));
    }
    if polylogue_model::ChangePosition::parse(&run.read_at.to_string()).as_ref()
        != Some(&run.read_at)
    {
        return Err(invalid("a query run has an invalid read position"));
    }
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

/// Source order breaks ties inside one confirmed position; timestamps never order runs.
pub(crate) fn apply_run(
    conn: &Connection,
    run: &QueryRun,
    commit_seq: i64,
    source_order: u64,
) -> Result<()> {
    validate_run(run)?;
    let source = Source::of(conn)?;
    let existing: Option<Vec<u8>> = conn
        .query_row(
            &source.sql("SELECT run FROM @source.query_run WHERE id=?1"),
            [&run.id],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(existing) = existing {
        let existing: QueryRun = crate::encoding::unpack_json(&existing)?;
        if existing != *run {
            return Err(StoreError::Conflict(format!("run:{}", run.id)));
        }
        return Ok(());
    }
    let order = i64::try_from(source_order)
        .map_err(|_| invalid("query run order exceeds SQLite's integer range"))?;
    conn.execute(
        &source.sql("INSERT INTO @source.query_run(id,query,at,read_at,author,via,outcome,count,exact,commit_seq,source_order,run) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)"),
        params![run.id, run.question.to_string(), run.at.unix_ns().to_string(), run.read_at.to_string(),
            serde_json::to_string(&run.authorship.author)?, run.authorship.via.as_str(),
            serde_json::to_string(&run.outcome)?, run.count.map(|count| count.to_string()),
            run.exact, commit_seq, order, pack(run)?],
    )?;
    first_run(conn, source, &run.question.to_string())
}

/// A confirmed projection supplies this only when historical coverage proves it.
pub(crate) fn set_first_run(
    conn: &Connection,
    question: QuestionId,
    run: Option<&str>,
) -> Result<()> {
    let source = Source::of(conn)?;
    conn.execute(
        &source.sql("UPDATE @source.query SET first_run=?2 WHERE hash=?1"),
        params![question.to_string(), run],
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

impl Reader<'_> {
    /// A pending primary body is already durable, but is not a Sinex admission.
    pub fn query_object(
        &self,
        key: &str,
        pending: Option<&dyn crate::primary::Sinex>,
    ) -> Result<Option<QueryLookup<QueryObject>>> {
        let key = prefix(key)?;
        let conn = self.conn()?;
        let mut found = BTreeMap::<String, QueryLookup<QueryObject>>::new();
        let mut statement = conn.prepare(
            "SELECT object FROM source_query WHERE hash LIKE ?1 || '%' ORDER BY hash LIMIT 2",
        )?;
        let bodies = statement.query_map([&key], |row| row.get::<_, Vec<u8>>(0))?;
        for body in bodies {
            let object: QueryObject = crate::encoding::unpack_json(&body?)?;
            found.insert(
                object.id.to_string(),
                QueryLookup {
                    value: object,
                    custody: QueryCustody::Canonical,
                },
            );
        }
        if let Some(pending) = pending {
            pending.visit_pending_products(&mut |product| {
                if let Product::Query { object } = product
                    && object.id.to_string().starts_with(&key)
                {
                    validate_object(object)?;
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
        self.conn()?;
        Ok(found.into_values().next())
    }

    pub fn query_run(
        &self,
        id: &str,
        pending: Option<&dyn crate::primary::Sinex>,
    ) -> Result<Option<QueryLookup<QueryRun>>> {
        let conn = self.conn()?;
        let bytes: Option<Vec<u8>> = conn
            .query_row(
                "SELECT run FROM source_query_run WHERE id=?1",
                [id],
                |row| row.get(0),
            )
            .optional()?;
        let mut found = bytes
            .map(|bytes| {
                crate::encoding::unpack_json(&bytes).map(|run| QueryLookup {
                    value: run,
                    custody: QueryCustody::Canonical,
                })
            })
            .transpose()?;
        if let Some(pending) = pending {
            pending.visit_pending_products(&mut |product| {
                if let Product::QueryRun { run } = product
                    && run.id == id
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
            })?;
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
