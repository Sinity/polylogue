//! Bounded projections from one authoritative read, consumed by the catalog publisher.
use super::catalog::Session;
use crate::source::Source;
use crate::{Reader, Result, StoreError};
use polylogue_model::ChangePosition;
use rusqlite::OptionalExtension;
use std::collections::{BTreeMap, BTreeSet};

pub enum SessionChange {
    Upsert(Box<Session>),
    Removed(String),
}
/// One source snapshot's authored changes, bounded by its existing source cursor.
pub struct VocabularyChanges {
    pub replaced: bool,
    pub definitions: Vec<DefinitionChange>,
    pub kept: Vec<NativeKept>,
    pub settings: Vec<String>,
    pub queries: BTreeSet<String>,
}

#[derive(Debug)]
pub struct DefinitionChange {
    pub family: String,
    pub names: BTreeMap<String, usize>,
    pub queries: BTreeMap<String, Vec<polylogue_query::SavedQuery>>,
    pub saved: Option<NativeSaved>,
    pub tag: Option<(String, Option<String>)>,
}
#[derive(Debug)]
pub struct NativeSaved {
    pub revision: String,
    pub name: String,
    pub text: String,
}
#[derive(Debug)]
pub struct NativeKept {
    pub output_metadata: Option<polylogue_model::kept::OutputMetadata>,
    pub shape: Option<polylogue_query::Shape>,
    pub native_identity: bool,
    pub id: String,
    pub name: Option<String>,
    pub unit: Option<String>,
    pub columns: Option<serde_json::Value>,
    pub match_columns: Option<serde_json::Value>,
    pub row_count: Option<u64>,
}
impl Reader<'_> {
    /// Streams session-grain projections. `None` is the single startup snapshot;
    /// subsequent calls use the indexed change feed, including removed sessions.
    pub fn visit_completion_sessions(
        &self,
        after: Option<&ChangePosition>,
        mut visit: impl FnMut(SessionChange) -> Result<()>,
    ) -> Result<ChangePosition> {
        let position = self.change_position()?;
        if let Some(after) = after
            && (after.generation != position.generation
                || after.index > position.index
                || after.archive > position.archive)
        {
            return Err(StoreError::Invalid("completion generation changed".into()));
        }
        let conn = self.conn()?;
        let columns = format!(
            "pk,id,native_id,coalesce({},''),origin,repo,last_activity_at,cwd,branch,harness",
            crate::session_sql::display_title("session")
        );
        // A session changes when its rows do or when its authored title does.
        let retitled = match after {
            Some(after) => crate::read::retitled_sessions(conn, after.archive)?,
            None => Default::default(),
        };
        let mut statement = conn.prepare(&if after.is_some() {
            format!(
                "SELECT {columns} FROM session WHERE changed>?1 OR id IN (SELECT value FROM json_each(?2)) ORDER BY changed,pk"
            )
        } else {
            format!("SELECT {columns} FROM session ORDER BY pk")
        })?;
        let mut rows = if let Some(after) = after {
            statement.query(rusqlite::params![
                after.index,
                serde_json::to_string(&retitled).map_err(|e| StoreError::Invalid(e.to_string()))?
            ])?
        } else {
            statement.query([])?
        };
        while let Some(row) = rows.next()? {
            self.check()?;
            let pk: i64 = row.get(0)?;
            let mut item = Session {
                id: row.get(1)?,
                native_id: row.get(2)?,
                prefix: String::new(),
                title: row.get(3)?,
                origin: row.get(4)?,
                repo: row.get(5)?,
                last_activity: row.get(6)?,
                first_prompt: None,
                values: BTreeMap::new(),
            };
            for (field, value) in [
                ("origin", Some(item.origin.clone())),
                ("repo", item.repo.clone()),
                ("cwd", row.get(7)?),
                ("branch", row.get(8)?),
                ("harness", row.get(9)?),
            ] {
                if let Some(value) = value.filter(|value| !value.is_empty()) {
                    let values = item.values.entry(field.into()).or_default();
                    if field == "cwd" {
                        for (at, _) in value.match_indices('/').skip(1) {
                            values.insert(value[..at].to_owned(), 1);
                        }
                    }
                    values.insert(value, 1);
                }
            }
            let mut counts=conn.prepare("SELECT 'model',model,count(*) FROM message WHERE session_pk=?1 AND model IS NOT NULL AND model<>'' GROUP BY model UNION ALL SELECT 'tool',tool,count(*) FROM tool_call WHERE session_pk=?1 GROUP BY tool")?;
            for row in counts.query_map([pk], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })? {
                let (field, value, count) = row?;
                item.values
                    .entry(field)
                    .or_default()
                    .insert(value, count as u64);
            }
            let prompt:Option<Vec<u8>>=conn.query_row("SELECT t.body FROM message m JOIN block b ON b.message_pk=m.pk JOIN text t ON t.id=b.text_id WHERE m.session_pk=?1 AND m.written_by='human' AND m.prompt_kind='prose' AND b.kind='text' ORDER BY m.order_input_id,m.seq,m.record_position,m.id,b.idx LIMIT 1",[pk],|row|row.get(0)).optional()?;
            item.first_prompt = prompt
                .map(|body| crate::encoding::unpack_text(&body))
                .transpose()?;
            visit(SessionChange::Upsert(Box::new(item)))?;
        }
        if let Some(after) = after {
            let mut removed = conn
                .prepare("SELECT r.session_id FROM removed_session r WHERE r.changed>?1 AND NOT EXISTS(SELECT 1 FROM session s WHERE s.id=r.session_id) ORDER BY r.changed,r.session_id")?;
            for id in removed.query_map([after.index], |row| row.get::<_, String>(0))? {
                visit(SessionChange::Removed(id?))?;
            }
        }
        Ok(position)
    }
    pub fn completion_vocabulary_changes(
        &self,
        after: Option<&ChangePosition>,
    ) -> Result<VocabularyChanges> {
        let replaced = self.completion_authored_replaced(after)?;
        let after = if replaced { None } else { after };
        let mut definitions = Vec::new();
        self.visit_completion_definitions(after.map(|position| position.archive), |change| {
            definitions.push(change);
            Ok(())
        })?;
        let mut kept = Vec::new();
        self.visit_completion_kept(after.map(|position| position.archive), |value| {
            kept.push(value);
            Ok(())
        })?;
        Ok(VocabularyChanges {
            replaced,
            definitions,
            kept,
            settings: self.completion_setting_keys(after.map(|position| position.archive))?,
            queries: after
                .map(|position| self.completion_changed_queries(position.archive))
                .transpose()?
                .unwrap_or_default(),
        })
    }

    /// Native authored lifecycle rebuilds replace that entire source category.
    /// Admission counters stay original, so they cannot describe its removals.
    pub fn completion_authored_replaced(&self, after: Option<&ChangePosition>) -> Result<bool> {
        let Some(after) = after else { return Ok(false) };
        let current = self.change_position()?;
        if after.generation != current.generation
            || after.archive > current.archive
            || after.index > current.index
        {
            return Err(StoreError::Invalid("completion generation changed".into()));
        }
        if matches!(Source::of(self.conn()?)?, Source::Ledger) {
            return Ok(false);
        }
        Ok(self.conn()?.query_row(
            "SELECT EXISTS(SELECT 1 FROM primary_fact WHERE kind='request' AND changed!=admitted AND changed>?1)",
            [after.archive], |row| row.get(0))?)
    }

    /// Changed objects and their exact retained dependents, not current names.
    /// Older NULL admissions belong to the startup snapshot, never to a guessed cut.
    pub fn completion_changed_queries(&self, after: i64) -> Result<BTreeSet<String>> {
        let lifecycle = if matches!(Source::of(self.conn()?)?, Source::Cache) {
            "UNION SELECT substr(reference,7) FROM primary_fact WHERE kind='query' AND changed!=admitted AND changed>?1 AND reference GLOB 'query:*'"
        } else {
            ""
        };
        let sql = format!(
            "WITH RECURSIVE affected(hash) AS (
            SELECT hash FROM source_query WHERE commit_seq>?1 {lifecycle}
            UNION SELECT e.query FROM source_query_edge e JOIN affected a ON e.uses='query:'||a.hash
        ) SELECT hash FROM affected"
        );
        let values = self
            .conn()?
            .prepare(&sql)?
            .query_map([after], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        self.check()?;
        Ok(values)
    }

    pub fn visit_completion_definitions(
        &self,
        after: Option<i64>,
        mut visit: impl FnMut(DefinitionChange) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn()?;
        let roots: BTreeSet<String> = match after {
            Some(after) => crate::assertions::changed_roots(conn, after)?,
            None => conn.prepare("SELECT DISTINCT root FROM source_assertion WHERE kind IN ('saved_query','tag') ORDER BY root")?
                .query_map([], |row| row.get(0))?.collect::<rusqlite::Result<_>>()?,
        };
        for root in roots {
            self.check()?;
            let relevant = conn.query_row("SELECT EXISTS(SELECT 1 FROM source_assertion WHERE root=?1 AND kind IN ('saved_query','tag'))", [&root], |row| row.get::<_,bool>(0))?;
            if relevant {
                visit(self.completion_definition(&root)?)?;
            }
        }
        self.check()?;
        Ok(())
    }
    /// The same current-family projection also resolves cache-only candidates
    /// after their final source row disappears.
    pub fn completion_definition(&self, family: &str) -> Result<DefinitionChange> {
        use polylogue_model::{
            Reference, RevisionReference,
            assertion::{AssertionKind, values},
        };
        let saved = self
            .selection_saved_queries_for_family(family)
            .map_err(crate::selection::SelectionError::into_store_error)?;
        let mut change = DefinitionChange {
            family: family.into(),
            names: saved.names,
            queries: saved.queries,
            saved: None,
            tag: None,
        };
        let Some(state) = self.assertion_family_currency(&RevisionReference {
            root: family.into(),
            revision: None,
        })?
        else {
            return Ok(change);
        };
        let [Reference::Assertion(exact)] = state.heads.as_slice() else {
            return Ok(change);
        };
        let Some(row) = self.assertion(exact)? else {
            return Ok(change);
        };
        let Some(name) = row.fact.key.as_ref() else {
            return Ok(change);
        };
        if let Some((name, definition)) = crate::assertions::saved_definition(&row.fact, &[])? {
            change.saved = Some(NativeSaved {
                revision: definition
                    .reference
                    .revision
                    .expect("exact saved definition"),
                name,
                text: definition.value.text,
            });
            return Ok(change);
        }
        if let (AssertionKind::Tag, values::Value::Tag(value)) = (
            row.fact.kind,
            row.fact
                .validate()
                .map_err(|error| StoreError::Invalid(error.to_string()))?,
        ) {
            let value = value
                .fields
                .value
                .map(|value| match value {
                    serde_json::Value::String(value) => Ok(value),
                    value => serde_json::to_string(&value),
                })
                .transpose()?;
            change.tag = Some((name.clone(), value));
        }
        Ok(change)
    }
    pub fn visit_completion_kept(
        &self,
        after: Option<i64>,
        mut visit: impl FnMut(NativeKept) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn()?;
        let fields = "k.id,k.name,CASE WHEN json_valid(k.rows_json) THEN CASE WHEN json_type(k.rows_json,'$.unit')='text' THEN json_extract(k.rows_json,'$.unit') END END,CASE WHEN json_valid(k.rows_json) THEN CASE WHEN json_type(k.rows_json,'$.columns')='array' THEN json_extract(k.rows_json,'$.columns') END END,CASE WHEN json_valid(k.rows_json) THEN CASE WHEN json_type(k.rows_json,'$.match_columns')='array' THEN json_extract(k.rows_json,'$.match_columns') END END,CASE WHEN json_valid(k.rows_json) THEN CASE WHEN json_type(k.rows_json,'$.rows')='array' THEN json_array_length(k.rows_json,'$.rows') END END,CASE WHEN json_valid(k.rows_json) THEN json_extract(k.rows_json,'$.shape') END,CASE WHEN json_valid(k.rows_json) THEN json_type(k.rows_json,'$.identities')='array' ELSE 0 END,CASE WHEN json_valid(k.rows_json) THEN json_extract(k.rows_json,'$.output_metadata') END";
        let sql = if after.is_some() {
            format!(
                "SELECT {fields} FROM source_kept_result k JOIN source_client_request r ON r.id=k.id WHERE r.commit_seq>?1 ORDER BY r.commit_seq,k.id"
            )
        } else {
            format!("SELECT {fields} FROM source_kept_result k ORDER BY k.id")
        };
        let mut statement = conn.prepare(&sql)?;
        let mut rows = if let Some(after) = after {
            statement.query([after])?
        } else {
            statement.query([])?
        };
        while let Some(row) = rows.next()? {
            self.check()?;
            let shape = row
                .get::<_, Option<String>>(6)?
                .map(|value| serde_json::from_value(serde_json::Value::String(value)))
                .transpose()?;
            let output_metadata = row.get::<_, Option<String>>(8)?;
            visit(NativeKept {
                // A body without a declaration declares nothing; the rows
                // then read with the types of their physical columns.
                output_metadata: match output_metadata {
                    None => Some(Default::default()),
                    Some(value) => serde_json::from_str(&value).ok(),
                },
                shape,
                native_identity: shape == Some(polylogue_query::Shape::Rows)
                    && row.get::<_, Option<bool>>(7)?.unwrap_or(false),
                id: row.get(0)?,
                name: row.get(1)?,
                unit: row.get(2)?,
                columns: row
                    .get::<_, Option<String>>(3)?
                    .map(|value| serde_json::from_str(&value))
                    .transpose()?,
                match_columns: row
                    .get::<_, Option<String>>(4)?
                    .map(|value| serde_json::from_str(&value))
                    .transpose()?,
                row_count: row.get::<_, Option<i64>>(5)?.map(|count| count as u64),
            })?;
        }
        Ok(())
    }
    /// Newly admitted assertions nominate exactly their affected setting families.
    /// The startup/category-replacement call takes the complete current snapshot.
    pub fn completion_setting_keys(&self, after: Option<i64>) -> Result<Vec<String>> {
        let conn = self.conn()?;
        let mut keys = BTreeSet::new();
        if let Some(after) = after {
            for root in crate::assertions::changed_roots(conn, after)? {
                keys.extend(conn.prepare("SELECT DISTINCT key FROM source_assertion WHERE root=?1 AND kind='setting' ORDER BY key")?
                    .query_map([root], |row| row.get::<_,String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?);
            }
        } else {
            keys.extend(
                conn.prepare(
                    "SELECT DISTINCT key FROM source_assertion WHERE kind='setting' ORDER BY key",
                )?
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?,
            );
        }
        self.check()?;
        Ok(keys.into_iter().collect())
    }
}

#[cfg(test)]
mod tests;
