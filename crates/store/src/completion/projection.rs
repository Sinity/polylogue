//! Bounded projections from one authoritative read, consumed by the catalog publisher.
use super::catalog::Session;
use crate::{Reader, Result, StoreError};
use polylogue_model::ChangePosition;
use rusqlite::OptionalExtension;
use std::collections::BTreeMap;

pub enum SessionChange {
    Upsert(Box<Session>),
    Removed(String),
}
#[derive(Debug)]
pub struct DefinitionChange {
    pub family: String,
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
        let mut statement=conn.prepare(if after.is_some() {
            "SELECT pk,id,native_id,coalesce(title,derived_title,''),origin,repo,last_activity_at,cwd,branch,harness FROM session WHERE changed>?1 ORDER BY changed,pk"
        } else {
            "SELECT pk,id,native_id,coalesce(title,derived_title,''),origin,repo,last_activity_at,cwd,branch,harness FROM session ORDER BY pk"
        })?;
        let mut rows = if let Some(after) = after {
            statement.query([after.index])?
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
    /// Family candidates are independent of first admission counters: a
    /// withdrawal may affect another family, or retract the whole last fact.
    pub fn visit_completion_definitions(
        &self,
        _after: Option<i64>,
        mut visit: impl FnMut(DefinitionChange) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn()?;
        let mut statement=conn.prepare("SELECT DISTINCT root FROM source_assertion WHERE kind IN ('saved_query','tag') ORDER BY root")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            self.check()?;
            visit(self.completion_definition(&row.get::<_, String>(0)?)?)?;
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
        let mut change = DefinitionChange {
            family: family.into(),
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
                output_metadata: output_metadata
                    .and_then(|value| serde_json::from_str(&value).ok()),
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
    /// Key vocabulary is a pinned source snapshot. No first-admission counter
    /// can serve as a withdrawal or whole-family retraction watermark.
    pub fn completion_setting_keys(&self, _after: Option<i64>) -> Result<Vec<String>> {
        Ok(self
            .conn()?
            .prepare("SELECT DISTINCT key FROM source_assertion WHERE kind='setting' ORDER BY key")?
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }
}

#[cfg(test)]
mod tests;
