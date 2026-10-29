//! Bounded projections from one authoritative read, consumed by the catalog publisher.
use super::catalog::Session;
use crate::{Reader, Result, StoreError};
use polylogue_model::ChangePosition;
use rusqlite::{OptionalExtension, params};
use std::collections::BTreeMap;

pub enum SessionChange {
    Upsert(Session),
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
    pub id: String,
    pub name: Option<String>,
    pub unit: String,
    pub columns: serde_json::Value,
    pub match_columns: Option<serde_json::Value>,
    pub row_count: u64,
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
        if let Some(after) = after {
            if after.generation != position.generation
                || after.index > position.index
                || after.archive > position.archive
            {
                return Err(StoreError::Invalid("completion generation changed".into()));
            }
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
            visit(SessionChange::Upsert(item))?;
        }
        if let Some(after) = after {
            let mut removed = conn
                .prepare("SELECT r.id FROM removed_session r WHERE r.changed>?1 AND NOT EXISTS(SELECT 1 FROM session s WHERE s.id=r.id) ORDER BY r.changed,r.id")?;
            for id in removed.query_map([after.index], |row| row.get::<_, String>(0))? {
                visit(SessionChange::Removed(id?))?;
            }
        }
        Ok(position)
    }
    /// Only families changed since the source receipt, with withdrawal revisions
    /// included. The live-note predicate is the same canonical model predicate.
    pub fn visit_completion_definitions(
        &self,
        after: Option<i64>,
        mut visit: impl FnMut(DefinitionChange) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn()?;
        let sql = if after.is_some() {
            "SELECT DISTINCT first_note FROM source_note WHERE commit_seq>?1 UNION SELECT substr(r.other,6) FROM source_note_relation r JOIN source_note n ON n.id=r.note_id WHERE n.commit_seq>?1 AND r.kind='withdraws' AND r.other LIKE 'note:%'"
        } else {
            "SELECT DISTINCT first_note FROM source_note WHERE kind IN ('saved_query','tag')"
        };
        let mut statement = conn.prepare(sql)?;
        let mut rows = if let Some(after) = after {
            statement.query([after])?
        } else {
            statement.query([])?
        };
        let current = format!(
            "SELECT n.id,n.kind,n.key,n.value FROM source_note n WHERE n.first_note=?1 AND {}",
            polylogue_model::note_sql::live_note("n")
        );
        let mut live = conn.prepare(&current)?;
        while let Some(row) = rows.next()? {
            self.check()?;
            let family: String = row.get(0)?;
            let row = live
                .query_row([&family], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, Option<String>>(3)?,
                    ))
                })
                .optional()?;
            let mut change = DefinitionChange {
                family,
                saved: None,
                tag: None,
            };
            if let Some((revision, kind, key, value)) = row {
                if kind == "saved_query" {
                    if let (Some(name), Some(text)) = (key, value) {
                        change.saved = Some(NativeSaved {
                            revision,
                            name,
                            text,
                        });
                    }
                } else if kind == "tag" {
                    change.tag = key.map(|key| (key, value));
                }
            }
            visit(change)?;
        }
        Ok(())
    }
    pub fn visit_completion_kept(
        &self,
        after: Option<i64>,
        mut visit: impl FnMut(NativeKept) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn()?;
        let mut statement=conn.prepare(if after.is_some() {
            "SELECT k.id,k.name,json_extract(k.rows_json,'$.unit'),json_extract(k.rows_json,'$.columns'),json_extract(k.rows_json,'$.match_columns'),json_array_length(k.rows_json,'$.rows') FROM source_kept_result k JOIN source_client_request r ON r.id=k.id WHERE r.commit_seq>?1 ORDER BY r.commit_seq,k.id"
        } else {"SELECT id,name,json_extract(rows_json,'$.unit'),json_extract(rows_json,'$.columns'),json_extract(rows_json,'$.match_columns'),json_array_length(rows_json,'$.rows') FROM source_kept_result ORDER BY id"})?;
        let mut rows = if let Some(after) = after {
            statement.query([after])?
        } else {
            statement.query([])?
        };
        while let Some(row) = rows.next()? {
            self.check()?;
            visit(NativeKept {
                id: row.get(0)?,
                name: row.get(1)?,
                unit: row.get(2)?,
                columns: serde_json::from_str(&row.get::<_, String>(3)?)?,
                match_columns: row
                    .get::<_, Option<String>>(4)?
                    .map(|value| serde_json::from_str(&value))
                    .transpose()?,
                row_count: row.get::<_, i64>(5)? as u64,
            })?;
        }
        Ok(())
    }
    pub fn completion_setting_keys(&self, after: Option<i64>) -> Result<Vec<String>> {
        let mut statement = self
            .conn()?
            .prepare("SELECT DISTINCT key FROM source_setting WHERE commit_seq>?1 ORDER BY key")?;
        Ok(statement
            .query_map(params![after.unwrap_or(-1)], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }
}
