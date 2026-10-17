//! Snapshot-pinned admission catalog, independent of selected Message rows.
use super::*;
use crate::{RetainedSource, parser_state};
#[derive(Clone)]
pub struct SessionDependencyCursor {
    read: SharedRead,
    after: String,
}
pub struct SessionInputDependency {
    read: SharedRead,
    id: String,
    source: RetainedSource,
    registered: Option<i64>,
    head: Option<parser_state::Head>,
    failure: Option<String>,
}
impl SessionInputDependency {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn format(&self) -> crate::InputFormat {
        self.source.format
    }
    pub fn context(&self) -> &[String] {
        &self.source.context
    }
    pub fn head(&self) -> Option<&parser_state::Head> {
        self.head.as_ref()
    }
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }
    pub fn registered(&self) -> Option<RetainedSource> {
        self.registered.map(|input_pk| {
            let mut source = self.source.clone();
            source.input.input_pk = input_pk;
            source
        })
    }
}
impl SessionCatalog {
    pub fn verify_dependency_bodies(&self, dependency: &SessionInputDependency) -> Result<()> {
        self.visible()?;
        if !Arc::ptr_eq(&self.read, &dependency.read) {
            return Err(Error::MessagePredecessorChanged);
        }
        let read = locked(&self.read)?;
        let archive = read
            .archive
            .as_ref()
            .ok_or_else(|| Error::Invariant("Session archive snapshot missing".into()))?;
        for body in dependency
            .source
            .companions
            .iter()
            .filter(|body| !body.inline_duplicate)
        {
            crate::claude_bodies::read_pinned(
                archive,
                self.path.parent().ok_or(Error::Range)?,
                body,
            )?;
        }
        Ok(())
    }
    /// One unique retained input, even when it has no selected/catalog Message.
    /// `None` ends only this fixed admission catalog, not Session completeness.
    pub fn next_dependency(
        &self,
        after: Option<&SessionDependencyCursor>,
        allowance: usize,
    ) -> Result<Option<(SessionDependencyCursor, SessionInputDependency)>> {
        self.visible()?;
        let key = match after {
            None => "",
            Some(cursor) if Arc::ptr_eq(&self.read, &cursor.read) => &cursor.after,
            Some(_) => return Err(Error::MessagePredecessorChanged),
        };
        let read = locked(&self.read)?;
        let archive = read
            .archive
            .as_ref()
            .ok_or_else(|| Error::Invariant("Session admission snapshot missing".into()))?;
        let path = self.path.with_file_name("archive.db");
        let id: Option<(u64, Option<String>)> = db(&path, || {
            archive.query_row("SELECT length(CAST(id AS BLOB)),CASE WHEN length(CAST(id AS BLOB))<=?2 THEN id END FROM input WHERE id>?1 AND format<>'claude-code-tool-result' ORDER BY id LIMIT 1", params![key,i64::try_from(allowance).unwrap_or(i64::MAX)],|r| Ok((parser_state::unsigned(r,0)?,r.get(1)?))).optional()
        })?;
        let Some((required, id)) = id else {
            return Ok(None);
        };
        let id = id.ok_or_else(|| metadata_admission("Session dependency identity", required))?;
        let source = crate::observed::decode(archive, &path, &id)?
            .ok_or_else(|| Error::Invariant("retained dependency disappeared".into()))?;
        let required = id.len() + source.context.iter().map(String::len).sum::<usize>();
        if required > allowance {
            return Err(metadata_admission(
                "Session dependency context",
                required as u64,
            ));
        }
        let registered: Option<(i64, bool)> = db(&self.path, || {
            read.connection
                .query_row("SELECT pk,excised FROM input_ref WHERE id=?1", [&id], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })
                .optional()
        })?;
        if matches!(registered, Some((_, true))) {
            return Err(Error::ExcisionCheckRequired);
        }
        let pk = registered.map(|(pk, _)| pk);
        let head = pk
            .map(|pk| parser_state::head(&read.connection, &self.path, pk, source.input.chain))
            .transpose()?
            .flatten();
        let failure = if let Some(head) = &head {
            let row: (u64, Option<String>) = db(&self.path, || {
                read.connection.query_row("SELECT coalesce(length(CAST(detail AS BLOB)),0),CASE WHEN length(CAST(detail AS BLOB))<=?3 THEN detail END FROM parse_state WHERE input_pk=?1 AND chain=?2",params![head.token.input_pk,head.token.chain,i64::try_from(allowance).unwrap_or(i64::MAX)],|r|Ok((parser_state::unsigned(r,0)?,r.get(1)?)))
            })?;
            if row.0 > allowance as u64 {
                return Err(metadata_admission("Session dependency diagnostic", row.0));
            }
            row.1
        } else {
            None
        };
        Ok(Some((
            SessionDependencyCursor {
                read: Arc::clone(&self.read),
                after: id.clone(),
            },
            SessionInputDependency {
                read: Arc::clone(&self.read),
                id,
                source,
                registered: pk,
                head,
                failure,
            },
        )))
    }
    /// Existing non-Message products must not be silently omitted by the
    /// narrow native Session reader. This check is local to the exact owner.
    pub fn has_other_session_products(&self, id: &SessionId) -> Result<bool> {
        self.visible()?;
        let read = locked(&self.read)?;
        let products: bool = db(&self.path, || {
            read.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM link WHERE child_id=?1 OR parent_id=?1) OR EXISTS(SELECT 1 FROM event e JOIN session s ON s.pk=e.session_pk WHERE s.id=?1) OR EXISTS(SELECT 1 FROM delivery WHERE recipient_id=?1) OR EXISTS(SELECT 1 FROM run_session r JOIN session s ON s.pk=r.session_pk WHERE s.id=?1)", [id.as_str()], |r|r.get(0))
        })?;
        let archive = read
            .archive
            .as_ref()
            .ok_or_else(|| Error::Invariant("Session archive snapshot missing".into()))?;
        let annotations: bool = db(&self.path.with_file_name("archive.db"), || {
            archive.query_row(
            "SELECT EXISTS(SELECT 1 FROM note_target WHERE ref=?1 OR ref=?2 OR substr(ref,1,length(?3))=?3 OR substr(ref,1,length(?4))=?4)",
            params![id.as_str(),format!("session:{}",id.as_str()),format!("message:{}:",id.as_str()),format!("block:{}:",id.as_str())],|r|r.get(0))
        })?;
        Ok(products || annotations)
    }
    pub fn dependency_first_path(
        &self,
        dependency: &SessionInputDependency,
        allowance: usize,
    ) -> Result<Option<String>> {
        self.visible()?;
        if !Arc::ptr_eq(&self.read, &dependency.read) {
            return Err(Error::MessagePredecessorChanged);
        }
        let read = locked(&self.read)?;
        let archive = read
            .archive
            .as_ref()
            .ok_or_else(|| Error::Invariant("Session archive snapshot missing".into()))?;
        let row: Option<(u64, Option<String>)> = db(
            &self.path.with_file_name("archive.db"),
            || {
                archive.query_row(
            "SELECT length(CAST(path AS BLOB)),CASE WHEN length(CAST(path AS BLOB))<=?2 THEN path END FROM observation WHERE input=?1 ORDER BY rowid LIMIT 1",params![dependency.id(),i64::try_from(allowance).unwrap_or(i64::MAX)],|r|Ok((parser_state::unsigned(r,0)?,r.get(1)?))).optional()
            },
        )?;
        row.map(|(n, path)| path.ok_or_else(|| metadata_admission("Session input path", n)))
            .transpose()
    }
    /// Reuse the source audit's current-token and selected-body proof within
    /// this catalog's already pinned index and retained admission view.
    pub fn dependency_selected_projection(
        &self,
        dependency: &SessionInputDependency,
        logical: &[u8; 32],
        allowance: usize,
    ) -> Result<Option<crate::contributions::SourceProjection>> {
        self.visible()?;
        if !Arc::ptr_eq(&self.read, &dependency.read) {
            return Err(Error::MessagePredecessorChanged);
        }
        let Some(head) = dependency.head() else {
            return Ok(None);
        };
        let read = locked(&self.read)?;
        super::super::source::read_selected(
            &read.connection,
            &self.path,
            &head.token,
            logical,
            allowance,
        )
    }
    /// A point read from the SAME pinned index as Session rows and admission
    /// traversal. The dependency is unforgeable and bound to that read Arc.
    pub fn dependency_cell(
        &self,
        dependency: &SessionInputDependency,
        namespace: &str,
        key: &[u8],
        allowance: usize,
    ) -> Result<Option<Vec<u8>>> {
        self.visible()?;
        if !Arc::ptr_eq(&self.read, &dependency.read) {
            return Err(Error::MessagePredecessorChanged);
        }
        let Some(head) = dependency.head.as_ref() else {
            return Ok(None);
        };
        let read = locked(&self.read)?;
        let row: Option<(u64, Option<Vec<u8>>)> = db(&self.path, || {
            read.connection.query_row("SELECT length(value),CASE WHEN length(value)<=?6 THEN value END FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",params![head.token.input_pk,head.token.chain,head.token.generation,namespace,key,i64::try_from(allowance).unwrap_or(i64::MAX)],|r|Ok((parser_state::unsigned(r,0)?,r.get(1)?))).optional()
        })?;
        row.map(|(n, bytes)| bytes.ok_or_else(|| metadata_admission("Session dependency cell", n)))
            .transpose()
    }
}

impl SessionView {
    pub fn source_archive_counter(&self) -> u64 {
        self.archive_counter
    }
    pub fn next_dependency(
        &self,
        after: Option<&SessionDependencyCursor>,
        allowance: usize,
    ) -> Result<Option<(SessionDependencyCursor, SessionInputDependency)>> {
        self.visible()?;
        self.catalog().next_dependency(after, allowance)
    }
    pub fn dependency_cell(
        &self,
        dependency: &SessionInputDependency,
        namespace: &str,
        key: &[u8],
        allowance: usize,
    ) -> Result<Option<Vec<u8>>> {
        self.visible()?;
        self.catalog()
            .dependency_cell(dependency, namespace, key, allowance)
    }
}
