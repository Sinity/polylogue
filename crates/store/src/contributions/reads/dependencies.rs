//! Snapshot-pinned admission catalog, independent of selected Message rows.
use super::*;
use crate::{ImmutableImport, RetainedInput, parser_state};
#[derive(Clone)]
pub struct SessionDependencyCursor {
    read: SharedRead,
    session: i64,
    after: String,
}
pub struct SessionInputDependency {
    read: SharedRead,
    id: String,
    hash: crate::BlobHash,
    format: crate::InputFormat,
    context: Vec<String>,
    length: u64,
    registered: Option<i64>,
    head: Option<parser_state::Head>,
    failure: Option<String>,
}
impl SessionInputDependency {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn format(&self) -> crate::InputFormat {
        self.format
    }
    pub fn context(&self) -> &[String] {
        &self.context
    }
    pub fn head(&self) -> Option<&parser_state::Head> {
        self.head.as_ref()
    }
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }
    pub fn registered(&self) -> Option<ImmutableImport> {
        self.registered.map(|input_pk| ImmutableImport {
            input: RetainedInput {
                input_pk,
                id: self.id.clone(),
                chain: 1,
                length: self.length,
            },
            hash: self.hash,
            format: self.format,
            context: self.context.clone(),
        })
    }
}
impl SessionView {
    pub fn source_archive_counter(&self) -> u64 {
        self.archive_counter
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
            Some(cursor) if Arc::ptr_eq(&self.read, &cursor.read) && cursor.session == self.pk => {
                &cursor.after
            }
            Some(_) => return Err(Error::MessagePredecessorChanged),
        };
        let read = locked(&self.read)?;
        let archive = read
            .archive
            .as_ref()
            .ok_or_else(|| Error::Invariant("Session admission snapshot missing".into()))?;
        let path = self.path.with_file_name("archive.db");
        // Current removal authority is independent of a computational snapshot.
        let visibility = read
            .archive_visibility
            .as_ref()
            .ok_or_else(|| Error::Invariant("Session archive authority missing".into()))?;
        let excision: bool = db(&path, || {
            visibility.query_row("SELECT EXISTS(SELECT 1 FROM excision LIMIT 1)", [], |r| {
                r.get(0)
            })
        })?;
        if excision {
            return Err(Error::ExcisionCheckRequired);
        }
        let row: Option<(
            u64,
            Option<String>,
            Option<String>,
            i64,
            Vec<u8>,
            Vec<u8>,
            Option<String>,
        )> = db(&path, || {
            archive.query_row(
            "SELECT length(CAST(i.id AS BLOB))+length(CAST(i.format AS BLOB))+length(CAST(i.context_json AS BLOB)),CASE WHEN length(CAST(i.id AS BLOB))+length(CAST(i.format AS BLOB))+length(CAST(i.context_json AS BLOB))<=?2 THEN i.id END,CASE WHEN length(CAST(i.id AS BLOB))+length(CAST(i.format AS BLOB))+length(CAST(i.context_json AS BLOB))<=?2 THEN i.format END,r.length,r.blake3,i.original_hash,CASE WHEN length(CAST(i.id AS BLOB))+length(CAST(i.format AS BLOB))+length(CAST(i.context_json AS BLOB))<=?2 THEN i.context_json END FROM input i JOIN revision r ON r.input=i.id AND r.number=1 WHERE i.id>?1 AND r.produced='imported' AND r.chain=1 ORDER BY i.id LIMIT 1",
            params![key,i64::try_from(allowance).unwrap_or(i64::MAX)],|r|Ok((parser_state::unsigned(r,0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?))).optional()
        })?;
        let Some((required, id, format, length, hash, original, context)) = row else {
            return Ok(None);
        };
        let (Some(id), Some(format), Some(context)) = (id, format, context) else {
            return Err(metadata_admission("Session dependency identity", required));
        };
        let (id, hash, format, context, length) =
            crate::retained::decode_import_identity(id, format, length, hash, original, context)?;
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
            .map(|pk| parser_state::head(&read.connection, &self.path, pk, 1))
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
                session: self.pk,
                after: id.clone(),
            },
            SessionInputDependency {
                read: Arc::clone(&self.read),
                id,
                hash,
                format,
                context,
                length,
                registered: pk,
                head,
                failure,
            },
        )))
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
