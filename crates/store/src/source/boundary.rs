use crate::{Archive, Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::os::unix::ffi::OsStrExt;

/// A pre-acquisition cut of source authority. It is not an index write counter.
/// Primary cuts use the scoped, atomically applied confirmed-stream prefix.
/// A reported stream head or a staged page cannot construct one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceBoundary {
    generation: String,
    scope_root: [u8; 32],
    authority: Authority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Authority {
    Standalone { through: u64 },
    Primary { through: u64, scope: String },
}

impl SourceBoundary {
    pub(crate) fn read(
        index: &Connection,
        source: &Connection,
        archive: &Archive,
    ) -> Result<Option<Self>> {
        let authority = if archive.is_primary() {
            let Some((through, scope)) = primary_position(index)? else {
                return Ok(None);
            };
            Authority::Primary { through, scope }
        } else {
            let Some(through) = source_position(source)? else {
                return Ok(None);
            };
            Authority::Standalone { through }
        };
        let generation = index.query_row(
            "SELECT value FROM main.meta WHERE key='generation'",
            [],
            |row| row.get(0),
        )?;
        Ok(Some(Self {
            generation,
            scope_root: scope_root(archive),
            authority,
        }))
    }

    pub(crate) fn through(&self, index: &Connection, archive: &Archive) -> Result<u64> {
        let generation: String = index.query_row(
            "SELECT value FROM main.meta WHERE key='generation'",
            [],
            |row| row.get(0),
        )?;
        if generation != self.generation || scope_root(archive) != self.scope_root {
            return Err(StoreError::ScopeChanged);
        }
        match &self.authority {
            Authority::Standalone { through } if !archive.is_primary() => Ok(*through),
            Authority::Primary { through, scope } if archive.is_primary() => {
                // Later unknown inputs affect current head claims, not this
                // already completed historical cut. Lifecycle changes rotate generation.
                match applied_prefix(index)? {
                    Some((applied, current_scope))
                        if applied >= *through && &current_scope == scope =>
                    {
                        Ok(*through)
                    }
                    _ => Err(StoreError::ScopeChanged),
                }
            }
            _ => Err(StoreError::ScopeChanged),
        }
    }
}

fn primary_position(index: &Connection) -> Result<Option<(u64, String)>> {
    crate::primary::scope::require_readable(index)?;
    let unreadable: bool = index.query_row(
        "SELECT EXISTS(SELECT 1 FROM primary_refused r LEFT JOIN primary_fact f ON f.event=r.event WHERE r.unreadable_input=1 AND (f.event IS NULL OR (f.state='live' AND f.product IS NULL)))",
        [], |row| row.get(0),
    )?;
    if unreadable {
        return Ok(None);
    }
    applied_prefix(index)
}

fn applied_prefix(index: &Connection) -> Result<Option<(u64, String)>> {
    crate::primary::scope::require_readable(index)?;
    let state = crate::primary::projection::replay_state(index)?;
    let Some(position) = state.applied_position else {
        return Ok(None);
    };
    let Some(scope) = state.scope.filter(|scope| !scope.is_empty()) else {
        return Ok(None);
    };
    let through = u64::try_from(position)
        .map_err(|_| StoreError::Invalid("confirmed source position cannot be negative".into()))?;
    if crate::primary::projection::checkpoint(index)? != Some(position)
        || crate::primary::scope::current(index)?.as_deref() != Some(scope.as_str())
        || source_position(index)? != Some(through)
    {
        return Ok(None);
    }
    Ok(Some((through, scope)))
}

fn source_position(conn: &Connection) -> Result<Option<u64>> {
    let source = super::Source::reading(conn)?;
    let position: Option<String> = conn
        .query_row(
            &source.sql("SELECT value FROM @source.meta WHERE key='write_counter'"),
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(position) = position {
        return position.parse::<u64>().map(Some).map_err(|_| {
            StoreError::Invalid("source admission fence is not a nonnegative integer".into())
        });
    }
    let empty: bool = conn.query_row(
        &source.sql("SELECT NOT EXISTS(SELECT 1 FROM @source.input)"),
        [],
        |row| row.get(0),
    )?;
    Ok(empty.then_some(0))
}

// This binds the existing scope registry key; it does not create an archive identity.
fn scope_root(archive: &Archive) -> [u8; 32] {
    *blake3::hash(Archive::resolved_root(&archive.root).as_os_str().as_bytes()).as_bytes()
}
