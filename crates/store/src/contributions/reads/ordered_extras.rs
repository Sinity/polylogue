//! Canonical extra-key traversal over one immutable selected root. SQLite's
//! private file-backed TEMP B-tree sorts keys/handles, never field payloads.
//! Current typed immediate keys are scalar strings: binary UTF-8 order is their
//! canonical WTF-8 order. Nested non-scalar keys stay owned by RetainedJson.
use super::*;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use std::time::Instant;

const PREFIX: &str = "polylogue_extra_order_";

/// Called once, before a new pinned connection starts its read transaction.
/// SQLite owns private TEMP files under the process startup temp environment;
/// deployment must qualify that placement as private and on disk. No global
/// SQLite directory or retained/provider path is changed here.
pub(super) fn configure(connection: &Connection, path: &std::path::Path) -> Result<()> {
    db(path, || {
        connection.execute_batch(
            "PRAGMA temp_store=FILE; PRAGMA temp.cache_size=-64; PRAGMA temp.cache_spill=ON;",
        )
    })?;
    db(path, || {
        connection.authorizer(Some(|context: AuthContext<'_>| {
            use AuthAction::*;
            match context.action {
                Read { .. } | Select | Recursive | Transaction { .. } | Savepoint { .. } => {
                    Authorization::Allow
                }
                Function { function_name }
                    if !function_name.eq_ignore_ascii_case("load_extension") =>
                {
                    Authorization::Allow
                }
                Pragma {
                    pragma_value: None, ..
                } => Authorization::Allow,
                CreateTempTable { table_name }
                | DropTempTable { table_name }
                | CreateTempIndex { table_name, .. }
                | DropTempIndex { table_name, .. }
                    if table_name.starts_with(PREFIX) =>
                {
                    Authorization::Allow
                }
                Insert { .. } | Update { .. } | Delete { .. }
                    if context.database_name == Some("temp") =>
                {
                    Authorization::Allow
                }
                // OS-read-only main remains an independent boundary. In particular,
                // TEMP support grants neither main writes nor arbitrary ATTACH.
                _ => Authorization::Deny,
            }
        }))
    })
}

pub struct ExtraFieldEntry<'a> {
    pub key: String,
    pub value: FieldRead<'a>,
}

/// Request-local ordering state. It borrows the selected Message, so body/root
/// membership and visibility remain the existing reader's authority. Dropping
/// it removes only its own TEMP table, including on sink/cancellation errors.
pub struct OrderedExtras<'a> {
    message: &'a MessageView,
    root: Option<i64>,
    scope: FieldScope,
    table: String,
    after: Option<Vec<u8>>,
    closed: bool,
    failed: bool,
    allowance: usize,
}
fn check(until: Instant, cancelled: &mut dyn FnMut() -> bool) -> Result<()> {
    if Instant::now() >= until || cancelled() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
impl MessageView {
    pub fn ordered_extras(
        &self,
        allowance: usize,
        until: Instant,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<OrderedExtras<'_>> {
        self.visible()?;
        let root = {
            let read = locked(&self.read)?;
            db(&self.path, || {
                read.connection.query_row("SELECT field_root_pk FROM message_body WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND ready=1", params![self.body,self.owner.message,self.owner.session], |r|r.get(0))
            })?
        };
        OrderedExtras::build(self, root, FieldScope::Message, allowance, until, cancelled)
    }
}
impl<'a> BlockView<'a> {
    pub fn ordered_extras(
        &self,
        allowance: usize,
        until: Instant,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<OrderedExtras<'a>> {
        OrderedExtras::build(
            self.message,
            self.field_root,
            FieldScope::Block {
                group: self.group,
                local: self.local,
            },
            allowance,
            until,
            cancelled,
        )
    }
}
impl<'a> OrderedExtras<'a> {
    fn build(
        message: &'a MessageView,
        root: Option<i64>,
        scope: FieldScope,
        allowance: usize,
        until: Instant,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Self> {
        check(until, cancelled)?;
        message.visible()?;
        let table = {
            let mut read = locked(&message.read)?;
            read.extra_orders = read.extra_orders.checked_add(1).ok_or(Error::Range)?;
            let table = format!("{PREFIX}{}", read.extra_orders);
            db(&message.path, || {
                read.connection.execute_batch(&format!("CREATE TEMP TABLE {table}(sort_key BLOB PRIMARY KEY, semantic_key BLOB NOT NULL) STRICT, WITHOUT ROWID;"))
            })?;
            table
        };
        let ordered = Self {
            message,
            root,
            scope,
            table,
            after: None,
            closed: false,
            failed: false,
            allowance,
        };
        let mut cursor = None;
        loop {
            check(until, cancelled)?;
            message.visible()?;
            // Existing directory traversal validates shape, ownership and
            // current-root membership; historical slots never enter the sort.
            let found = {
                let read = locked(&message.read)?;
                message.field_successor(
                    &read.connection,
                    root,
                    scope,
                    cursor.as_ref(),
                    allowance,
                )?
            };
            let Some((next, semantic)) = found else { break };
            let key = extra_key(scope, &semantic)?;
            if let Some(key) = key {
                check(until, cancelled)?;
                let read = locked(&message.read)?;
                db(&message.path, || {
                    read.connection.execute(
                        &format!(
                            "INSERT INTO {}(sort_key,semantic_key) VALUES(?1,?2)",
                            ordered.table
                        ),
                        params![key.as_bytes(), semantic],
                    )
                })?;
            }
            cursor = Some(next);
        }
        Ok(ordered)
    }
    /// One indexed successor and one existing pinned field resolution. No
    /// statement or lock survives the call, and no model-page state is kept.
    pub fn next(
        &mut self,
        until: Instant,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Option<ExtraFieldEntry<'a>>> {
        if self.failed {
            return Err(Error::Invariant(
                "ordered extra cursor already failed".into(),
            ));
        }
        if self.closed {
            return Ok(None);
        }
        match self.next_inner(until, cancelled) {
            Ok(Some(entry)) => Ok(Some(entry)),
            Ok(None) => match self.close() {
                Ok(()) => Ok(None),
                Err(error) => {
                    self.failed = true;
                    Err(error)
                }
            },
            Err(error) => {
                self.failed = true;
                let _ = self.close();
                Err(error)
            }
        }
    }
    fn next_inner(
        &mut self,
        until: Instant,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Option<ExtraFieldEntry<'a>>> {
        check(until, cancelled)?;
        self.message.visible()?;
        let row: Option<(Vec<u8>, Vec<u8>)> = {
            let read = locked(&self.message.read)?;
            if let Some(after) = &self.after {
                db(&self.message.path, || {
                    read.connection.query_row(&format!("SELECT sort_key,semantic_key FROM {} WHERE sort_key>?1 ORDER BY sort_key LIMIT 1", self.table), [after], |r|Ok((r.get(0)?,r.get(1)?))).optional()
                })?
            } else {
                db(&self.message.path, || {
                    read.connection
                        .query_row(
                            &format!(
                                "SELECT sort_key,semantic_key FROM {} ORDER BY sort_key LIMIT 1",
                                self.table
                            ),
                            [],
                            |r| Ok((r.get(0)?, r.get(1)?)),
                        )
                        .optional()
                })?
            }
        };
        let Some((key, semantic)) = row else {
            return Ok(None);
        };
        if key.len() > self.allowance || semantic.len() > self.allowance {
            return Err(metadata_admission(
                "ordered extra key",
                key.len().max(semantic.len()) as u64,
            ));
        }
        let key_text = std::str::from_utf8(&key)
            .map_err(|_| Error::Invariant("ordered extra key is not scalar UTF-8".into()))?
            .to_owned();
        if extra_key(self.scope, &semantic)?.as_deref() != Some(key_text.as_str()) {
            return Err(Error::Invariant("ordered extra key/handle mismatch".into()));
        }
        let value = self
            .message
            .field_at(self.root, &semantic)?
            .ok_or_else(|| Error::Invariant("ordered extra lost selected membership".into()))?;
        self.after = Some(key);
        Ok(Some(ExtraFieldEntry {
            key: key_text,
            value,
        }))
    }
    fn close(&mut self) -> Result<()> {
        if !self.closed {
            let read = locked(&self.message.read)?;
            db(&self.message.path, || {
                read.connection
                    .execute_batch(&format!("DROP TABLE IF EXISTS temp.{}", self.table))
            })?;
            self.closed = true;
        }
        Ok(())
    }
}
impl Drop for OrderedExtras<'_> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
fn extra_key(scope: FieldScope, semantic: &[u8]) -> Result<Option<String>> {
    let invalid = |_| Error::Invariant("invalid selected extra semantic key".into());
    match scope {
        FieldScope::Message => {
            match serde_json::from_slice::<polylogue_model::merge::Slot>(semantic)
                .map_err(invalid)?
            {
                polylogue_model::merge::Slot::Extra { block: None, key } => Ok(Some(key)),
                polylogue_model::merge::Slot::Block { .. }
                | polylogue_model::merge::Slot::Extra { block: Some(_), .. } => Err(
                    Error::Invariant("block-local address in Message field root".into()),
                ),
                _ => Ok(None),
            }
        }
        FieldScope::Block { .. } => {
            match serde_json::from_slice::<super::super::writer::BlockFieldKey<'_>>(semantic)
                .map_err(invalid)?
            {
                super::super::writer::BlockFieldKey::Extra(key) => Ok(Some(key.into_owned())),
                _ => Ok(None),
            }
        }
    }
}

#[cfg(test)]
mod tests;
