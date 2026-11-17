//! Reconstructible keyed continuation. Speculative values live in one owned
//! scratch database; committed mappings share the parser checkpoint transaction.
use crate::ConnExt as _;
use crate::ResultExt as _;
use crate::{Archive, Result, StoreError};
use polylogue_model::parser_context::{Context, Delta, Entry, Key, Reference, Value};
use rusqlite::{Connection, OptionalExtension, params};
use std::sync::{Arc, Mutex};

// A chain's context belongs to one reading, named once in its state row: a
// reading from byte zero replaces the whole context, and an extension applies
// only on top of a state row of its own reading version. Each key keeps only
// its value at that checkpoint (`through`): a resumed parse reads it under the
// read snapshot it opened at the checkpoint, so a value a later checkpoint
// replaced is never read again.
pub(crate) const SCHEMA: &str = "CREATE TABLE parser_context_state(input_pk INTEGER NOT NULL REFERENCES input_ref(pk),chain INTEGER NOT NULL,reading_version TEXT NOT NULL,through INTEGER NOT NULL,PRIMARY KEY(input_pk,chain)) STRICT, WITHOUT ROWID;
CREATE TABLE parser_context_response(input_pk INTEGER NOT NULL REFERENCES input_ref(pk),chain INTEGER NOT NULL,namespace TEXT NOT NULL,session TEXT NOT NULL,key TEXT NOT NULL,version_pk INTEGER NOT NULL REFERENCES message_version(pk),source_start INTEGER NOT NULL,source_end INTEGER NOT NULL,PRIMARY KEY(input_pk,chain,namespace,session,key)) STRICT, WITHOUT ROWID;
CREATE TABLE parser_context_alias(input_pk INTEGER NOT NULL REFERENCES input_ref(pk),chain INTEGER NOT NULL,namespace TEXT NOT NULL,uuid TEXT NOT NULL,target TEXT,PRIMARY KEY(input_pk,chain,namespace,uuid)) STRICT, WITHOUT ROWID;
CREATE TABLE parser_context_value(input_pk INTEGER NOT NULL REFERENCES input_ref(pk),chain INTEGER NOT NULL,namespace TEXT NOT NULL,key TEXT NOT NULL,value BLOB NOT NULL,PRIMARY KEY(input_pk,chain,namespace,key)) STRICT, WITHOUT ROWID;";

fn scratch_values() -> Result<Connection> {
    let scratch = Connection::open("")?;
    scratch.execute_batch("PRAGMA cache_size=-256; PRAGMA mmap_size=0; PRAGMA temp_store=FILE; PRAGMA journal_mode=OFF; PRAGMA cache_spill=ON; CREATE TABLE entry(key TEXT PRIMARY KEY,value BLOB NOT NULL,dirty INTEGER NOT NULL,state_namespace TEXT) WITHOUT ROWID; CREATE INDEX entry_dirty ON entry(key) WHERE dirty=1; CREATE INDEX entry_state_namespace ON entry(state_namespace) WHERE state_namespace IS NOT NULL; CREATE TABLE cleared_state(namespace TEXT PRIMARY KEY,dirty INTEGER NOT NULL) WITHOUT ROWID; BEGIN;")?;
    Ok(scratch)
}

#[derive(Clone)]
pub struct Overlay(Arc<Mutex<Attempt>>);
struct Attempt {
    scratch: Connection,
    index: Option<Connection>,
    input: Option<i64>,
    chain: Option<i64>,
    source: String,
    source_chain: Option<i64>,
    version: String,
    /// Fixed cutoff of the committed snapshot selected when this attempt opened.
    lookup_through: u64,
    /// Cursor through which this private attempt has emitted staged deltas.
    base: u64,
    replay_chain: Option<i64>,
}
impl Overlay {
    /// Select only the committed continuation. No public-index row stands
    /// for a pending producer reading.
    pub fn open(
        archive: &Archive,
        input: &str,
        chain: Option<i64>,
        version: &str,
        base: u64,
    ) -> Result<Self> {
        let source = input.to_owned();
        let source_chain = chain;
        let sql_base = sql_cursor(base)?;
        // A fresh attempt gets every value from its private replay. It never
        // consults committed rows, so it must not hold a read snapshot that
        // prevents the writer from restarting its WAL between transactions.
        let index = if base == 0 {
            None
        } else {
            let index = crate::archive::open_read(&archive.index_db())?;
            crate::read::limit::ReadLimit::observe_current(&index)?;
            // A build file has a rollback journal, where a read held across
            // the parse would block every writer commit
            // (`crate::archive::index_build`). There each lookup reads on its
            // own: the chain's committed context changes only by this
            // attempt's own slices, whose values the scratch view already
            // holds, so the lookups see what the snapshot would.
            if archive.is_primary() || !archive.index_build.building() {
                index.execute_batch("BEGIN")?;
            }
            Some(index)
        };
        let selected = match (index.as_ref(), chain) {
            (Some(index), Some(chain)) => index.query_row("SELECT i.pk,p.chain FROM input_ref i JOIN parse_state p ON p.input_pk=i.pk WHERE i.id=?1 AND p.chain=?2 AND p.reading_version=?3 AND p.parsed_upto=?4",params![input,chain,version,sql_base],|r| <(i64, i64)>::try_from(r)).optional()?,
            (None, _) => None,
            (_, None) => return Err(StoreError::Invalid("parser continuation has no selected chain identity".into())),
        };
        let (input, chain) = selected.map_or((None, None), |(i, c)| (Some(i), Some(c)));
        let mut replay_chain = None;
        if base > 0 {
            let complete = if let (Some(index), Some(input), Some(chain)) =
                (index.as_ref(), input, chain)
            {
                index.query_row("SELECT through FROM parser_context_state WHERE input_pk=?1 AND chain=?2 AND reading_version=?3",params![input,chain,version],|r|r.get::<_,i64>(0)).optional()?.map(stored_cursor).transpose()? == Some(base)
            } else {
                false
            };
            if !complete {
                replay_chain = Some(chain.ok_or_else(|| {
                    StoreError::Invalid(
                        "keyed parser continuation has no committed reading at the selected cursor"
                            .into(),
                    )
                })?);
            }
        }
        let scratch = scratch_values()?;
        Ok(Self(Arc::new(Mutex::new(Attempt {
            scratch,
            index,
            input: if base == 0 || replay_chain.is_some() {
                None
            } else {
                input
            },
            chain,
            source,
            source_chain,
            version: version.into(),
            lookup_through: base,
            base: if replay_chain.is_some() { 0 } else { base },
            replay_chain,
        }))))
    }
    /// Export items reconstruct their cross-item state by replaying from byte zero.
    pub fn temporary() -> Result<Self> {
        Ok(Self(Arc::new(Mutex::new(Attempt {
            scratch: scratch_values()?,
            index: None,
            input: None,
            chain: None,
            source: String::new(),
            source_chain: None,
            version: String::new(),
            lookup_through: 0,
            base: 0,
            replay_chain: None,
        }))))
    }
    pub fn replay_chain(&self) -> Option<i64> {
        self.0.lock().expect("parser context lock").replay_chain
    }
    /// Reuse only the private continuation belonging to this attempt.
    pub fn check_source(
        &self,
        input: &str,
        chain: Option<i64>,
        version: &str,
        base: u64,
    ) -> Result<()> {
        let attempt = self.0.lock().or_invalid("parser context lock poisoned")?;
        if attempt.source != input
            || attempt.source_chain != chain
            || attempt.version != version
            || attempt.base != base
        {
            return Err(StoreError::Invalid(
                "private parser context differs from its selected source cursor".into(),
            ));
        }
        Ok(())
    }
    /// Save this slice's changes while keeping earlier private values for lookup.
    /// Advancing this scratch view never advances the committed checkpoint.
    pub fn checkpoint_delta(&self, through: u64) -> Result<Delta> {
        let mut attempt = self.0.lock().or_invalid("parser context lock poisoned")?;
        sql_cursor(through)?;
        if through < attempt.base {
            return Err(StoreError::Invalid(
                "private parser context cursor moved backward".into(),
            ));
        }
        let mut statement = attempt
            .scratch
            .prepare("SELECT key,value FROM entry WHERE dirty=1 ORDER BY key")?;
        let mut rows = statement.query([])?;
        let mut entries = Vec::new();
        while let Some(row) = rows.next()? {
            let key: Key = serde_json::from_str(&row.get::<_, String>(0)?)?;
            let value: Value = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(1)?)?;
            let value = match value {
                Value::Response { row } => Reference::Response {
                    version: row.message.content_hash()?,
                    message: row.message.id,
                    offset: row.offset,
                    end: row.end,
                },
                Value::Alias { target } => Reference::Alias { target },
                Value::State { value } => Reference::State { value },
            };
            entries.push(Entry { key, value });
        }
        drop(rows);
        drop(statement);
        let cleared_states = {
            let mut statement = attempt
                .scratch
                .prepare("SELECT namespace FROM cleared_state WHERE dirty=1 ORDER BY namespace")?;
            statement
                .query_map([], |row| row.get(0))?
                .collect::<std::result::Result<Vec<String>, _>>()?
        };
        let base = attempt.base;
        attempt
            .scratch
            .execute("UPDATE entry SET dirty=0 WHERE dirty=1", [])?;
        attempt
            .scratch
            .execute("UPDATE cleared_state SET dirty=0 WHERE dirty=1", [])?;
        attempt.scratch.execute_batch("COMMIT; BEGIN")?;
        attempt.base = through;
        attempt.replay_chain = None;
        Ok(Delta {
            version: 1,
            base,
            through,
            cleared_states,
            entries,
        })
    }
}
fn context_error(error: StoreError) -> polylogue_model::parser_context::Error {
    match error {
        StoreError::ReadCancelled | StoreError::Stopped => {
            polylogue_model::parser_context::Error::Cancelled
        }
        other => polylogue_model::parser_context::Error::Failed(other.to_string()),
    }
}
impl Context for Overlay {
    fn clear_state(
        &mut self,
        namespace: &str,
    ) -> std::result::Result<(), polylogue_model::parser_context::Error> {
        let clear = || -> Result<()> {
            if let Some(limit) = crate::read::limit::ReadLimit::current() {
                limit.check()?;
            }
            let attempt = self.0.lock().or_invalid("parser context lock poisoned")?;
            attempt
                .scratch
                .execute("DELETE FROM entry WHERE state_namespace=?1", [namespace])?;
            attempt.scratch.execute("INSERT INTO cleared_state VALUES(?1,1) ON CONFLICT(namespace) DO UPDATE SET dirty=1", [namespace])?;
            Ok(())
        };
        clear().map_err(context_error)
    }
    fn get(
        &mut self,
        key: &Key,
    ) -> std::result::Result<Option<Value>, polylogue_model::parser_context::Error> {
        self.lookup(key).map_err(context_error)
    }
    fn put(
        &mut self,
        key: Key,
        value: Value,
    ) -> std::result::Result<(), polylogue_model::parser_context::Error> {
        let put = || -> Result<()> {
            if let Some(limit) = crate::read::limit::ReadLimit::current() {
                limit.check()?;
            }
            let attempt = self.0.lock().or_invalid("parser context lock poisoned")?;
            let namespace = match &key {
                Key::State { namespace, .. } => Some(namespace.clone()),
                _ => None,
            };
            let key = serde_json::to_string(&key)?;
            let bytes = crate::encoding::pack_json(&value)?;
            attempt.scratch.execute("INSERT INTO entry VALUES(?1,?2,1,?3) ON CONFLICT(key) DO UPDATE SET value=excluded.value,dirty=1,state_namespace=excluded.state_namespace",params![key,bytes,namespace])?;
            Ok(())
        };
        put().map_err(context_error)
    }
}
impl Overlay {
    fn lookup(&self, key: &Key) -> Result<Option<Value>> {
        if let Some(limit) = crate::read::limit::ReadLimit::current() {
            limit.check()?;
        }
        let attempt = self.0.lock().or_invalid("parser context lock poisoned")?;
        let changed: Option<Vec<u8>> = attempt.scratch.scalar_optional(
            "SELECT value FROM entry WHERE key=?1",
            [serde_json::to_string(key)?],
        )?;
        if let Some(bytes) = changed {
            let value = crate::encoding::unpack_json(&bytes)?;
            return Ok(match value {
                Value::State { value } if value.is_null() => None,
                value => Some(value),
            });
        }
        if let Key::State { namespace, .. } = key
            && attempt.scratch.exists(
                "SELECT EXISTS(SELECT 1 FROM cleared_state WHERE namespace=?1)",
                [namespace],
            )?
        {
            return Ok(None);
        }
        let (Some(input), Some(chain)) = (attempt.input, attempt.chain) else {
            return Ok(None);
        };
        let index = attempt.index.as_ref().ok_or_else(|| {
            StoreError::Invalid("committed parser context has no read snapshot".into())
        })?;
        match key {
            Key::State { namespace, key } => {
                let value: Option<Vec<u8>> = index.scalar_optional(
                    "SELECT value FROM parser_context_value WHERE input_pk=?1 AND chain=?2 AND namespace=?3 AND key=?4",
                    params![input,chain,namespace,key],
                )?;
                value
                    .map(|value| {
                        Ok(Value::State {
                            value: crate::encoding::unpack_json(&value)?,
                        })
                    })
                    .transpose()
            }
            Key::Alias { namespace, uuid } => {
                let row = index
                    .prepare_cached("SELECT target FROM parser_context_alias WHERE input_pk=?1 AND chain=?2 AND namespace=?3 AND uuid=?4")?
                    .query_row(params![input, chain, namespace, uuid], |r| {
                        r.get::<_, Option<String>>(0)
                    })
                    .optional()?;
                let Some(target) = row else {
                    return Ok(None);
                };
                let target = target
                    .map(|target| polylogue_model::MessageId::parse(&target))
                    .transpose()
                    .map_err(|e| StoreError::Invalid(format!("alias context target: {e}")))?;
                Ok(Some(Value::Alias { target }))
            }
            Key::Response {
                namespace,
                session,
                key,
            } => {
                let row = index
                    .prepare_cached("SELECT v.version_hash,r.source_start,r.source_end FROM parser_context_response r JOIN message_version v ON v.pk=r.version_pk WHERE r.input_pk=?1 AND r.chain=?2 AND r.namespace=?3 AND r.session=?4 AND r.key=?5")?
                    .query_row(params![input, chain, namespace, session, key], |r| <(Vec<u8>, i64, i64)>::try_from(r))
                    .optional()?;
                let Some((hash, offset, end)) = row else {
                    return Ok(None);
                };
                let (offset, end) = (stored_cursor(offset)?, stored_cursor(end)?);
                if offset > end || end > attempt.lookup_through {
                    return Err(StoreError::Invalid(
                        "response context range exceeds its recorded cursor".into(),
                    ));
                }
                let (message, _) =
                    crate::versions::read_message(index, &hash)?.ok_or_else(|| {
                        StoreError::Invalid("response context message version is absent".into())
                    })?;
                if message.session.to_string() != *session {
                    return Err(StoreError::Invalid(
                        "response key has another source session".into(),
                    ));
                }
                Ok(Some(Value::Response {
                    row: Box::new(polylogue_model::MessageRow {
                        page_text: false,
                        id_kind: polylogue_model::IdKind::Native,
                        offset,
                        end,
                        message,
                    }),
                }))
            }
        }
    }
}

/// Returns whether the opaque checkpoint can reuse the stored keyed context.
pub(crate) fn apply(
    tx: &Connection,
    input: i64,
    chain: i64,
    step: &crate::ParseStep,
    registered: &std::collections::HashMap<[u8; 32], crate::versions::RegisteredMessage<'_>>,
) -> Result<bool> {
    let Some(state) = step
        .checkpoint
        .as_ref()
        .and_then(|c| c.parser_state.as_ref())
    else {
        return Ok(true);
    };
    let Some(delta) = polylogue_model::parser_context::delta(state)? else {
        return Ok(true);
    };
    if delta.version != 1
        || (delta.base != step.start && delta.base != 0)
        || delta.through != step.parsed_upto
    {
        return Err(StoreError::Invalid(
            "parser context differs from its reading cursor".into(),
        ));
    }
    let through = sql_cursor(delta.through)?;
    if step.source_prefix_missing {
        return Ok(true);
    }
    let prior=tx.scalar_optional::<i64>("SELECT through FROM parser_context_state WHERE input_pk=?1 AND chain=?2 AND reading_version=?3",params![input,chain,step.reading_version])?.map(stored_cursor).transpose()?;
    if delta.base != 0 && prior != Some(delta.base) {
        return Ok(true);
    }
    let response_key = |entry: &polylogue_model::parser_context::Entry| {
        let (
            Key::Response { session, .. },
            Reference::Response {
                message, version, ..
            },
        ) = (&entry.key, &entry.value)
        else {
            return None;
        };
        registered
            .get(&version.0)?
            .response_key(&version.0, message, session)
    };
    let mut cleared = std::collections::BTreeSet::new();
    for namespace in &delta.cleared_states {
        if !cleared.insert(namespace) {
            return Err(StoreError::Invalid(
                "parser context repeats a cleared state namespace".into(),
            ));
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    for entry in &delta.entries {
        if !seen.insert(entry.key.clone()) {
            return Err(StoreError::Invalid(
                "parser context repeats a key in one checkpoint".into(),
            ));
        }
        if !matches!(
            (&entry.key, &entry.value),
            (Key::Response { .. }, Reference::Response { .. })
                | (Key::Alias { .. }, Reference::Alias { .. })
                | (Key::State { .. }, Reference::State { .. })
        ) {
            return Err(StoreError::Invalid(
                "parser context key and value have different kinds".into(),
            ));
        }
        if let Reference::Response {
            message,
            version,
            offset,
            end,
        } = &entry.value
        {
            if offset > end || *end > delta.through {
                return Err(StoreError::Invalid(
                    "response context range exceeds its reading".into(),
                ));
            }
            // Registration already validated and wrote this exact payload in
            // this transaction. Old or mismatched references keep the full
            // stored-message integrity and identity checks below.
            if response_key(entry).is_some() {
                continue;
            }
            let Some((body, _)) = crate::versions::read_message(tx, &version.0)? else {
                return Err(StoreError::Invalid(
                    "response context has no archived message version".into(),
                ));
            };
            if body.id != *message {
                return Err(StoreError::Invalid(
                    "response context has another message identity".into(),
                ));
            }
            if let Key::Response { session, .. } = &entry.key
                && session != &body.session.to_string()
            {
                return Err(StoreError::Invalid(
                    "response key has another source session".into(),
                ));
            }
        }
    }
    if delta.base == 0 {
        for table in [
            "parser_context_response",
            "parser_context_alias",
            "parser_context_value",
        ] {
            tx.prepare_cached(&format!(
                "DELETE FROM {table} WHERE input_pk=?1 AND chain=?2"
            ))?
            .execute(params![input, chain])?;
        }
    }
    for namespace in &delta.cleared_states {
        tx.prepare_cached(
            "DELETE FROM parser_context_value WHERE input_pk=?1 AND chain=?2 AND namespace=?3",
        )?
        .execute(params![input, chain, namespace])?;
    }
    // A key changed at this checkpoint replaces its earlier value in place.
    for entry in &delta.entries {
        match (&entry.key, &entry.value) {
            (
                Key::Response {
                    namespace,
                    session,
                    key,
                },
                Reference::Response {
                    version,
                    offset,
                    end,
                    ..
                },
            ) => {
                let version_pk = match response_key(entry) {
                    Some(pk) => pk,
                    None => crate::versions::message_key(tx, &version.0)?,
                };
                tx.prepare_cached("INSERT INTO parser_context_response VALUES(?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(input_pk,chain,namespace,session,key) DO UPDATE SET version_pk=excluded.version_pk,source_start=excluded.source_start,source_end=excluded.source_end")?
                    .execute(params![input, chain, namespace, session, key, version_pk, sql_cursor(*offset)?, sql_cursor(*end)?])?;
            }
            (Key::Alias { namespace, uuid }, Reference::Alias { target }) => {
                tx.prepare_cached("INSERT INTO parser_context_alias VALUES(?1,?2,?3,?4,?5) ON CONFLICT(input_pk,chain,namespace,uuid) DO UPDATE SET target=excluded.target")?
                    .execute(params![input, chain, namespace, uuid, target.as_ref().map(ToString::to_string)])?;
            }
            (Key::State { namespace, key }, Reference::State { value }) => {
                if value.is_null() {
                    tx.prepare_cached("DELETE FROM parser_context_value WHERE input_pk=?1 AND chain=?2 AND namespace=?3 AND key=?4")?
                        .execute(params![input,chain,namespace,key])?;
                } else {
                    tx.prepare_cached("INSERT INTO parser_context_value VALUES(?1,?2,?3,?4,?5) ON CONFLICT(input_pk,chain,namespace,key) DO UPDATE SET value=excluded.value")?
                        .execute(params![input,chain,namespace,key,crate::encoding::pack_json(value)?])?;
                }
            }
            _ => unreachable!("key and value kinds were checked above"),
        }
    }
    tx.prepare_cached("INSERT INTO parser_context_state VALUES(?1,?2,?3,?4) ON CONFLICT(input_pk,chain) DO UPDATE SET reading_version=excluded.reading_version,through=excluded.through")?
        .execute(params![input, chain, step.reading_version, through])?;
    Ok(true)
}

/// Record aliases from the source survive corrected response grouping. Authored
/// targets keep their original reference; only lookup follows the stored alias.
pub(crate) fn resolved_message_sql(
    id: &polylogue_model::MessageId,
    bind: &mut dyn FnMut(String) -> String,
) -> String {
    let original = bind(id.to_string());
    if id.origin().as_str() != "claude-code" {
        return original;
    }
    let uuid = bind(id.native().as_str().to_owned());
    format!(
        "CASE WHEN EXISTS(SELECT 1 FROM message direct JOIN message_source ds ON ds.message_pk=direct.pk WHERE direct.id={original}) THEN {original} ELSE pl_source_alias((SELECT json_group_array(DISTINCT pc.target) FROM parser_context_alias pc JOIN parser_context_state cs ON cs.input_pk=pc.input_pk AND cs.chain=pc.chain JOIN parse_state ps ON ps.input_pk=cs.input_pk AND ps.chain=cs.chain AND ps.reading_version=cs.reading_version WHERE pc.uuid={uuid}),{original}) END"
    )
}
pub(crate) fn resolved_message(
    conn: &Connection,
    id: &polylogue_model::MessageId,
) -> Result<polylogue_model::MessageId> {
    if id.origin().as_str() != "claude-code" {
        return Ok(id.clone());
    }
    let mut params = Vec::new();
    let expression = resolved_message_sql(id, &mut |value| {
        params.push(value);
        format!("?{}", params.len())
    });
    let value: String = conn.scalar(
        &format!("SELECT {expression}"),
        rusqlite::params_from_iter(params),
    )?;
    polylogue_model::MessageId::parse(&value).map_err(crate::StoreError::invalid)
}

fn sql_cursor(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        StoreError::Invalid("parser context cursor exceeds SQLite's integer range".into())
    })
}

fn stored_cursor(value: i64) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| crate::StoreError::stored(0, "parser context cursor is negative"))
}

#[cfg(test)]
mod tests;
