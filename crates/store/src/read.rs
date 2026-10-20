//! Read connections. The daemon keeps a pool; every read in one request
//! sees one consistent snapshot of `index.db` with `archive.db` attached.

use crate::Result;
use crate::archive::{Archive, open_read};
use crate::blobs::FileFacts;
use crate::writer::{restore_text, unpack_block, unpack_text};
use polylogue_model::{
    Block, Message, MessageId, PromptKind, Role, Session, SessionId, Timestamp, Usage,
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;

/// Read-only connections to one archive, reused across requests.
pub struct ReadPool {
    archive: Archive,
    idle: Mutex<Vec<Connection>>,
}

/// One consistent read: a read transaction on `index.db` with `archive.db`
/// attached as `a`. Returned to the pool when dropped.
pub struct Reader<'a> {
    pool: &'a ReadPool,
    conn: Option<Connection>,
}

/// What the loop knows about a path from its last look, for comparing
/// with the file now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownInput {
    pub path: String,
    pub input_id: String,
    pub format: String,
    /// The file's facts at the last observation; `None` once it was seen
    /// gone.
    pub facts: Option<FileFacts>,
    /// The current revision chain and how many of its bytes are stored.
    pub chain: Option<i64>,
    pub stored: u64,
    /// Offset after the last complete record parsed in that chain, and the
    /// parser version that parsed it.
    pub parsed_upto: u64,
    pub parser_version: Option<u32>,
}

/// Totals for `status`. Unknown is never shown as zero: a count that could
/// not be read is an error, not a 0.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Status {
    pub sessions: u64,
    pub messages: u64,
    pub tool_calls: u64,
    pub inputs: u64,
    pub revisions: u64,
    /// Bytes of input content stored (uncompressed).
    pub stored_bytes: u64,
    /// Input chains by parse status (`done`, `parsing`).
    pub parse_states: BTreeMap<String, u64>,
    /// The most frequent record kinds and fields parsers did not read.
    pub unrecognized: Vec<(String, u64)>,
    /// Texts stored but not yet in the search index: how far search lags
    /// the writer (architecture 5.5c).
    pub search_pending: u64,
}

impl ReadPool {
    pub fn open(archive: &Archive) -> Result<Self> {
        Ok(Self {
            archive: archive.clone(),
            idle: Mutex::new(Vec::new()),
        })
    }

    fn connect(&self) -> Result<Connection> {
        let conn = open_read(&self.archive.index_db())?;
        let uri = format!("file:{}?mode=ro", self.archive.archive_db().display());
        conn.execute("ATTACH DATABASE ?1 AS a", [uri])?;
        Ok(conn)
    }

    pub fn reader(&self) -> Result<Reader<'_>> {
        let conn = match self.idle.lock().map(|mut v| v.pop()) {
            Ok(Some(c)) => c,
            _ => self.connect()?,
        };
        conn.execute_batch("BEGIN")?;
        Ok(Reader {
            pool: self,
            conn: Some(conn),
        })
    }
}

impl Drop for Reader<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take()
            && conn.execute_batch("COMMIT").is_ok()
            && let Ok(mut idle) = self.pool.idle.lock()
        {
            idle.push(conn);
        }
    }
}

/// A parameter of read-only SQL, so callers outside the store need no
/// SQLite types.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlValue {
    Int(i64),
    Real(f64),
    Text(String),
    Null,
}

impl rusqlite::ToSql for SqlValue {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        use rusqlite::types::{ToSqlOutput, ValueRef};
        Ok(match self {
            SqlValue::Int(n) => ToSqlOutput::Borrowed(ValueRef::Integer(*n)),
            SqlValue::Real(f) => ToSqlOutput::Borrowed(ValueRef::Real(*f)),
            SqlValue::Text(s) => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
            SqlValue::Null => ToSqlOutput::Borrowed(ValueRef::Null),
        })
    }
}

pub(crate) fn parse_name<T: serde::de::DeserializeOwned>(s: &str) -> Option<T> {
    serde_json::from_value(serde_json::Value::String(s.to_owned())).ok()
}

pub(crate) fn ts(ms: Option<i64>) -> Option<Timestamp> {
    ms.and_then(Timestamp::from_unix_ms)
}

impl Reader<'_> {
    pub(crate) fn conn(&self) -> &Connection {
        self.conn
            .as_ref()
            .expect("reader holds a connection until dropped")
    }

    pub fn session(&self, id: &SessionId) -> Result<Option<Session>> {
        Ok(self
            .conn()
            .prepare_cached(&format!(
                "SELECT {} FROM session WHERE id = ?1",
                crate::session_row::COLUMNS
            ))?
            .query_row([id.to_string()], |r| {
                crate::session_row::read(id.clone(), r)
            })
            .optional()?)
    }

    /// The session's messages in order, the preferred version of each. With
    /// `logical`, messages it inherited come first: a fork's parent prefix up
    /// to the branch point, and messages its file copied from another
    /// session, which are stored once, in the session they came from.
    pub fn messages(&self, id: &SessionId, logical: bool) -> Result<Vec<Message>> {
        let Some(pk) = self.session_pk(id)? else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        if logical {
            // Inherited messages come in runs, one per ancestor session.
            let mut runs: Vec<(i64, SessionId, HashSet<i64>)> = Vec::new();
            for (session_pk, session, message_pk) in self.inherited_keys(pk, id)? {
                match runs.last_mut() {
                    Some((s, _, set)) if *s == session_pk => {
                        set.insert(message_pk);
                    }
                    _ => runs.push((session_pk, session, HashSet::from([message_pk]))),
                }
            }
            for (session_pk, session, set) in runs {
                out.extend(self.messages_where(session_pk, &session, |m| set.contains(&m))?);
            }
        }
        out.extend(self.own_messages(pk, id)?);
        Ok(out)
    }

    /// The message keys a session shows: what it inherits through its
    /// fork and resume links, in reading order, then its own. Each message
    /// once.
    pub fn logical_message_keys(&self, id: &SessionId) -> Result<Vec<i64>> {
        let Some(pk) = self.session_pk(id)? else {
            return Ok(Vec::new());
        };
        let mut keys: Vec<i64> = self
            .inherited_keys(pk, id)?
            .into_iter()
            .map(|(_, _, m)| m)
            .collect();
        keys.extend(self.own_keys(pk)?.into_iter().map(|(m, _)| m));
        Ok(keys)
    }

    fn session_pk(&self, id: &SessionId) -> Result<Option<i64>> {
        Ok(self
            .conn()
            .prepare_cached("SELECT pk FROM session WHERE id = ?1")?
            .query_row([id.to_string()], |r| r.get(0))
            .optional()?)
    }

    /// A session's own message keys and IDs in reading order.
    fn own_keys(&self, session_pk: i64) -> Result<Vec<(i64, String)>> {
        Ok(self
            .conn()
            .prepare_cached(
                "SELECT pk, id FROM message WHERE session_pk = ?1 ORDER BY order_input_pk, seq",
            )?
            .query_map([session_pk], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// What a session inherits: `(session key, session, message key)` in
    /// reading order. A fork shows its parent's view up to the branch
    /// point; a resume shows the messages its own file copied.
    fn inherited_keys(
        &self,
        child_pk: i64,
        child: &SessionId,
    ) -> Result<Vec<(i64, SessionId, i64)>> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        self.inherited(child_pk, child, &mut out, &mut seen, 0)?;
        Ok(out)
    }

    fn inherited(
        &self,
        child_pk: i64,
        child: &SessionId,
        out: &mut Vec<(i64, SessionId, i64)>,
        seen: &mut HashSet<i64>,
        depth: usize,
    ) -> Result<()> {
        if depth > 64 {
            return Ok(());
        }
        let links: Vec<(String, String, Option<String>)> = self
            .conn()
            .prepare_cached(
                "SELECT parent_id, how, branch_point FROM link
                 WHERE child_id = ?1 AND kind = 'continuation' AND how IN ('fork', 'resume')
                 ORDER BY pk",
            )?
            .query_map([child.to_string()], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        for (parent, kind, at) in links {
            let Ok(parent) = SessionId::parse(&parent) else {
                continue;
            };
            let Some(parent_pk) = self.session_pk(&parent)? else {
                continue;
            };
            let keys: Vec<i64> = if kind == "fork" {
                self.inherited(parent_pk, &parent, out, seen, depth + 1)?;
                let mut own = self.own_keys(parent_pk)?;
                if let Some(at) = at
                    && let Some(i) = own.iter().position(|(_, id)| *id == at)
                {
                    own.truncate(i + 1);
                }
                own.into_iter().map(|(m, _)| m).collect()
            } else {
                self.conn()
                    .prepare_cached(
                        "SELECT DISTINCT m.pk, m.order_input_pk, m.seq FROM message m
                         JOIN message_source s ON s.message_pk = m.pk
                         JOIN parsed p ON p.input_pk = s.input_pk AND p.session_pk = ?2
                         WHERE m.session_pk = ?1 ORDER BY m.order_input_pk, m.seq",
                    )?
                    .query_map(params![parent_pk, child_pk], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?
            };
            for m in keys {
                if seen.insert(m) {
                    out.push((parent_pk, parent.clone(), m));
                }
            }
        }
        Ok(())
    }

    fn own_messages(&self, session_pk: i64, session: &SessionId) -> Result<Vec<Message>> {
        self.messages_where(session_pk, session, |_| true)
    }

    fn messages_where(
        &self,
        session_pk: i64,
        session: &SessionId,
        keep: impl Fn(i64) -> bool,
    ) -> Result<Vec<Message>> {
        let mut blocks: HashMap<i64, Vec<Block>> = HashMap::new();
        {
            let mut stmt = self.conn().prepare_cached(
                "SELECT b.message_pk, b.body, t.body FROM block b
                 LEFT JOIN text t ON t.id = b.text_id
                 WHERE b.session_pk = ?1 ORDER BY b.message_pk, b.idx",
            )?;
            let rows = stmt.query_map([session_pk], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, Option<Vec<u8>>>(2)?,
                ))
            })?;
            for row in rows {
                let (pk, body, text) = row?;
                let mut block: Block = unpack_block(&body)?;
                if let Some(packed) = text {
                    restore_text(&mut block, unpack_text(&packed)?);
                }
                blocks.entry(pk).or_default().push(block);
            }
        }
        let mut stmt = self.conn().prepare_cached(
            "SELECT pk, id, role, prompt_kind, model, at, parent_id, input_tokens, cache_read_tokens,
                    cache_write_tokens, output_tokens, reasoning_tokens, service_tier, stop_reason,
                    duration_ms, attrs_json, cache_write_1h_tokens, speed
             FROM message WHERE session_pk = ?1 ORDER BY order_input_pk, seq",
        )?;
        let rows = stmt.query_map([session_pk], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<i64>>(5)?,
                r.get::<_, Option<String>>(6)?,
                [
                    r.get::<_, Option<i64>>(7)?,
                    r.get::<_, Option<i64>>(8)?,
                    r.get::<_, Option<i64>>(9)?,
                    r.get::<_, Option<i64>>(10)?,
                    r.get::<_, Option<i64>>(11)?,
                ],
                r.get::<_, Option<String>>(12)?,
                r.get::<_, Option<String>>(13)?,
                r.get::<_, Option<i64>>(14)?,
                r.get::<_, Option<String>>(15)?,
                (
                    r.get::<_, Option<i64>>(16)?,
                    r.get::<_, Option<String>>(17)?,
                ),
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (
                pk,
                id,
                role,
                kind,
                model,
                at,
                parent,
                tokens,
                tier,
                stop,
                duration,
                attrs,
                (w1h, speed),
            ) = row?;
            if !keep(pk) {
                continue;
            }
            let Ok(id) = MessageId::parse(&id) else {
                continue;
            };
            let n = |x: Option<i64>| x.map(|x| x as u64);
            let attrs: Option<serde_json::Value> =
                attrs.and_then(|a| serde_json::from_str(&a).ok());
            out.push(Message {
                id,
                session: session.clone(),
                parent: parent.and_then(|p| MessageId::parse(&p).ok()),
                role: parse_name(&role).unwrap_or(Role::User),
                at: ts(at),
                model,
                prompt_kind: kind.as_deref().and_then(parse_name::<PromptKind>),
                blocks: blocks.remove(&pk).unwrap_or_default(),
                usage: Usage {
                    input: n(tokens[0]),
                    cached_input: n(tokens[1]),
                    cache_write: n(tokens[2]),
                    output: n(tokens[3]),
                    reasoning: n(tokens[4]),
                    cache_write_1h: n(w1h),
                    service_tier: tier,
                    speed,
                },
                stop_reason: stop,
                reported_cost_usd: attrs
                    .as_ref()
                    .and_then(|a| a.get("reported_cost_usd"))
                    .and_then(serde_json::Value::as_number)
                    .cloned(),
                duration_ms: n(duration),
                production: attrs
                    .as_ref()
                    .and_then(|a| a.get("production"))
                    .and_then(|p| serde_json::from_value(p.clone()).ok())
                    .unwrap_or_default(),
            });
        }
        Ok(out)
    }

    pub fn status(&self) -> Result<Status> {
        let c = self.conn();
        let count =
            |sql: &str| -> Result<u64> { Ok(c.query_row(sql, [], |r| r.get::<_, i64>(0))? as u64) };
        let mut parse_states = BTreeMap::new();
        let mut stmt =
            c.prepare_cached("SELECT status, COUNT(*) FROM parse_state GROUP BY status")?;
        for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (s, n) = row?;
            parse_states.insert(s, n as u64);
        }
        let mut unrecognized = Vec::new();
        let mut stmt = c.prepare_cached(
            "SELECT what, SUM(count) AS n FROM unrecognized GROUP BY what ORDER BY n DESC LIMIT 20",
        )?;
        for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (w, n) = row?;
            unrecognized.push((w, n as u64));
        }
        Ok(Status {
            sessions: count("SELECT COUNT(*) FROM session")?,
            messages: count("SELECT COUNT(*) FROM message")?,
            tool_calls: count("SELECT COUNT(*) FROM tool_call")?,
            inputs: count("SELECT COUNT(*) FROM a.input")?,
            revisions: count("SELECT COUNT(*) FROM a.revision")?,
            stored_bytes: count("SELECT COALESCE(SUM(size), 0) FROM a.blob")?,
            parse_states,
            unrecognized,
            search_pending: count("SELECT COUNT(*) FROM search_pending")?,
        })
    }

    /// Every path this machine observed, with its last facts and how far
    /// its current chain is stored and parsed. The loop compares these with
    /// the files now.
    /// The stored bytes of an input's revision chain, for parsing them
    /// again without the file.
    pub fn stored_chain(
        &self,
        input: &str,
        chain: i64,
        length: u64,
    ) -> Result<crate::blobs::StoredChain> {
        crate::blobs::StoredChain::load(self.conn(), "a.", input, chain, length)
    }

    pub fn known_inputs(&self, machine: &str) -> Result<Vec<KnownInput>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT o.path, o.input, i.format, o.class, o.device, o.inode, o.size, o.mtime_ns,
                    o.ctime_ns,
                    (SELECT chain FROM a.revision r WHERE r.input = o.input ORDER BY number DESC LIMIT 1),
                    (SELECT length FROM a.revision r WHERE r.input = o.input ORDER BY number DESC LIMIT 1)
             FROM a.observation o JOIN a.input i ON i.id = o.input
             WHERE o.machine = ?1 AND o.rowid = (
                 SELECT MAX(rowid) FROM a.observation x WHERE x.machine = o.machine AND x.path = o.path)",
        )?;
        let rows = stmt.query_map([machine], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                [
                    r.get::<_, Option<i64>>(4)?,
                    r.get::<_, Option<i64>>(5)?,
                    r.get::<_, Option<i64>>(6)?,
                    r.get::<_, Option<i64>>(7)?,
                    r.get::<_, Option<i64>>(8)?,
                ],
                r.get::<_, Option<i64>>(9)?,
                r.get::<_, Option<i64>>(10)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (path, input_id, format, class, f, chain, length) = row?;
            let facts = match (class.as_str(), f) {
                ("deleted", _) => None,
                (_, [Some(d), Some(i), Some(s), Some(m), Some(c)]) => Some(FileFacts {
                    device: d as u64,
                    inode: i as u64,
                    size: s as u64,
                    mtime_ns: i128::from(m),
                    ctime_ns: i128::from(c),
                }),
                _ => None,
            };
            out.push(KnownInput {
                path,
                input_id,
                format,
                facts,
                chain,
                stored: length.unwrap_or(0) as u64,
                parsed_upto: 0,
                parser_version: None,
            });
        }
        let mut stmt = self.conn().prepare_cached(
            "SELECT p.parsed_upto, p.parser_version FROM parse_state p
             JOIN input_ref r ON r.pk = p.input_pk WHERE r.id = ?1 AND p.chain = ?2",
        )?;
        for k in &mut out {
            if let Some(chain) = k.chain
                && let Some((upto, version)) = stmt
                    .query_row(params![k.input_id, chain], |r| {
                        Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
                    })
                    .optional()?
            {
                k.parsed_upto = upto as u64;
                k.parser_version = Some(version as u32);
            }
        }
        Ok(out)
    }

    /// Whether `archive.db` holds an input with this id.
    pub fn has_input(&self, id: &str) -> Result<bool> {
        Ok(self
            .conn()
            .prepare_cached("SELECT 1 FROM a.input WHERE id = ?1")?
            .exists([id])?)
    }

    /// Runs read-only SQL and returns each row as a JSON object. Parameters
    /// bind by number (`?1`, `?2`, …); those past the highest number the
    /// statement uses are not bound, so one parameter list serves several
    /// statements of one compiled query.
    pub fn query(&self, sql: &str, params: &[SqlValue]) -> Result<Vec<serde_json::Value>> {
        crate::fault::check("read.query", sql)?;
        let mut stmt = self.conn().prepare_cached(sql)?;
        let names: Vec<String> = stmt
            .column_names()
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let wanted = stmt.parameter_count();
        for (i, p) in params.iter().take(wanted).enumerate() {
            stmt.raw_bind_parameter(i + 1, p)?;
        }
        let mut rows = stmt.raw_query();
        let mut out = vec![];
        while let Some(r) = rows.next()? {
            let mut obj = serde_json::Map::new();
            for (i, name) in names.iter().enumerate() {
                use rusqlite::types::ValueRef;
                let v = match r.get_ref(i)? {
                    ValueRef::Null => serde_json::Value::Null,
                    ValueRef::Integer(n) => n.into(),
                    ValueRef::Real(f) => serde_json::json!(f),
                    ValueRef::Text(t) => String::from_utf8_lossy(t).into_owned().into(),
                    ValueRef::Blob(b) => b
                        .iter()
                        .map(|x| format!("{x:02x}"))
                        .collect::<String>()
                        .into(),
                };
                obj.insert(name.clone(), v);
            }
            out.push(serde_json::Value::Object(obj));
        }
        Ok(out)
    }
}
