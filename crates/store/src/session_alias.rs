//! Copies of a session named by their content (docs/capture.md, AI Studio
//! through Drive). A route that knows the session's own ID records the name
//! its copies carry (`session_alias`); the index shows a copy whose name
//! exactly one session records as that session. Readings stay as parsed:
//! the writer resolves the copy's session and message keys to the session's
//! rows, and when the recording route arrives after its copies, it moves
//! their rows in the same transaction.
use crate::Result;
use polylogue_model::Session;
use rusqlite::{Connection, OptionalExtension, params};
use std::borrow::Cow;
use std::collections::BTreeMap;

/// Each alias that exactly one session records, with that session's row.
/// An alias two sessions record names neither.
pub(crate) const NAMED: &str = "SELECT alias, min(session_pk) AS session_pk FROM session_alias
    GROUP BY alias HAVING count(DISTINCT session_pk) = 1";

/// A copy's session whose recording route arrives in the batches being
/// resolved: its rows move to that route's session.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Absorption {
    pub copy: String,
    pub session: String,
    /// Each message of the copy's session, with the ID it has in the session.
    pub messages: Vec<(String, String)>,
}

/// The session identities a resolution uses: copies named by an alias, and
/// copies the batches being resolved join to their session.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Identities {
    /// Copy session ID to session ID, for aliases already stored.
    stored: BTreeMap<String, String>,
    absorbed: Vec<Absorption>,
}

impl Identities {
    /// The stored aliases, and the copies that sessions these batches state
    /// (`recorded`) newly name.
    pub(crate) fn load<'a>(
        index: &Connection,
        recorded: impl IntoIterator<Item = &'a Session>,
    ) -> Result<Self> {
        let stored = index
            .prepare_cached(&format!(
                "SELECT n.alias, s.id FROM ({NAMED}) n JOIN session s ON s.pk = n.session_pk"
            ))?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<BTreeMap<String, String>>>()?;
        let mut identities = Self {
            stored,
            absorbed: Vec::new(),
        };
        for session in recorded {
            let id = session.id.to_string();
            for alias in &session.aliases {
                let copy = alias.to_string();
                if identities.stored.contains_key(&copy)
                    || identities.absorbed.iter().any(|a| a.copy == copy)
                {
                    continue;
                }
                let other: bool = index.query_row(
                    "SELECT EXISTS(SELECT 1 FROM session_alias a JOIN session s ON s.pk=a.session_pk
                     WHERE a.alias=?1 AND s.id<>?2)",
                    params![copy, id],
                    |row| row.get(0),
                )?;
                let held: Option<i64> = index
                    .query_row("SELECT pk FROM session WHERE id=?1", [&copy], |row| {
                        row.get(0)
                    })
                    .optional()?;
                let Some(held) = held.filter(|_| !other) else {
                    continue;
                };
                let messages = index
                    .prepare_cached("SELECT id FROM message WHERE session_pk=?1 ORDER BY id")?
                    .query_map([held], |row| row.get::<_, String>(0))?
                    .map(|old| {
                        let old = old?;
                        let new = renamed(&old, &copy, &id).unwrap_or_else(|| old.clone());
                        Ok((old, new))
                    })
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                identities.absorbed.push(Absorption {
                    copy,
                    session: id.clone(),
                    messages,
                });
            }
        }
        Ok(identities)
    }

    pub(crate) fn absorbed(&self) -> &[Absorption] {
        &self.absorbed
    }

    /// The session ID a key of `id` resolves to.
    pub(crate) fn session<'a>(&'a self, id: &'a str) -> &'a str {
        self.stored.get(id).map_or(id, String::as_str)
    }

    /// The message ID a key of `id` resolves to: a copy's message takes the
    /// ID it has in the session its copy name is an alias of.
    pub(crate) fn message<'a>(&self, id: &'a str) -> Cow<'a, str> {
        match self.lookup(id, |copy| self.stored.get(copy).map(String::as_str)) {
            Some(new) => Cow::Owned(new),
            None => Cow::Borrowed(id),
        }
    }

    /// The session ID a row is written with, including copies these batches
    /// join to their session.
    pub(crate) fn shown_session<'a>(&'a self, id: &'a str) -> &'a str {
        self.absorbed
            .iter()
            .find(|a| a.copy == id)
            .map_or_else(|| self.session(id), |a| a.session.as_str())
    }

    /// The message ID a row is written with.
    pub(crate) fn shown_message<'a>(&self, id: &'a str) -> Cow<'a, str> {
        let found = self.lookup(id, |copy| {
            self.absorbed
                .iter()
                .find(|a| a.copy == copy)
                .map(|a| a.session.as_str())
                .or_else(|| self.stored.get(copy).map(String::as_str))
        });
        match found {
            Some(new) => Cow::Owned(new),
            None => Cow::Borrowed(id),
        }
    }

    fn lookup<'s>(&self, id: &str, target: impl Fn(&str) -> Option<&'s str>) -> Option<String> {
        if self.stored.is_empty() && self.absorbed.is_empty() {
            return None;
        }
        let copy = message_session(id)?;
        renamed(id, &copy, target(&copy)?)
    }
}

/// The session a content-derived message ID names: `origin:<session>~…`,
/// or `origin:<length>:<session>#…` for a native ID scoped to its session.
fn message_session(id: &str) -> Option<String> {
    let (origin, native) = id.split_once(':')?;
    if let Some((session, _)) = native.split_once('~') {
        return Some(format!("{origin}:{session}"));
    }
    let (length, rest) = native.split_once(':')?;
    let length: usize = length.parse().ok()?;
    let session = rest.get(..length)?;
    rest[length..]
        .starts_with('#')
        .then(|| format!("{origin}:{session}"))
}

/// `id` with its session part `from` replaced by `to`, when it has one.
fn renamed(id: &str, from: &str, to: &str) -> Option<String> {
    let (origin, native) = id.split_once(':')?;
    let from = from.strip_prefix(origin)?.strip_prefix(':')?;
    let to = to.strip_prefix(origin)?.strip_prefix(':')?;
    if let Some(rest) = native.strip_prefix(from).and_then(|r| r.strip_prefix('~')) {
        return Some(format!("{origin}:{to}~{rest}"));
    }
    let scoped = format!("{}:{from}#", from.len());
    native
        .strip_prefix(&scoped)
        .map(|rest| format!("{origin}:{}:{to}#{rest}", to.len()))
}

#[cfg(test)]
mod tests;
