//! Which session an ID names, decided from the readings present each time
//! sessions are composed (docs/sessions.md, sessions). A route that knows a
//! session's own ID records the IDs its copies carry when they have no
//! native ID; an ID exactly one session records names that session, and an
//! ID no session or several sessions record names itself. A copy's rows are
//! then shown in the session, its content-named messages, call IDs and
//! events taking the session's ID in their names.
use crate::Result;
use polylogue_model::{Event, MessageId, SessionId};
use rusqlite::Connection;
use std::collections::HashMap;

pub(crate) struct Identities<'c> {
    conn: &'c Connection,
    shown: HashMap<String, String>,
    copies: HashMap<String, Vec<String>>,
}

impl<'c> Identities<'c> {
    pub(crate) fn new(conn: &'c Connection) -> Self {
        Self {
            conn,
            shown: HashMap::new(),
            copies: HashMap::new(),
        }
    }

    /// The session a stated session ID is shown as.
    pub(crate) fn session(&mut self, stated: &str) -> Result<String> {
        if let Some(shown) = self.shown.get(stated) {
            return Ok(shown.clone());
        }
        let recorded = self
            .conn
            .prepare_cached(
                "SELECT DISTINCT session_id FROM reading_session_alias WHERE alias=?1 AND session_id<>?1 LIMIT 2",
            )?
            .query_map([stated], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let shown = match <[String; 1]>::try_from(recorded) {
            Ok([session]) => session,
            Err(_) => stated.to_owned(),
        };
        self.shown.insert(stated.to_owned(), shown.clone());
        Ok(shown)
    }

    /// The copies shown as `session`: the IDs it records that no other
    /// session records.
    pub(crate) fn copies(&mut self, session: &str) -> Result<Vec<String>> {
        if let Some(copies) = self.copies.get(session) {
            return Ok(copies.clone());
        }
        let recorded = self
            .conn
            .prepare_cached(
                "SELECT DISTINCT alias FROM reading_session_alias WHERE session_id=?1 AND alias<>?1 ORDER BY alias",
            )?
            .query_map([session], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut copies = Vec::new();
        for copy in recorded {
            if self.session(&copy)? == session {
                copies.push(copy);
            }
        }
        self.copies.insert(session.to_owned(), copies.clone());
        Ok(copies)
    }

    /// Every stated session ID shown as `session`.
    pub(crate) fn stated(&mut self, session: &str) -> Result<Vec<String>> {
        let mut stated = self.copies(session)?;
        if self.session(session)? == session {
            stated.insert(0, session.to_owned());
        }
        Ok(stated)
    }

    /// The ID a stated message ID is shown with.
    pub(crate) fn message(&mut self, stated: &str) -> Result<String> {
        if let Some(copy) = message_session(stated) {
            let session = self.session(&copy)?;
            if session != copy
                && let Some(renamed) = renamed(stated, &copy, &session)
            {
                return Ok(renamed);
            }
        }
        Ok(stated.to_owned())
    }

    /// The stated message IDs shown as `shown`.
    pub(crate) fn stated_messages(&mut self, shown: &str) -> Result<Vec<String>> {
        let mut stated = vec![shown.to_owned()];
        if let Some(session) = message_session(shown) {
            for copy in self.copies(&session)? {
                if let Some(id) = renamed(shown, &session, &copy) {
                    stated.push(id);
                }
            }
        }
        Ok(stated)
    }

    /// The call ID a block is shown with: a call ID a copy's message names
    /// after itself (`<message>:code1`) takes its message's ID in the
    /// session, as the message does.
    pub(crate) fn call(&mut self, origin: &str, call: &str) -> Result<String> {
        let id = format!("{origin}:{call}");
        let shown = self.message(&id)?;
        Ok(if shown == id {
            call.to_owned()
        } else {
            shown
                .strip_prefix(origin)
                .and_then(|rest| rest.strip_prefix(':'))
                .unwrap_or(call)
                .to_owned()
        })
    }

    /// An event as shown: in the session its stated session is shown as,
    /// following the message as shown, and with the ID it has there when it
    /// is named by its content. `id` is its stated ID.
    pub(crate) fn event(&mut self, event: &Event, id: &str) -> Result<(Option<Event>, String)> {
        let Some(session) = &event.session else {
            return Ok((None, id.to_owned()));
        };
        let stated = session.to_string();
        let shown = self.session(&stated)?;
        let after = event
            .after
            .as_ref()
            .map(|after| self.message(&after.to_string()))
            .transpose()?;
        if shown == stated && after == event.after.as_ref().map(ToString::to_string) {
            return Ok((None, id.to_owned()));
        }
        let invalid = |e: polylogue_model::IdError| crate::StoreError::Invalid(e.to_string());
        let mut moved = event.clone();
        moved.session = Some(SessionId::parse(&shown).map_err(invalid)?);
        moved.after = after
            .map(|after| MessageId::parse(&after).map_err(invalid))
            .transpose()?;
        let id = if crate::encoding::event_content_id(event)? == id {
            crate::encoding::event_content_id(&moved)?
        } else {
            id.to_owned()
        };
        Ok((Some(moved), id))
    }
}

/// The session a content-derived message ID names: `origin:<session>~…`,
/// or `origin:<length>:<session>#…` for a native ID scoped to its session.
pub(crate) fn message_session(id: &str) -> Option<String> {
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
pub(crate) fn renamed(id: &str, from: &str, to: &str) -> Option<String> {
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
