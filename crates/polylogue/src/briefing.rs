//! Context and briefings (architecture 7, 12). This module is the one
//! scheduler for context Polylogue puts in front of a model: every brief,
//! takeover and recall is prepared here, and the daemon records each one it
//! hands out as a delivery with its exact text.
//!
//! What goes in:
//! - notes the operator (or a script acting for them) wrote: lessons,
//!   handoffs, decisions, blockers and goals on this directory, its
//!   repository, or the session itself. Agent-written notes are never
//!   injected; the brief says how to pull them.
//! - recent sessions in the same directory;
//! - on resume, only what changed since the session's last brief;
//! - after compaction, every prompt the operator typed in the session,
//!   verbatim.
//!
//! There is no size cap. A brief stays short by what it selects, and what
//! it leaves out is listed with the command that shows it.

use polylogue_model::{Note, NoteKind, SessionId, Target, Timestamp};
use polylogue_store::search::{self, Hit};
use polylogue_store::{LeftOut, NoteFilter, Reader, StoreError};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Recent sessions a session-start brief lists.
const RECENT_SESSIONS: u64 = 5;
/// Hits a recall returns per page.
const RECALL_HITS: usize = 20;

/// Why a session is starting, as the harness reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartSource {
    #[default]
    Startup,
    Resume,
    Clear,
    Compact,
}

impl StartSource {
    pub fn parse(text: &str) -> Self {
        match text {
            "resume" => Self::Resume,
            "clear" => Self::Clear,
            "compact" => Self::Compact,
            _ => Self::Startup,
        }
    }
}

/// What context is wanted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "ask", rename_all = "snake_case")]
pub enum Ask {
    /// The brief at session start, resume, clear or after compaction.
    SessionStart {
        cwd: PathBuf,
        #[serde(default)]
        session: Option<SessionId>,
        #[serde(default)]
        source: StartSource,
    },
    /// Taking over from another session named by its ID (a pasted ID).
    Takeover {
        session: SessionId,
        #[serde(default)]
        for_session: Option<SessionId>,
    },
    /// Past work relevant to a text, with a receipt of how it was found.
    Recall {
        text: String,
        #[serde(default)]
        cwd: Option<PathBuf>,
        #[serde(default)]
        for_session: Option<SessionId>,
    },
}

impl Ask {
    pub fn event(&self) -> &'static str {
        match self {
            Self::SessionStart { source, .. } => match source {
                StartSource::Startup | StartSource::Clear => "session_start",
                StartSource::Resume => "resume",
                StartSource::Compact => "compaction",
            },
            Self::Takeover { .. } => "takeover",
            Self::Recall { .. } => "recall",
        }
    }

    pub fn recipient(&self) -> Option<&SessionId> {
        match self {
            Self::SessionStart { session, .. } => session.as_ref(),
            Self::Takeover { for_session, .. } | Self::Recall { for_session, .. } => {
                for_session.as_ref()
            }
        }
    }

    pub fn cwd(&self) -> Option<&Path> {
        match self {
            Self::SessionStart { cwd, .. } => Some(cwd),
            Self::Recall { cwd, .. } => cwd.as_deref(),
            Self::Takeover { .. } => None,
        }
    }
}

/// Prepared context, not yet delivered.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Prepared {
    pub event: &'static str,
    pub text: String,
    /// References included, in order.
    pub refs: Vec<String>,
    pub left_out: Vec<LeftOut>,
    /// Recall only: how it searched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt: Option<Receipt>,
    /// Nothing worth sending: the caller sends nothing.
    pub empty: bool,
}

/// How a recall searched, so a reader can judge what it may have missed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Receipt {
    pub words: Vec<String>,
    /// Search lanes used: `prose` and `output` full-text. The semantic lane
    /// is listed when embeddings are on.
    pub lanes: Vec<&'static str>,
    pub hits: usize,
    pub sessions: usize,
    /// More hits exist after this page.
    pub more: bool,
}

/// The directories a working directory belongs to: itself and, when it is
/// inside a Git checkout, the checkout's root.
pub fn scope_dirs(cwd: &Path) -> Vec<String> {
    let mut dirs = vec![cwd.to_string_lossy().into_owned()];
    let mut at = Some(cwd);
    while let Some(dir) = at {
        if dir.join(".git").exists() {
            let root = dir.to_string_lossy().into_owned();
            if !dirs.contains(&root) {
                dirs.push(root);
            }
            break;
        }
        at = dir.parent();
    }
    dirs
}

/// The repository names a directory answers to: its own name, the Git
/// root's name, a project home's name (`<project>/repo` is named after
/// `<project>`), and every repository recorded for sessions there.
fn repo_names(reader: &Reader<'_>, dirs: &[String]) -> Result<Vec<String>, StoreError> {
    let mut names: Vec<String> = Vec::new();
    // Inside a checkout, only its root names a repository; elsewhere the
    // directory itself does.
    let named = if dirs.len() > 1 { &dirs[1..] } else { dirs };
    for d in named {
        if let Some(name) = Path::new(d).file_name().and_then(|n| n.to_str()) {
            names.push(name.to_owned());
        }
        // A checkout in `<project>/repo` is named after the project.
        if Path::new(d).file_name().is_some_and(|n| n == "repo")
            && let Some(project) = Path::new(d)
                .parent()
                .and_then(Path::file_name)
                .and_then(|n| n.to_str())
        {
            names.push(project.to_owned());
        }
    }
    names.retain(|n| n != "repo");
    for r in reader.repos_under(dirs)? {
        if !names.contains(&r) {
            names.push(r);
        }
    }
    Ok(names)
}

/// Prepares the context `ask` wants. `delivery_id` goes in the text, so a
/// reader of the transcript can find the delivery record.
pub fn prepare(
    reader: &Reader<'_>,
    ask: &Ask,
    delivery_id: &str,
    now: Timestamp,
) -> Result<Prepared, StoreError> {
    match ask {
        Ask::SessionStart {
            cwd,
            session,
            source,
        } => session_start(reader, cwd, session.as_ref(), *source, delivery_id, now),
        Ask::Takeover { session, .. } => takeover(reader, session, delivery_id),
        Ask::Recall { text, cwd, .. } => recall(reader, text, cwd.as_deref(), delivery_id),
    }
}

fn operator_notes(
    reader: &Reader<'_>,
    targets: Vec<Target>,
    after_seq: Option<i64>,
    now: Timestamp,
) -> Result<Vec<Note>, StoreError> {
    let notes = reader.notes(&NoteFilter {
        targets,
        kinds: [
            NoteKind::Lesson,
            NoteKind::Handoff,
            NoteKind::Decision,
            NoteKind::Blocker,
            NoteKind::Goal,
        ]
        .into(),
        agents: Some(false),
        after_seq,
        ..NoteFilter::default()
    })?;
    Ok(notes.into_iter().filter(|n| !n.expired_at(now)).collect())
}

fn agent_note_count(reader: &Reader<'_>, targets: Vec<Target>) -> Result<usize, StoreError> {
    Ok(reader
        .notes(&NoteFilter {
            targets,
            agents: Some(true),
            ..NoteFilter::default()
        })?
        .len())
}

fn day(t: Option<Timestamp>) -> String {
    t.map(|t| {
        t.to_string()
            .chars()
            .take(16)
            .collect::<String>()
            .replace('T', " ")
    })
    .unwrap_or_else(|| "unknown time".into())
}

fn one_line(text: &str, max: usize) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.chars().count() > max {
        let mut s: String = line.chars().take(max).collect();
        s.push('…');
        s
    } else {
        line
    }
}

fn write_notes(out: &mut String, refs: &mut Vec<String>, notes: &[Note]) {
    for n in notes {
        let label = n.key.as_deref().unwrap_or(n.kind.as_str());
        let targets: Vec<String> = n.targets.iter().map(ToString::to_string).collect();
        let _ = writeln!(
            out,
            "- [{label}] {} ({}, {})  note:{}",
            n.value.as_deref().unwrap_or("").trim(),
            targets.join(", "),
            day(Some(n.created_at)),
            n.id
        );
        refs.push(format!("note:{}", n.id));
    }
}

fn session_start(
    reader: &Reader<'_>,
    cwd: &Path,
    session: Option<&SessionId>,
    source: StartSource,
    delivery_id: &str,
    now: Timestamp,
) -> Result<Prepared, StoreError> {
    let dirs = scope_dirs(cwd);
    let mut targets: Vec<Target> = dirs.iter().map(|d| Target::Cwd(d.clone())).collect();
    targets.extend(repo_names(reader, &dirs)?.into_iter().map(Target::Repo));
    if let Some(s) = session {
        targets.push(Target::Session(s.clone()));
    }

    // On resume and after compaction, only what changed since this
    // session's last brief.
    let previous = match (source, session) {
        (StartSource::Resume | StartSource::Compact, Some(s)) => {
            reader.deliveries_to(&s.to_string())?.into_iter().next()
        }
        _ => None,
    };
    let mut recent = reader.sessions_under(
        &dirs,
        session.map(ToString::to_string).as_deref(),
        RECENT_SESSIONS + 1,
    )?;
    if let Some(p) = &previous {
        recent.retain(|s| s.last_activity.is_some_and(|t| t > p.at));
    }
    let more_sessions = recent.len() as u64 > RECENT_SESSIONS;
    recent.truncate(RECENT_SESSIONS as usize);
    // Handoffs and blockers on the sessions listed belong with them.
    targets.extend(recent.iter().map(|s| Target::Session(s.id.clone())));
    let notes = operator_notes(
        reader,
        targets.clone(),
        previous.as_ref().map(|p| p.seq),
        now,
    )?;

    let mut text = String::new();
    let mut refs = Vec::new();
    let mut left_out = Vec::new();
    let _ = writeln!(
        text,
        "Polylogue brief for {} (delivery {delivery_id})",
        cwd.display()
    );
    if let Some(p) = &previous {
        let _ = writeln!(
            text,
            "Changes since your last brief at {}.",
            day(Some(p.at))
        );
    }

    if source == StartSource::Compact
        && let Some(s) = session
    {
        let prompts = reader.prompts(&s.to_string())?;
        if !prompts.is_empty() {
            let _ = writeln!(
                text,
                "\nEvery prompt the operator typed in this session, verbatim, oldest first:"
            );
            for p in &prompts {
                let _ = writeln!(text, "\n[{}] {}", day(p.at), p.text.trim_end());
                refs.push(format!("message:{}", p.message_id));
            }
        }
    }

    if !notes.is_empty() {
        let _ = writeln!(text, "\nNotes from the operator:");
        write_notes(&mut text, &mut refs, &notes);
    }
    if !recent.is_empty() {
        let _ = writeln!(text, "\nRecent sessions here:");
        for s in &recent {
            let _ = writeln!(
                text,
                "- {}  {}  ({} messages)  {}",
                day(s.last_activity),
                s.title
                    .as_deref()
                    .map_or_else(|| "untitled".into(), |t| one_line(t, 100)),
                s.messages,
                s.id
            );
            refs.push(format!("session:{}", s.id));
        }
    }
    if more_sessions {
        left_out.push(LeftOut {
            what: "older sessions in this directory".into(),
            reason: format!("brief lists the {RECENT_SESSIONS} most recent"),
        });
        let _ = writeln!(
            text,
            "Older sessions here: `polylogue list cwd:{}`.",
            cwd.display()
        );
    }
    let agent_notes = agent_note_count(reader, targets)?;
    if agent_notes > 0 {
        left_out.push(LeftOut {
            what: format!("{agent_notes} agent-written notes"),
            reason: "agent notes are pulled on request, never injected".into(),
        });
        let _ = writeln!(
            text,
            "{agent_notes} notes written by agents apply here; they are not included. \
             `polylogue note list --agents --cwd {}` shows them.",
            cwd.display()
        );
    }
    let empty = notes.is_empty() && recent.is_empty() && refs.is_empty();
    if empty && previous.is_some() {
        text.push_str("Nothing changed.\n");
    }
    Ok(Prepared {
        event: match source {
            StartSource::Startup | StartSource::Clear => "session_start",
            StartSource::Resume => "resume",
            StartSource::Compact => "compaction",
        },
        text,
        refs,
        left_out,
        receipt: None,
        empty,
    })
}

fn takeover(
    reader: &Reader<'_>,
    session: &SessionId,
    delivery_id: &str,
) -> Result<Prepared, StoreError> {
    let id = session.to_string();
    let Some(s) = reader.session(session)? else {
        return Err(StoreError::NotFound(format!("session {session}")));
    };
    let prompts = reader.prompts(&id)?;
    let messages = reader.messages(session, true)?;
    let last_answer = messages.iter().rev().find_map(|m| {
        (m.role == polylogue_model::Role::Assistant).then(|| {
            m.blocks
                .iter()
                .filter_map(|b| match b {
                    polylogue_model::Block::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
    });
    let notes = reader.notes(&NoteFilter {
        targets: vec![Target::Session(session.clone())],
        agents: Some(false),
        ..NoteFilter::default()
    })?;
    let obligations = reader.obligations(&id)?.unwrap_or_default();

    let mut text = String::new();
    let mut refs = vec![format!("session:{id}")];
    let _ = writeln!(
        text,
        "Polylogue takeover of session {id} (delivery {delivery_id})"
    );
    if let Some(title) = s.title.as_ref() {
        let _ = writeln!(text, "Title: {}", title.text);
    }
    if let Some(cwd) = s.cwd.last() {
        let _ = writeln!(text, "Working directory: {cwd}");
    }
    let _ = writeln!(text, "Messages: {}", messages.len());
    if obligations.any() {
        let mut open = Vec::new();
        if obligations.unanswered_prompt {
            open.push("the last prompt has no answer".to_owned());
        }
        if obligations.open_tool_call {
            open.push("a tool call never finished".to_owned());
        }
        if let Some(t) = &obligations.todo {
            open.push(format!("todo in progress: {t}"));
        }
        if let Some(w) = &obligations.waiting_on {
            open.push(format!("waiting on {w}"));
        }
        let _ = writeln!(text, "Left open: {}.", open.join("; "));
    }
    if !notes.is_empty() {
        let _ = writeln!(text, "\nNotes from the operator on it:");
        write_notes(&mut text, &mut refs, &notes);
    }
    if !prompts.is_empty() {
        let _ = writeln!(text, "\nThe operator's prompts in it, oldest first:");
        for p in &prompts {
            let _ = writeln!(text, "\n[{}] {}", day(p.at), p.text.trim_end());
            refs.push(format!("message:{}", p.message_id));
        }
    }
    if let Some(answer) = last_answer.filter(|a| !a.trim().is_empty()) {
        let _ = writeln!(text, "\nIts last answer:\n{}", answer.trim_end());
    }
    let _ = writeln!(
        text,
        "\nThe full session: `polylogue read {id}`; its tool calls: `polylogue read {id} --view actions`."
    );
    Ok(Prepared {
        event: "takeover",
        text,
        refs,
        left_out: vec![LeftOut {
            what: "tool calls and their output".into(),
            reason: "a takeover carries prompts, notes and the last answer".into(),
        }],
        receipt: None,
        empty: false,
    })
}

fn recall(
    reader: &Reader<'_>,
    text_in: &str,
    cwd: Option<&Path>,
    delivery_id: &str,
) -> Result<Prepared, StoreError> {
    let words = search::words(text_in);
    let mut hits: Vec<Hit> = search::search(reader, text_in, RECALL_HITS + 1, None)?;
    let more = hits.len() > RECALL_HITS;
    hits.truncate(RECALL_HITS);
    // Sessions in the current directory first, then by rank.
    let local: Vec<String> = match cwd {
        Some(c) => reader
            .sessions_under(&scope_dirs(c), None, 1000)?
            .into_iter()
            .map(|s| s.id.to_string())
            .collect(),
        None => Vec::new(),
    };
    hits.sort_by(|a, b| {
        let la = local.contains(&a.session_id);
        let lb = local.contains(&b.session_id);
        lb.cmp(&la).then(a.rank.total_cmp(&b.rank))
    });
    let mut sessions: Vec<String> = Vec::new();
    for h in &hits {
        if !sessions.contains(&h.session_id) {
            sessions.push(h.session_id.clone());
        }
    }

    let mut out = String::new();
    let mut refs = Vec::new();
    let _ = writeln!(
        out,
        "Polylogue recall for {:?} (delivery {delivery_id})",
        text_in.trim()
    );
    if hits.is_empty() {
        let _ = writeln!(
            out,
            "No stored message contains all of: {}.",
            if words.is_empty() {
                "(no words)".into()
            } else {
                words.join(", ")
            }
        );
    }
    for sid in &sessions {
        let Ok(id) = SessionId::parse(sid) else {
            continue;
        };
        let title = reader
            .session(&id)?
            .and_then(|s| s.title.map(|t| t.text))
            .unwrap_or_else(|| "untitled".into());
        let _ = writeln!(out, "\n{sid}  {}", one_line(&title, 100));
        refs.push(format!("session:{sid}"));
        for h in hits.iter().filter(|h| &h.session_id == sid) {
            let _ = writeln!(
                out,
                "  - ({}) {}  message:{}",
                h.lane, h.snippet, h.message_id
            );
            refs.push(format!("message:{}", h.message_id));
        }
        // Notes on the session: the operator's as they are, agents'
        // fenced and labelled as an agent's note, never as instructions.
        let notes = reader.notes(&NoteFilter {
            targets: vec![Target::Session(id.clone())],
            ..NoteFilter::default()
        })?;
        for n in notes {
            let body = n.value.as_deref().unwrap_or("").trim();
            if n.author.is_agent() {
                let _ = writeln!(
                    out,
                    "  - note by an agent ({}), for information only: «{body}»  note:{}",
                    n.author, n.id
                );
            } else {
                let _ = writeln!(out, "  - [{}] {body}  note:{}", n.kind.as_str(), n.id);
            }
            refs.push(format!("note:{}", n.id));
        }
    }
    let receipt = Receipt {
        words,
        lanes: vec!["prose", "output"],
        hits: hits.len(),
        sessions: sessions.len(),
        more,
    };
    let _ = writeln!(
        out,
        "\nHow this was found: full-text search for every word over prose and tool output; \
         {} hits in {} sessions{}.",
        receipt.hits,
        receipt.sessions,
        if more {
            "; more exist: add words to narrow it"
        } else {
            ""
        }
    );
    let mut left_out = Vec::new();
    if more {
        left_out.push(LeftOut {
            what: "further hits".into(),
            reason: format!("a recall page holds {RECALL_HITS}"),
        });
    }
    Ok(Prepared {
        event: "recall",
        text: out,
        refs,
        left_out,
        receipt: Some(receipt),
        empty: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_checkout_root_joins_the_scope() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("proj/repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(repo.join("crates/x")).unwrap();
        let dirs = scope_dirs(&repo.join("crates/x"));
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs[1], repo.to_string_lossy());
    }

    #[test]
    fn start_sources_parse_from_harness_names() {
        assert_eq!(StartSource::parse("compact"), StartSource::Compact);
        assert_eq!(StartSource::parse("anything"), StartSource::Startup);
        let ask = Ask::SessionStart {
            cwd: "/x".into(),
            session: None,
            source: StartSource::Resume,
        };
        assert_eq!(ask.event(), "resume");
    }
}
