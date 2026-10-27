//! Context and briefings. This module is the one scheduler for context
//! Polylogue puts in front of a model: every brief, takeover, recall and
//! piece of advice before a tool runs is prepared here, and the daemon
//! records each one it hands out as a delivery with its exact text.
//!
//! A session-start brief brings an agent up to speed on its directory:
//! - notes the operator wrote, or agents wrote and the operator accepted:
//!   lessons, handoffs, decisions, blockers and goals on this directory,
//!   its repository, or the session itself;
//! - other agents working here now;
//! - Beads claimed here in the last week that the claiming session has not
//!   closed or released;
//! - what the recent sessions here did: commits, pull requests, files
//!   changed, checks, Beads, plan items left open, the last answer;
//! - checks that failed here with no later pass: `xtask` check and test
//!   runs (what AgentCTL's jobs run) in this checkout, from devkit's run
//!   history, and test, build and lint commands sessions ran themselves;
//! - signal notices of the last day;
//! - on resume, only what changed since the session's last brief; after
//!   compaction, every prompt the operator typed in the session, verbatim.
//!
//! A brief fits its budget (`briefing.budget`, in tokens of about four
//! bytes; [`DEFAULT_BUDGET`]). Sections fill in the order above and each
//! names what did not fit and the command that shows it. The operator's
//! prompts after compaction are never cut.

mod advice;

use crate::daemon::query::{NearFn, notes};
use polylogue_model::DeliveryEvent;
use polylogue_model::{Note, NoteKind, SessionId, Target, Timestamp};
use polylogue_store::search;
use polylogue_store::{LeftOut, Reader, StoreError};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Recent sessions a session-start brief reports on.
const RECENT_SESSIONS: u64 = 5;
/// Hits a recall returns per page.
const RECALL_PAGE: u64 = 20;
/// A brief's budget in tokens when `briefing.budget` is unset.
pub const DEFAULT_BUDGET: u64 = 2_000;
/// Bytes kept for the lines that name what did not fit.
const POINTER_RESERVE: usize = 600;
/// How far back failing checks, Bead claims and notices are reported.
const FAILING_WINDOW_MS: i64 = 7 * 24 * 60 * 60 * 1000;
const NOTICE_WINDOW_MS: i64 = 24 * 60 * 60 * 1000;

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
    /// Past work relevant to a text, with a summary of how it was found.
    /// `cursor` continues from a previous page's `next`.
    Recall {
        text: String,
        #[serde(default)]
        cwd: Option<PathBuf>,
        #[serde(default)]
        for_session: Option<SessionId>,
        #[serde(default)]
        cursor: Option<String>,
    },
    /// A command a session is about to run (a harness's pre-tool hook).
    ToolUse {
        cwd: PathBuf,
        #[serde(default)]
        session: Option<SessionId>,
        command: String,
    },
}

impl Ask {
    pub fn recipient(&self) -> Option<&SessionId> {
        match self {
            Self::SessionStart { session, .. } | Self::ToolUse { session, .. } => session.as_ref(),
            Self::Takeover { for_session, .. } | Self::Recall { for_session, .. } => {
                for_session.as_ref()
            }
        }
    }

    pub fn cwd(&self) -> Option<&Path> {
        match self {
            Self::SessionStart { cwd, .. } | Self::ToolUse { cwd, .. } => Some(cwd),
            Self::Recall { cwd, .. } => cwd.as_deref(),
            Self::Takeover { .. } => None,
        }
    }
}

/// Prepared context, not yet delivered.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Prepared {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advice_id: Option<String>,
    pub event: &'static str,
    pub text: String,
    /// References included, in order.
    pub refs: Vec<String>,
    pub left_out: Vec<LeftOut>,
    /// Recall only: how it searched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_summary: Option<SearchSummary>,
    /// Nothing worth sending: the caller sends nothing.
    pub empty: bool,
    /// Recall only: the cursor of the next page, when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
}

/// How a recall searched, so a reader can judge what it may have missed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchSummary {
    pub words: Vec<String>,
    /// What was searched: `prose` and `output` (full text, every word),
    /// and `semantic` (similarity to the whole text) when vectors exist.
    pub indexes: Vec<&'static str>,
    /// Hits on this page and the sessions they are in.
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
/// reader of the transcript can find the delivery record. `near` ranks a
/// recall by similarity when vectors exist.
pub fn prepare(
    reader: &Reader<'_>,
    ask: &Ask,
    delivery_id: &str,
    now: Timestamp,
    near: Option<&NearFn<'_>>,
) -> Result<Prepared, StoreError> {
    match ask {
        Ask::SessionStart {
            cwd,
            session,
            source,
        } => session_start(reader, cwd, session.as_ref(), *source, delivery_id, now),
        Ask::Takeover { session, .. } => takeover(reader, session, delivery_id, now),
        Ask::Recall {
            text, cwd, cursor, ..
        } => recall(
            reader,
            text,
            cwd.as_deref(),
            cursor.as_deref(),
            delivery_id,
            now,
            near,
        ),
        Ask::ToolUse {
            cwd,
            command,
            session,
            ..
        } => advice::advice(reader, cwd, command, session.as_ref(), now),
    }
}

/// Notes a brief carries: the operator's, and agents' the operator
/// accepted.
fn operator_notes(
    reader: &Reader<'_>,
    targets: Vec<Target>,
    after_seq: Option<i64>,
    now: Timestamp,
) -> Result<Vec<Note>, StoreError> {
    let kinds = [
        NoteKind::Lesson,
        NoteKind::Handoff,
        NoteKind::Decision,
        NoteKind::Blocker,
        NoteKind::Goal,
    ]
    .map(|k| format!("kind:{}", k.as_str()))
    .join(" OR ");
    let mut conditions = format!(
        "{} ({kinds}) (agent:false OR judged:accepted)",
        notes::about(&targets)
    );
    if let Some(after) = after_seq {
        conditions.push_str(&format!(" write_counter:>{after}"));
    }
    let notes = notes::list(reader, &conditions, now)?;
    Ok(notes.into_iter().filter(|n| !n.expired_at(now)).collect())
}

/// Agent-written notes here the operator has not judged yet.
fn unjudged_agent_notes(
    reader: &Reader<'_>,
    targets: &[Target],
    now: Timestamp,
) -> Result<u64, StoreError> {
    notes::count(
        reader,
        &format!("{} agent:true judged:unknown", notes::about(targets)),
        now,
    )
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

fn note_line(n: &Note) -> String {
    let label = n.key.as_deref().unwrap_or(n.kind.as_str());
    let targets: Vec<String> = n.targets.iter().map(ToString::to_string).collect();
    let by = if n.author.is_agent() {
        format!(", by {} (accepted)", n.author)
    } else {
        String::new()
    };
    format!(
        "- [{label}] {} ({}, {}{by})  note:{}",
        n.value.as_deref().unwrap_or("").trim(),
        targets.join(", "),
        day(Some(n.created_at)),
        n.id
    )
}

fn write_notes(out: &mut String, refs: &mut Vec<String>, notes: &[Note]) {
    for n in notes {
        let _ = writeln!(out, "{}", note_line(n));
        refs.push(format!("note:{}", n.id));
    }
}

/// One line (or a few) of a brief section, with what it cites.
struct Item {
    text: String,
    refs: Vec<String>,
}

/// A brief being filled within its budget.
struct Fitter {
    text: String,
    refs: Vec<String>,
    left_out: Vec<LeftOut>,
    /// Bytes the sections may take.
    room: usize,
    /// Lines naming what did not fit, written last from the reserve.
    pointers: Vec<String>,
    budget: u64,
}

impl Fitter {
    fn new(budget: u64, header: String) -> Self {
        let bytes = usize::try_from(budget.saturating_mul(4)).unwrap_or(usize::MAX);
        Self {
            room: bytes.saturating_sub(POINTER_RESERVE + header.len()),
            text: header,
            refs: Vec::new(),
            left_out: Vec::new(),
            pointers: Vec::new(),
            budget,
        }
    }

    /// Adds a section's items while they fit. Items that do not are
    /// counted as `what` left out, with `pull`, the command that shows
    /// them.
    fn section(&mut self, heading: &str, items: Vec<Item>, what: &str, pull: &str) {
        if items.is_empty() {
            return;
        }
        let heading = format!("\n{heading}\n");
        let mut body = String::new();
        let mut cut = 0;
        for item in items {
            let line = format!("{}\n", item.text.trim_end());
            if cut == 0 && heading.len() + body.len() + line.len() <= self.room {
                body.push_str(&line);
                self.refs.extend(item.refs);
            } else {
                cut += 1;
            }
        }
        if !body.is_empty() {
            self.room -= heading.len() + body.len();
            self.text.push_str(&heading);
            self.text.push_str(&body);
        }
        if cut > 0 {
            self.left_out.push(LeftOut {
                what: format!("{cut} {what}"),
                reason: format!("the brief's budget is {} tokens", self.budget),
            });
            self.pointers.push(format!("{cut} more {what}: `{pull}`"));
        }
    }

    /// Lines that need no budget accounting of their own (the pointers
    /// left over and the closing hint), as long as the reserve lasts.
    fn finish(mut self, closing: &str) -> (String, Vec<String>, Vec<LeftOut>) {
        let mut reserve = POINTER_RESERVE + self.room;
        let mut tail = Vec::new();
        if !self.pointers.is_empty() {
            tail.push("\nLeft out for the budget:".to_owned());
            tail.extend(self.pointers.iter().map(|p| format!("- {p}")));
        }
        tail.push(closing.to_owned());
        for line in tail {
            if line.len() + 1 > reserve {
                break;
            }
            reserve -= line.len() + 1;
            self.text.push_str(&line);
            self.text.push('\n');
        }
        (self.text, self.refs, self.left_out)
    }
}

/// The budget in tokens: `briefing.budget`, else [`DEFAULT_BUDGET`].
fn budget(reader: &Reader<'_>) -> Result<u64, StoreError> {
    let machine = polylogue_store::user_writes::machine_id();
    match reader.setting("briefing.budget", &machine)? {
        None => Ok(DEFAULT_BUDGET),
        Some(v) => v
            .trim()
            .parse::<u64>()
            .ok()
            .filter(|b| *b > 0)
            .ok_or_else(|| {
                StoreError::Invalid(format!(
                    "briefing.budget is {v:?}; it is a number of tokens, such as 2000"
                ))
            }),
    }
}

/// Where the development tooling's check history is: the
/// `briefing.check_history` setting, else devkit's file under the home
/// directory.
fn check_history(reader: &Reader<'_>) -> Result<PathBuf, StoreError> {
    let machine = polylogue_store::user_writes::machine_id();
    Ok(match reader.setting("briefing.check_history", &machine)? {
        Some(path) => PathBuf::from(path),
        None => polylogue_store::check_runs::default_path(Path::new(
            &std::env::var_os("HOME").unwrap_or_default(),
        )),
    })
}

/// Sessions working under `dirs` now, other than `me`, through the query
/// `live` runs.
fn live_here(
    reader: &Reader<'_>,
    dirs: &[String],
    me: Option<&SessionId>,
    now: Timestamp,
) -> Result<Vec<Item>, StoreError> {
    let places: Vec<String> = dirs
        .iter()
        .map(|d| format!("cwd:{}", notes::quoted(d)))
        .collect();
    let text = format!(
        "sessions where ({}) (live:working OR live:waiting OR live:idle) | sort by live_rank, last_event_at desc",
        places.join(" OR ")
    );
    let fields = [
        "id".to_owned(),
        "title".to_owned(),
        "live".to_owned(),
        "todo".to_owned(),
        "last_event_at".to_owned(),
    ];
    let mut items = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let page = crate::daemon::query::run(
            reader,
            crate::daemon::query::Ask {
                text: &text,
                cursor: cursor.as_deref(),
                page_size: None,
                why: false,
                session: None,
                now: now.unix_ms(),
                day: None,
                embeddings: false,
                near: None,
                fields: &fields,
            },
        )
        .map_err(|f| StoreError::Invalid(format!("live sessions {text:?}: {}", f.message)))?;
        for row in page["rows"].as_array().into_iter().flatten() {
            let Some(id) = row["id"].as_str() else {
                continue;
            };
            if me.is_some_and(|m| m.to_string() == id) {
                continue;
            }
            let title = row["title"]
                .as_str()
                .map_or("untitled".into(), |t| one_line(t, 80));
            let mut line = format!(
                "- {id}  {title}: {}",
                row["live"].as_str().unwrap_or("unknown")
            );
            if let Some(todo) = row["todo"].as_str() {
                let _ = write!(line, ", doing \"{}\"", one_line(todo, 100));
            }
            if let Some(at) = row["last_event_at"].as_str() {
                let _ = write!(
                    line,
                    ", last event {}",
                    at.chars().take(16).collect::<String>().replace('T', " ")
                );
            }
            items.push(Item {
                text: line,
                refs: vec![format!("session:{id}")],
            });
        }
        match page["next"].as_str() {
            Some(next) => cursor = Some(next.to_owned()),
            None => return Ok(items),
        }
    }
}

/// What a recent session did, in a few lines.
fn session_item(
    reader: &Reader<'_>,
    s: &polylogue_store::NearbySession,
) -> Result<Item, StoreError> {
    let id = s.id.to_string();
    let mut text = format!(
        "- {}  {}  ({} messages)  {id}",
        day(s.last_activity),
        s.title
            .as_deref()
            .map_or_else(|| "untitled".into(), |t| one_line(t, 100)),
        s.messages,
    );
    let work = reader.session_work(&id)?.unwrap_or_default();
    let mut did = Vec::new();
    if !work.commits.is_empty() {
        did.push(format!(
            "{} commit{} ({})",
            work.commits.len(),
            if work.commits.len() == 1 { "" } else { "s" },
            work.commits
                .iter()
                .map(|c| c.chars().take(7).collect::<String>())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !work.pull_requests.is_empty() {
        did.push(format!("pull requests {}", work.pull_requests.join(", ")));
    }
    if !work.files_changed.is_empty() {
        let shown: Vec<&str> = work
            .files_changed
            .iter()
            .take(4)
            .map(|p| p.rsplit('/').next().unwrap_or(p))
            .collect();
        let more = work.files_changed.len().saturating_sub(shown.len());
        did.push(format!(
            "changed {} file{}: {}{}",
            work.files_changed.len(),
            if work.files_changed.len() == 1 {
                ""
            } else {
                "s"
            },
            shown.join(", "),
            if more > 0 {
                format!(" and {more} more")
            } else {
                String::new()
            }
        ));
    }
    if work.checks_passed + work.checks_failed > 0 {
        did.push(format!(
            "checks {} passed, {} failed",
            work.checks_passed, work.checks_failed
        ));
    }
    for (bead, state) in &work.beads {
        did.push(match state.as_str() {
            "open" => format!("claimed {bead}, still open"),
            "" => format!("worked on {bead}"),
            other => format!("{other} {bead}"),
        });
    }
    if !did.is_empty() {
        let _ = write!(text, "\n    did: {}", did.join("; "));
    }
    let mut open = Vec::new();
    if let Some(items) = reader.open_items(&id)? {
        if items.unanswered_prompt {
            open.push("the last prompt has no answer".to_owned());
        }
        if items.open_tool_call {
            open.push("a tool call never finished".to_owned());
        }
        if let Some(w) = &items.waiting_on {
            open.push(format!("waiting on {w}"));
        }
    }
    if !work.plan_open.is_empty() {
        open.push(format!(
            "plan {} of {} done, next: \"{}\"",
            work.plan_done,
            work.plan_done + work.plan_open.len() as u64,
            one_line(&work.plan_open[0], 100)
        ));
    }
    if !open.is_empty() {
        let _ = write!(text, "\n    open: {}", open.join("; "));
    }
    if let Some(answer) = &work.last_answer {
        let _ = write!(text, "\n    last answer: \"{}\"", one_line(answer, 300));
    }
    Ok(Item {
        text,
        refs: vec![format!("session:{id}")],
    })
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
    let here = cwd.display().to_string();
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
    let after = |t: Option<Timestamp>| match &previous {
        Some(p) => t.is_some_and(|t| t > p.at),
        None => true,
    };
    let mut recent = reader.sessions_under(
        &dirs,
        session.map(ToString::to_string).as_deref(),
        RECENT_SESSIONS + 1,
    )?;
    recent.retain(|s| after(s.last_activity));
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

    let mut header = format!("Polylogue brief for {here} (delivery {delivery_id})\n");
    if let Some(p) = &previous {
        let _ = writeln!(
            header,
            "Changes since your last brief at {}.",
            day(Some(p.at))
        );
    }
    // The operator's own words after compaction are never cut.
    let mut prompt_refs = Vec::new();
    if source == StartSource::Compact
        && let Some(s) = session
    {
        let prompts = reader.prompts(&s.to_string())?;
        if !prompts.is_empty() {
            let _ = writeln!(
                header,
                "\nEvery prompt the operator typed in this session, verbatim, oldest first:"
            );
            for p in &prompts {
                let _ = writeln!(header, "\n[{}] {}", day(p.at), p.text.trim_end());
                prompt_refs.push(format!("message:{}", p.message_id));
            }
        }
    }
    let mut fit = Fitter::new(budget(reader)?, header);
    fit.refs.extend(prompt_refs);

    fit.section(
        "Notes from the operator:",
        notes
            .iter()
            .map(|n| Item {
                text: note_line(n),
                refs: vec![format!("note:{}", n.id)],
            })
            .collect(),
        "operator notes",
        &format!("polylogue note list --cwd {here}"),
    );
    fit.section(
        "Other agents working here now:",
        live_here(reader, &dirs, session, now)?,
        "live sessions",
        "polylogue live",
    );
    let claims_since = previous
        .as_ref()
        .map_or(now.unix_ms() - FAILING_WINDOW_MS, |p| p.at.unix_ms());
    let beads: Vec<Item> = reader
        .open_beads(&dirs, claims_since)?
        .into_iter()
        .map(|b| Item {
            text: format!(
                "- {} claimed {} by {}, which has not closed or released it",
                b.id,
                day(b.since),
                b.session
            ),
            refs: vec![format!("session:{}", b.session)],
        })
        .collect();
    fit.section(
        "Beads claimed here this week and still open:",
        beads,
        "open Beads",
        "polylogue work kind:attempt state:open",
    );
    let mut sessions = Vec::new();
    for s in &recent {
        sessions.push(session_item(reader, s)?);
    }
    if more_sessions {
        fit.left_out.push(LeftOut {
            what: "older sessions in this directory".into(),
            reason: format!("a brief reports on the {RECENT_SESSIONS} most recent"),
        });
        fit.pointers
            .push(format!("older sessions here: `polylogue list cwd:{here}`"));
    }
    fit.section(
        "Recent sessions here, and what they did:",
        sessions,
        "recent sessions",
        &format!("polylogue list cwd:{here}"),
    );
    let since = previous
        .as_ref()
        .map_or(now.unix_ms() - FAILING_WINDOW_MS, |p| p.at.unix_ms());
    // Check runs the development tooling recorded for this checkout (what
    // AgentCTL's check and test jobs run) come first; checks a session ran
    // itself follow.
    let mut failing: Vec<Item> =
        polylogue_store::check_runs::failed_since(&check_history(reader)?, &dirs, since)?
            .into_iter()
            .map(|f| {
                let mut text = format!(
                    "- `{}` failed {} in {}",
                    f.check,
                    day(Timestamp::from_unix_ms(f.started_ms)),
                    f.checkout
                );
                if f.times > 1 {
                    let _ = write!(text, " ({} failures)", f.times);
                }
                if let Some(job) = &f.job {
                    let _ = write!(text, ", job {job}");
                }
                if let Some(why) = &f.failure {
                    let _ = write!(text, ": {}", one_line(why, 120));
                }
                let _ = write!(text, "; `xtask history failures {}`", f.run);
                Item {
                    text,
                    refs: vec![format!("check_run:{}", f.run)],
                }
            })
            .collect();
    failing.extend(
        reader
            .failing_commands(&dirs, since, None)?
            .into_iter()
            .map(|f| {
                let mut text = format!("- `{}`", f.command);
                if let Some(code) = f.exit_code {
                    let _ = write!(text, " exit {code}");
                }
                let _ = write!(text, ", last {} in {}", day(f.at), f.session);
                if f.times > 1 {
                    let _ = write!(text, " ({} failures)", f.times);
                }
                if let Some(latest) = &f.latest {
                    let _ = write!(text, ": {}", one_line(latest, 120));
                }
                Item {
                    text,
                    refs: vec![format!("session:{}", f.session)],
                }
            }),
    );
    fit.section(
        "Checks that failed here with no later pass:",
        failing,
        "failing checks",
        "polylogue work kind:verification state:error",
    );
    let notices: Vec<Item> = reader
        .notices_since(
            previous
                .as_ref()
                .map_or(now.unix_ms() - NOTICE_WINDOW_MS, |p| p.at.unix_ms()),
        )?
        .into_iter()
        .map(|(row, text)| Item {
            text: format!("- {}  {}", day(Some(row.at)), one_line(&text, 200)),
            refs: vec![format!("delivery:{}", row.id)],
        })
        .collect();
    fit.section("Signal notices:", notices, "notices", "polylogue notices");

    let agent_notes = unjudged_agent_notes(reader, &targets, now)?;
    if agent_notes > 0 {
        fit.left_out.push(LeftOut {
            what: format!("{agent_notes} agent-written notes"),
            reason: "an agent's note is included once the operator accepts it".into(),
        });
        fit.pointers.push(format!(
            "{agent_notes} notes by agents, not yet accepted: `polylogue note list --agents --cwd {here}`"
        ));
    }
    let empty = fit.refs.is_empty();
    let closing = if empty && previous.is_some() {
        "Nothing changed."
    } else {
        "Earlier work on a topic: `polylogue recall WORDS`."
    };
    let (text, refs, left_out) = fit.finish(closing);
    Ok(Prepared {
        advice_id: None,
        event: match source {
            StartSource::Startup | StartSource::Clear => DeliveryEvent::SessionStart,
            StartSource::Resume => DeliveryEvent::Resume,
            StartSource::Compact => DeliveryEvent::Compaction,
        }
        .as_str(),
        text,
        refs,
        left_out,
        search_summary: None,
        empty,
        next: None,
    })
}

fn takeover(
    reader: &Reader<'_>,
    session: &SessionId,
    delivery_id: &str,
    now: Timestamp,
) -> Result<Prepared, StoreError> {
    let id = session.to_string();
    let Some(s) = reader.session(session)? else {
        return Err(StoreError::NotFound(format!("session {session}")));
    };
    let prompts = reader.prompts(&id)?;
    let selected = reader.select_messages(session, true)?;
    let mut messages = Vec::new();
    selected.visit(|message| {
        messages.push(message);
        std::ops::ControlFlow::Continue(())
    })?;
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
    let notes = notes::list(
        reader,
        &format!(
            "{} agent:false",
            notes::about(&[Target::Session(session.clone())])
        ),
        now,
    )?;
    let open_items = reader.open_items(&id)?.unwrap_or_default();

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
    let _ = writeln!(text, "Messages available: {}", messages.len());
    let mut left_out = vec![LeftOut {
        what: "tool calls and their output".into(),
        reason: "a takeover carries prompts, notes and the last answer".into(),
    }];
    for issue in &selected.parents().issues {
        let _ = writeln!(text, "Parents incomplete: {issue}");
        left_out.push(LeftOut {
            what: "inherited session content".into(),
            reason: issue.to_string(),
        });
    }
    if open_items.any() {
        let mut open = Vec::new();
        if open_items.unanswered_prompt {
            open.push("the last prompt has no answer".to_owned());
        }
        if open_items.open_tool_call {
            open.push("a tool call never finished".to_owned());
        }
        if let Some(t) = &open_items.todo {
            open.push(format!("todo in progress: {t}"));
        }
        if let Some(w) = &open_items.waiting_on {
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
        advice_id: None,
        event: DeliveryEvent::Takeover.as_str(),
        text,
        refs,
        left_out,
        search_summary: None,
        empty: false,
        next: None,
    })
}

/// One message a recall found.
struct Hit {
    session: String,
    message: String,
    /// The lists that rank it: `prose`, `output`, `semantic`.
    found_in: Vec<&'static str>,
    snippet: Option<String>,
}

/// A recall's query: messages with every word in prose, every word in
/// tool output and, with `similarity`, every message ranked by similarity
/// to the whole text, ranked together by reciprocal rank fusion; with
/// `directory`, sessions there rank in one more list. Also the name of
/// each list, in `found_in`'s numbering.
fn recall_query(
    words: &[String],
    text: &str,
    similarity: bool,
    directory: Option<&str>,
) -> Option<(String, Vec<&'static str>)> {
    let quoted = |prefix: &str| {
        words
            .iter()
            .map(|w| format!("{prefix}{}", notes::quoted(w)))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let mut lists: Vec<(&'static str, String)> = Vec::new();
    if !words.is_empty() {
        lists.push(("prose", format!("messages where {}", quoted(""))));
        lists.push(("output", format!("messages where {}", quoted("output:"))));
    }
    if similarity {
        lists.push((
            "semantic",
            format!("messages where near:{}", notes::quoted(text)),
        ));
    }
    let ((_, first), rest) = lists.split_first()?;
    let mut query = first.clone();
    for (_, other) in rest {
        query.push_str(&format!(" | union ({other})"));
    }
    if let Some(dir) = directory {
        query.push_str(&format!(
            " | fuse (messages where cwd:{})",
            notes::quoted(dir)
        ));
    }
    Some((query, lists.iter().map(|(name, _)| *name).collect()))
}

#[allow(clippy::too_many_arguments)]
fn recall(
    reader: &Reader<'_>,
    text_in: &str,
    cwd: Option<&Path>,
    cursor: Option<&str>,
    delivery_id: &str,
    now: Timestamp,
    near: Option<&NearFn<'_>>,
) -> Result<Prepared, StoreError> {
    let words = search::words(text_in);
    let text = text_in.trim();
    let directory = cwd.and_then(|c| scope_dirs(c).pop());
    let mut similarity = near.is_some() && !text.is_empty();
    let mut similarity_failed = None;
    let found = loop {
        let Some((query, indexes)) = recall_query(&words, text, similarity, directory.as_deref())
        else {
            break None;
        };
        let answer = crate::daemon::query::run(
            reader,
            crate::daemon::query::Ask {
                text: &query,
                cursor,
                page_size: Some(RECALL_PAGE),
                why: false,
                session: None,
                now: now.unix_ms(),
                day: None,
                embeddings: similarity,
                near: if similarity { near } else { None },
                fields: &[],
            },
        );
        match answer {
            Ok(page) => break Some((page, indexes)),
            // The provider could not embed the text: the words still search.
            Err(f) if similarity && f.kind == "embed" => {
                similarity = false;
                similarity_failed = Some(f.message);
            }
            Err(f) => {
                return Err(StoreError::Invalid(format!(
                    "recall query {query:?}: {}",
                    f.message
                )));
            }
        }
    };
    let mut page: Vec<Hit> = Vec::new();
    let mut indexes = Vec::new();
    let mut next = None;
    if let Some((answer, names)) = found {
        let mut exhausted = false;
        for row in answer["rows"].as_array().into_iter().flatten() {
            // Without fusion one list ranks every row it ranks at all.
            let found_in: Vec<&'static str> = match row.get("found_in") {
                Some(lists) => lists
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|n| names.get(n.as_u64()? as usize - 1).copied())
                    .collect(),
                None if row["rank"].is_null() => vec![],
                None => vec![names[0]],
            };
            // Rows no list ranks (messages with no vector and no word)
            // come last: the hits have ended.
            if found_in.is_empty() {
                exhausted = true;
                break;
            }
            let (Some(session), Some(message)) = (row["session"].as_str(), row["id"].as_str())
            else {
                continue;
            };
            page.push(Hit {
                session: session.to_owned(),
                message: message.to_owned(),
                found_in,
                snippet: row["snippet"]["text"].as_str().map(str::to_owned),
            });
        }
        if !exhausted {
            next = answer["next"].as_str().map(str::to_owned);
        }
        indexes = names;
    }
    let mut sessions: Vec<String> = Vec::new();
    for h in &page {
        if !sessions.contains(&h.session) {
            sessions.push(h.session.clone());
        }
    }

    let mut out = String::new();
    let mut refs = Vec::new();
    let _ = writeln!(
        out,
        "Polylogue recall for {:?} (delivery {delivery_id})",
        text_in.trim()
    );
    if cursor.is_some() {
        let _ = writeln!(out, "A later page of hits.");
    }
    if let Some(reason) = &similarity_failed {
        let _ = writeln!(out, "Similarity could not run ({reason}); words only.");
    }
    if page.is_empty() {
        let _ = writeln!(
            out,
            "No stored message matches{}.",
            if words.is_empty() {
                String::new()
            } else {
                format!(" all of: {}", words.join(", "))
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
        for h in page.iter().filter(|h| &h.session == sid) {
            let snippet = match &h.snippet {
                Some(s) => s.clone(),
                None => reader
                    .message_text(&h.message)?
                    .map(|t| one_line(&t, 200))
                    .unwrap_or_default(),
            };
            let _ = writeln!(
                out,
                "  - ({}) {}  message:{}",
                h.found_in.join(", "),
                snippet,
                h.message
            );
            refs.push(format!("message:{}", h.message));
        }
        // Notes on the session: the operator's as they are, agents'
        // fenced and labelled as an agent's note, never as instructions.
        let notes = notes::list(reader, &notes::about(&[Target::Session(id.clone())]), now)?;
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
    let search_summary = SearchSummary {
        words,
        indexes,
        hits: page.len(),
        sessions: sessions.len(),
        more: next.is_some(),
    };
    let _ = writeln!(
        out,
        "\nHow this was found: {}, ranked together by reciprocal rank fusion{}; \
         {} hits in {} sessions{}.",
        search_summary
            .indexes
            .iter()
            .map(|i| match *i {
                "prose" => "every word in prose",
                "output" => "every word in tool output",
                _ => "similarity to the whole text",
            })
            .collect::<Vec<_>>()
            .join(", "),
        if directory.is_some() {
            ", with this directory's sessions ranked in one more list"
        } else {
            ""
        },
        search_summary.hits,
        search_summary.sessions,
        match &next {
            Some(c) => format!("; more: `polylogue recall --cursor {c} {}`", text_in.trim()),
            None => String::new(),
        }
    );
    Ok(Prepared {
        advice_id: None,
        event: DeliveryEvent::Recall.as_str(),
        text: out,
        refs,
        left_out: Vec::new(),
        search_summary: Some(search_summary),
        empty: false,
        next,
    })
}

#[cfg(test)]
mod tests;
