//! Context and briefings. This module is the one scheduler for context
//! Polylogue puts in front of a model: every brief, takeover and recall is
//! prepared here, and the daemon records each one it hands out as a delivery
//! with its exact text.
//!
//! A session-start brief brings an agent up to speed on its directory:
//! - notes the operator wrote, or agents wrote and the operator accepted:
//!   lessons, handoffs, decisions, problems and goals on this directory,
//!   its repository, or the session itself;
//! - other agents working here now;
//! - Beads claimed here in the last week that the claiming session has not
//!   closed or released;
//! - what the recent sessions here did: commits, pull requests, files
//!   changed, Beads, what they left open, the last answer;
//! - signal notices of the last day;
//! - on resume, only what changed since the session's last brief; after
//!   compaction, every prompt the operator typed in the session, verbatim.
//!
//! A brief fits its request budget (in tokens of about four
//! bytes; [`DEFAULT_BUDGET`]). Sections fill in the order above and each
//! names what did not fit and the command that shows it. The operator's
//! prompts after compaction are never cut.
//!
//! A brief's reads are bounded by what it reports, not by how much work
//! happened under its directory: which sessions are under the directory,
//! working there now or holding Beads comes from the daemon's scope index
//! in memory, and each listed session costs one stored summary. Sections
//! run cheapest first within the request's deadline; a section that meets
//! the deadline is left out with reason `deadline`, and the brief answers
//! with what it has.

use crate::daemon::query::{NearFn, assertions};
use crate::daemon::scope::SharedScopeIndex;
use polylogue_model::DeliveryEvent;
use polylogue_model::user::LeftOut;
use polylogue_model::{
    Reference, SessionId, Timestamp,
    assertion::{Assertion, AssertionKind},
};
use polylogue_store::search;
use polylogue_store::{Reader, ScopeSession, SessionSummary, StoreError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Recent sessions a session-start brief reports on.
const RECENT_SESSIONS: u64 = 5;
/// Hits a recall returns per page.
const RECALL_PAGE: u64 = 20;
/// The default named-context budget in tokens.
pub const DEFAULT_BUDGET: u32 = 10_000;
const RESUME_BUDGET: u32 = 4_000;
/// Bytes kept for the lines that name what did not fit.
const POINTER_RESERVE: usize = 600;
/// How far back Bead claims are reported.
const CLAIM_WINDOW_MS: i64 = 7 * 24 * 60 * 60 * 1000;
/// How far back signal notices are reported.
const NOTICE_WINDOW_MS: i64 = 24 * 60 * 60 * 1000;

/// Why a session is starting, as the harness reports it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
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
    /// Taking over from another session named by its ID or a pasted session link.
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
}

impl Ask {
    pub fn default_budget_tokens(&self) -> u32 {
        match self {
            Self::SessionStart {
                source: StartSource::Resume,
                ..
            } => RESUME_BUDGET,
            _ => DEFAULT_BUDGET,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Prepared {
    pub event: String,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SearchSummary {
    pub words: Vec<String>,
    /// What was searched: `prose` and `output` (full text, every word),
    /// and `semantic` (similarity to the whole text) when vectors exist.
    pub indexes: Vec<String>,
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
/// `<project>`), and every repository recorded for sessions there
/// (`recorded`, from the scope index).
fn repo_names(dirs: &[String], recorded: Vec<String>) -> Vec<String> {
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
    for r in recorded {
        if !names.contains(&r) {
            names.push(r);
        }
    }
    names
}

/// The latest brief a session received: when, and the archive's write
/// counter then. Assertions written after it are the next brief's delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LastBrief {
    pub at: Timestamp,
    pub seq: i64,
}

impl LastBrief {
    /// The later of two.
    fn newest(a: Option<Self>, b: Option<Self>) -> Option<Self> {
        match (a, b) {
            (Some(a), Some(b)) => Some(if b.at > a.at { b } else { a }),
            (a, b) => a.or(b),
        }
    }
}

/// Prepares the context `ask` wants. `delivery_id` goes in the text, so a
/// reader of the transcript can find the delivery record. `scope` is the
/// daemon's scope index; `queued` is the latest brief handed to the ask's
/// recipient whose delivery record the writer has not committed yet. `near`
/// ranks a recall by similarity when vectors exist.
#[allow(clippy::too_many_arguments)]
pub fn prepare(
    reader: &Reader<'_>,
    scope: &SharedScopeIndex,
    queued: Option<LastBrief>,
    ask: &Ask,
    delivery_id: &str,
    context: &std::sync::Arc<crate::operation::Context>,
    ranking_identity: Option<&str>,
    near: Option<&NearFn<'_>>,
    budget_tokens: Option<u32>,
) -> Result<Prepared, StoreError> {
    let budget = u64::from(budget_tokens.unwrap_or_else(|| ask.default_budget_tokens()));
    match ask {
        Ask::SessionStart {
            cwd,
            session,
            source,
        } => session_start(
            reader,
            scope,
            queued,
            cwd,
            session.as_ref(),
            *source,
            budget,
            delivery_id,
            context,
        ),
        Ask::Takeover { session, .. } => takeover(reader, session, budget, delivery_id, context),
        Ask::Recall {
            text, cwd, cursor, ..
        } => recall(
            reader,
            text,
            cwd.as_deref(),
            cursor.as_deref(),
            budget,
            delivery_id,
            context,
            ranking_identity,
            near,
        ),
    }
}
/// Notes a brief carries: the operator's, and agents' the operator
/// accepted.
fn operator_notes(
    reader: &Reader<'_>,
    targets: Vec<Reference>,
    after_seq: Option<i64>,
    context: &std::sync::Arc<crate::operation::Context>,
) -> Result<Vec<Assertion>, StoreError> {
    let now: Timestamp = context.clock.now.into();
    let kinds = [
        AssertionKind::Lesson,
        AssertionKind::Handoff,
        AssertionKind::Decision,
        AssertionKind::Problem,
        AssertionKind::Goal,
    ]
    .map(|k| format!("kind:{}", k.as_str()))
    .join(" OR ");
    let mut conditions = format!(
        "{} ({kinds}) (author:operator OR judged:accepted)",
        assertions::about(&targets)
    );
    if let Some(after) = after_seq {
        conditions.push_str(&format!(" write_counter > {after}"));
    }
    let notes = assertions::list(reader, &conditions, context)?;
    Ok(notes
        .into_iter()
        .filter(|n| !n.fact.expired_at(now))
        .collect())
}

/// Agent-written notes here the operator has not judged yet.
fn unjudged_other_assertions(
    reader: &Reader<'_>,
    targets: &[Reference],
    context: &std::sync::Arc<crate::operation::Context>,
) -> Result<u64, StoreError> {
    assertions::count(
        reader,
        &format!(
            "{} NOT author:operator judged:unknown",
            assertions::about(targets)
        ),
        context,
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

fn note_line(n: &Assertion) -> String {
    let label = n.fact.key.as_deref().unwrap_or(n.fact.kind.as_str());
    let targets: Vec<String> = n.fact.about().map(ToString::to_string).collect();
    let by = if n.fact.author.is_agent() {
        format!(", declared author {}", n.fact.author)
    } else {
        String::new()
    };
    format!(
        "- [{label}] {} ({}, {}{by})  assertion:{}@{}",
        n.fact.text().unwrap_or("").trim(),
        targets.join(", "),
        day(Some(n.fact.created_at)),
        n.fact.root,
        n.fact.id
    )
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
    /// The line naming the sections the deadline left out, and the command
    /// that prints the whole brief.
    late: Option<String>,
    budget: u64,
    reserve: usize,
}

impl Fitter {
    fn new(budget: u64, header: String) -> Result<Self, StoreError> {
        let bytes = usize::try_from(budget.saturating_mul(4)).unwrap_or(usize::MAX);
        if header.len() > bytes {
            return Err(StoreError::Invalid(
                "context budget cannot fit the required header; increase budget_tokens".into(),
            ));
        }
        let available = bytes - header.len();
        let reserve = POINTER_RESERVE.min(available);
        Ok(Self {
            room: available - reserve,
            text: header,
            refs: Vec::new(),
            left_out: Vec::new(),
            pointers: Vec::new(),
            late: None,
            budget,
            reserve,
        })
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
        let mut reserve = self.reserve + self.room;
        let mut tail = Vec::new();
        if !self.pointers.is_empty() {
            tail.push("\nLeft out for the budget:".to_owned());
            tail.extend(self.pointers.iter().map(|p| format!("- {p}")));
        }
        tail.extend(self.late.take());
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

/// What the scope index says about a brief's directories, copied out of it
/// so that no archive read happens under its lock.
struct Here {
    repos: Vec<String>,
    /// Sessions there active after the last brief, newest first: one more
    /// than a brief reports, so it knows whether there are more.
    recent: Vec<ScopeSession>,
    live: Vec<(ScopeSession, &'static str)>,
    beads: Vec<polylogue_store::OpenBead>,
}

impl Here {
    fn of(
        index: &polylogue_store::ScopeIndex,
        dirs: &[String],
        me: Option<&str>,
        now: i64,
        after: Option<Timestamp>,
        claims_since: i64,
    ) -> Self {
        let recent = index
            .sessions_under(dirs, me)
            .into_iter()
            .filter(|s| after.is_none_or(|p| s.session.last_activity.is_some_and(|t| t > p)))
            .take(RECENT_SESSIONS as usize + 1)
            .cloned()
            .collect();
        Self {
            repos: index.repos_under(dirs),
            recent,
            live: index
                .live_under(dirs, me, now)
                .into_iter()
                .map(|live| (live.session.clone(), live.state))
                .collect(),
            beads: index.open_beads(dirs, claims_since),
        }
    }
}

/// The sections that ran into the brief's deadline. Each section that reads
/// the archive runs through [`Deadline::section`]; one that meets the
/// deadline is left out with reason `deadline`, and the brief answers with
/// the sections it has instead of nothing.
#[derive(Default)]
struct Deadline {
    late: Vec<&'static str>,
}

impl Deadline {
    fn section<T>(
        &mut self,
        reader: &Reader<'_>,
        what: &'static str,
        work: impl FnOnce() -> Result<T, StoreError>,
    ) -> Result<Option<T>, StoreError> {
        match reader.check().and_then(|()| work()) {
            Ok(value) => Ok(Some(value)),
            // A query's own failure wraps the deadline; the reader knows it.
            Err(error)
                if matches!(error, StoreError::ReadDeadlineExceeded)
                    || matches!(reader.check(), Err(StoreError::ReadDeadlineExceeded)) =>
            {
                self.late.push(what);
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
}

/// Other sessions working, waiting or idle under the brief's directories,
/// leaving out the ones a `hidden` note hides.
fn live_items(
    reader: &Reader<'_>,
    live: &[(ScopeSession, &'static str)],
    now: i64,
) -> Result<Vec<Item>, StoreError> {
    let ids: Vec<String> = live.iter().map(|(s, _)| s.session.id.to_string()).collect();
    let hidden = reader.hidden_sessions(&ids)?;
    Ok(live
        .iter()
        .zip(ids)
        .filter(|(_, id)| !hidden.contains(id))
        .map(|((s, state), id)| {
            let title = s
                .session
                .title
                .as_deref()
                .map_or("untitled".into(), |t| one_line(t, 80));
            let mut line = format!("- {id}  {title}: {state}");
            if let Some(on) = s.waiting_on_at(now) {
                let _ = write!(line, " ({on})");
            }
            if let Some(cwd) = s.session.cwd.as_deref() {
                let _ = write!(line, ", in {cwd}");
            }
            if let Some(task) = s.task() {
                let _ = write!(line, ", task {task}");
            }
            if s.footprint() > 0 {
                let _ = write!(line, ", uncommitted edits to {}", s.footprint());
            }

            if let Some(todo) = s.todo() {
                let _ = write!(line, ", doing \"{}\"", one_line(todo, 100));
            }
            if let Some(at) = s.last_event_at().and_then(Timestamp::from_unix_ms) {
                let _ = write!(line, ", last event {}", day(Some(at)));
            }
            Item {
                text: line,
                refs: vec![format!("session:{id}")],
            }
        })
        .collect())
}

/// What a recent session did, in a few lines, from its stored summary and
/// its live facts.
fn session_item(s: &ScopeSession, summary: Option<&SessionSummary>) -> Item {
    let id = s.session.id.to_string();
    let counted = s.session.messages.map_or_else(
        || "messages not counted yet".to_owned(),
        |n| format!("{n} messages"),
    );
    let mut text = format!(
        "- {}  {}  ({counted})  {id}",
        day(s.session.last_activity),
        s.session
            .title
            .as_deref()
            .map_or_else(|| "untitled".into(), |t| one_line(t, 100)),
    );
    let mut did = Vec::new();
    let mut open = Vec::new();
    if let Some(work) = summary {
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
        if work.files_changed > 0 {
            let shown: Vec<&str> = work
                .recent_files
                .iter()
                .map(|p| p.rsplit('/').next().unwrap_or(p))
                .collect();
            let more = work.files_changed.saturating_sub(shown.len() as u64);
            did.push(format!(
                "changed {} file{}: {}{}",
                work.files_changed,
                if work.files_changed == 1 { "" } else { "s" },
                shown.join(", "),
                if more > 0 {
                    format!(" and {more} more")
                } else {
                    String::new()
                }
            ));
        }
        if work.unanswered_prompt {
            open.push("the last prompt has no answer".to_owned());
        }
    }
    if let Some(task) = s.task() {
        did.push(format!("claimed {task}, still open"));
    }
    if !did.is_empty() {
        let _ = write!(text, "\n    did: {}", did.join("; "));
    }
    if s.open_tool_call() || summary.is_some_and(|w| w.unfinished_tool_call) {
        open.push("a tool call never finished".to_owned());
    }
    if let Some(w) = s.waiting_on() {
        open.push(format!("waiting on {w}"));
    }
    if !open.is_empty() {
        let _ = write!(text, "\n    open: {}", open.join("; "));
    }
    if let Some(answer) = summary.and_then(|w| w.last_answer.as_deref()) {
        let _ = write!(text, "\n    last answer: \"{}\"", one_line(answer, 300));
    }
    Item {
        text,
        refs: vec![format!("session:{id}")],
    }
}

#[allow(clippy::too_many_arguments)]
fn session_start(
    reader: &Reader<'_>,
    scope: &SharedScopeIndex,
    queued: Option<LastBrief>,
    cwd: &Path,
    session: Option<&SessionId>,
    source: StartSource,
    budget: u64,
    delivery_id: &str,
    context: &std::sync::Arc<crate::operation::Context>,
) -> Result<Prepared, StoreError> {
    let now: Timestamp = context.clock.now.into();
    let dirs = scope_dirs(cwd);
    let here = cwd.display().to_string();
    let me = session.map(ToString::to_string);

    // On resume and after compaction, only what changed since this
    // session's last brief, committed or still queued.
    let previous = match (source, &me) {
        (StartSource::Resume | StartSource::Compact, Some(s)) => LastBrief::newest(
            reader
                .deliveries_to(s)?
                .into_iter()
                .next()
                .map(|d| LastBrief {
                    at: d.at,
                    seq: d.seq,
                }),
            queued,
        ),
        _ => None,
    };
    let claims_since = previous.map_or(now.unix_ms() - CLAIM_WINDOW_MS, |p| p.at.unix_ms());
    let mut deadline = Deadline::default();
    let stop = polylogue_store::read::limit::ReadLimit::current().and_then(|l| l.deadline());
    let found = scope.read(stop, |index| {
        Here::of(
            index,
            &dirs,
            me.as_deref(),
            now.unix_ms(),
            previous.map(|p| p.at),
            claims_since,
        )
    });
    let mut targets: Vec<Reference> = dirs.iter().map(|d| Reference::Cwd(d.clone())).collect();
    let recorded = found.as_ref().map(|h| h.repos.clone()).unwrap_or_default();
    targets.extend(repo_names(&dirs, recorded).into_iter().map(Reference::Repo));
    if let Some(s) = session {
        targets.push(Reference::Session(s.clone()));
    }
    let (mut recent, live, beads) = match found {
        Some(h) => (h.recent, Some(h.live), Some(h.beads)),
        None => {
            deadline
                .late
                .extend(["live sessions", "open Beads", "recent sessions"]);
            (Vec::new(), None, None)
        }
    };
    let more_sessions = recent.len() as u64 > RECENT_SESSIONS;
    recent.truncate(RECENT_SESSIONS as usize);
    // Handoffs and blockers on the sessions listed belong with them.
    targets.extend(
        recent
            .iter()
            .map(|s| Reference::Session(s.session.id.clone())),
    );

    let assertion_scope = targets
        .iter()
        .map(|reference| {
            format!(
                "about:{}",
                serde_json::to_string(&reference.to_string()).expect("reference serializes")
            )
        })
        .collect::<Vec<_>>()
        .join(" OR ");
    let operator_notes_command = format!(
        "polylogue query {}",
        polylogue_render::shell_quote(&format!(
            "assertions where author:operator ({assertion_scope})"
        ))
    );
    let other_assertions_command = format!(
        "polylogue query {}",
        polylogue_render::shell_quote(&format!(
            "assertions where NOT author:operator judged:unknown ({assertion_scope})"
        ))
    );

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
    let mut protected_prompts = String::new();
    if source == StartSource::Compact
        && let Some(s) = &me
    {
        let prompts = reader.prompts(s)?;
        if !prompts.is_empty() {
            let _ = writeln!(
                protected_prompts,
                "\nEvery prompt the operator typed in this session, verbatim, oldest first:"
            );
            for p in &prompts {
                let _ = writeln!(protected_prompts, "\n[{}] {}", day(p.at), p.text.trim_end());
                prompt_refs.push(format!("message:{}", p.message_id));
            }
        }
    }
    // Sections run cheapest first, each within the deadline.
    let notes = deadline
        .section(reader, "operator notes", || {
            operator_notes(reader, targets.clone(), previous.map(|p| p.seq), context)
        })?
        .unwrap_or_default();

    let mut fit = Fitter::new(budget, header)?;
    fit.text.push_str(&protected_prompts);
    fit.refs.extend(prompt_refs);

    fit.section(
        "Notes from the operator:",
        notes
            .iter()
            .map(|n| Item {
                text: note_line(n),
                refs: vec![format!("assertion:{}@{}", n.fact.root, n.fact.id)],
            })
            .collect(),
        "operator notes",
        &operator_notes_command,
    );
    if let Some(live) = live.filter(|live| !live.is_empty())
        && let Some(items) = deadline.section(reader, "live sessions", || {
            live_items(reader, &live, now.unix_ms())
        })?
    {
        fit.section(
            "Other agents working here now:",
            items,
            "live sessions",
            "polylogue live",
        );
    }
    // Beads come from the scope index and read nothing.
    let beads: Vec<Item> = beads
        .unwrap_or_default()
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
        "polylogue list task:",
    );
    let sessions = deadline.section(reader, "recent sessions", || {
        recent
            .iter()
            .map(|s| Ok(session_item(s, reader.session_summary(s.pk)?.as_ref())))
            .collect::<Result<Vec<Item>, StoreError>>()
    })?;
    if let Some(sessions) = sessions {
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
    }
    let notices = deadline.section(reader, "notices", || {
        reader.notices_since(previous.map_or(now.unix_ms() - NOTICE_WINDOW_MS, |p| p.at.unix_ms()))
    })?;
    let notices: Vec<Item> = notices
        .unwrap_or_default()
        .into_iter()
        .map(|(row, text)| Item {
            text: format!("- {}  {}", day(Some(row.at)), one_line(&text, 200)),
            refs: vec![format!("delivery:{}", row.id)],
        })
        .collect();
    fit.section("Signal notices:", notices, "notices", "polylogue notices");

    let other_assertions = deadline
        .section(reader, "assertions by other authors", || {
            unjudged_other_assertions(reader, &targets, context)
        })?
        .unwrap_or(0);
    if other_assertions > 0 {
        fit.left_out.push(LeftOut {
            what: format!("{other_assertions} assertions by other authors"),
            reason: "an assertion by another author needs an eligible operator judgment before injection".into(),
        });
        fit.pointers.push(format!(
            "{other_assertions} assertions by other authors, not yet accepted: `{other_assertions_command}`"
        ));
    }
    if !deadline.late.is_empty() {
        for what in &deadline.late {
            fit.left_out.push(LeftOut {
                what: (*what).into(),
                reason: "deadline".into(),
            });
        }
        fit.late = Some(format!(
            "Not available before the deadline: {}; the whole brief: `polylogue brief --cwd {} --deadline none`",
            deadline.late.join(", "),
            polylogue_render::shell_quote(&here)
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
        event: match source {
            StartSource::Startup | StartSource::Clear => DeliveryEvent::SessionStart,
            StartSource::Resume => DeliveryEvent::Resume,
            StartSource::Compact => DeliveryEvent::Compaction,
        }
        .as_str()
        .into(),
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
    budget: u64,
    delivery_id: &str,
    context: &std::sync::Arc<crate::operation::Context>,
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
    let notes = assertions::list(
        reader,
        &format!(
            "{} author:operator",
            assertions::about(&[Reference::Session(session.clone())])
        ),
        context,
    )?;
    let session_reference = Reference::Session(session.clone());
    let other_assertions =
        unjudged_other_assertions(reader, std::slice::from_ref(&session_reference), context)?;
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
    if other_assertions > 0 {
        let _ = writeln!(
            text,
            "Assertions by other authors, not yet accepted: {other_assertions}; list them with `polylogue query {}`.",
            polylogue_render::shell_quote(&format!(
                "assertions where NOT author:operator judged:unknown about:{}",
                serde_json::to_string(&session_reference.to_string())
                    .expect("reference serializes")
            ))
        );
        left_out.push(LeftOut {
            what: format!("{other_assertions} assertions by other authors"),
            reason: "an assertion by another author needs an eligible operator judgment before injection".into(),
        });
    }
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
    let mut fit = Fitter::new(budget, text)?;
    fit.refs.append(&mut refs);
    fit.left_out.append(&mut left_out);
    let pull = format!("polylogue read {id}");
    fit.section(
        "Notes from the operator on it:",
        notes
            .iter()
            .map(|n| Item {
                text: note_line(n),
                refs: vec![format!("assertion:{}@{}", n.fact.root, n.fact.id)],
            })
            .collect(),
        "operator notes",
        &pull,
    );
    fit.section(
        "The operator's prompts in it, oldest first:",
        prompts
            .iter()
            .map(|p| Item {
                text: format!("[{}] {}", day(p.at), p.text.trim_end()),
                refs: vec![format!("message:{}", p.message_id)],
            })
            .collect(),
        "operator prompts",
        &pull,
    );
    fit.section(
        "Its last answer:",
        last_answer
            .filter(|a| !a.trim().is_empty())
            .into_iter()
            .map(|text| Item { text, refs: vec![] })
            .collect(),
        "last answer",
        &pull,
    );
    let (text, refs, left_out) = fit.finish(&format!(
        "\nThe full session: `{pull}`; its tool calls: `{pull} --view actions`."
    ));
    Ok(Prepared {
        event: DeliveryEvent::Takeover.as_str().into(),
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
            .map(|w| format!("{prefix}{}", assertions::quoted(w)))
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
            format!("messages where similar:{}", assertions::quoted(text)),
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
            assertions::quoted(dir)
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
    budget: u64,
    delivery_id: &str,
    context: &std::sync::Arc<crate::operation::Context>,
    ranking_identity: Option<&str>,
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
                context: std::sync::Arc::clone(context),
                embeddings: similarity,
                ranking_identity: similarity
                    .then(|| ranking_identity.map(str::to_owned))
                    .flatten(),
                near: if similarity { near } else { None },
                fields: &[],
            },
        );
        match answer {
            Ok(page) => {
                let rows = page.value.into_rows().map_err(|f| {
                    StoreError::Invalid(format!("recall query {query:?}: {}", f.message))
                })?;
                break Some((rows, indexes));
            }
            // The provider could not embed the text: the words still search.
            Err(f) if similarity && f.code() == "embed" => {
                similarity = false;
                similarity_failed = Some(f.error.message);
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
        for row in &answer.rows {
            let field = |name: &str| row.get(name).unwrap_or(&Value::Null);
            // Without fusion one list ranks every row it ranks at all.
            let found_in: Vec<&'static str> = match row.get("found_in") {
                Some(lists) => lists
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|n| names.get(n.as_u64()? as usize - 1).copied())
                    .collect(),
                None if field("rank").is_null() => vec![],
                None => vec![names[0]],
            };
            // Rows no list ranks (messages with no vector and no word)
            // come last: the hits have ended.
            if found_in.is_empty() {
                exhausted = true;
                break;
            }
            let (Some(session), Some(message)) = (field("session").as_str(), field("id").as_str())
            else {
                continue;
            };
            page.push(Hit {
                session: session.to_owned(),
                message: message.to_owned(),
                found_in,
                snippet: field("snippet")["text"].as_str().map(str::to_owned),
            });
        }
        if !exhausted {
            next = answer.next;
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
    let mut items = Vec::new();
    for sid in &sessions {
        let mut out = String::new();
        let mut refs = Vec::new();
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
        let notes = assertions::list(
            reader,
            &assertions::about(&[Reference::Session(id.clone())]),
            context,
        )?;
        for n in notes {
            let body = n.fact.text().unwrap_or("").trim();
            if n.fact.author.is_agent() {
                let _ = writeln!(
                    out,
                    "  - note by an agent ({}), for information only: «{body}»  assertion:{}@{}",
                    n.fact.author, n.fact.root, n.fact.id
                );
            } else {
                let _ = writeln!(
                    out,
                    "  - [{}] {body}  assertion:{}@{}",
                    n.fact.kind.as_str(),
                    n.fact.root,
                    n.fact.id
                );
            }
            refs.push(format!("assertion:{}@{}", n.fact.root, n.fact.id));
        }
        items.push(Item { text: out, refs });
    }
    let search_summary = SearchSummary {
        words,
        indexes: indexes.into_iter().map(str::to_owned).collect(),
        hits: page.len(),
        sessions: sessions.len(),
        more: next.is_some(),
    };
    let mut closing = String::new();
    let _ = writeln!(
        closing,
        "\nHow this was found: {}, ranked together by reciprocal rank fusion{}; \
         {} hits in {} sessions{}.",
        search_summary
            .indexes
            .iter()
            .map(|i| match i.as_str() {
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
    let mut fit = Fitter::new(budget, out)?;
    fit.section(
        "Stored context:",
        items,
        "matching sessions",
        &format!(
            "polylogue recall {}",
            polylogue_render::shell_quote(text_in)
        ),
    );
    let (out, refs, left_out) = fit.finish(&closing);
    Ok(Prepared {
        event: DeliveryEvent::Recall.as_str().into(),
        text: out,
        refs,
        left_out,
        search_summary: Some(search_summary),
        empty: false,
        next,
    })
}

#[cfg(test)]
mod tests;
