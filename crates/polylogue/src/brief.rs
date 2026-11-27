//! Context and briefings. This module is the one scheduler for context
//! Polylogue puts in front of a model: every brief (of a directory, a session or search words) is
//! prepared here, and the daemon records each one it hands out as a delivery
//! with its exact text.
//!
//! A session-start brief brings an agent up to speed on its directory:
//! - notes the operator wrote, or agents wrote and the operator accepted:
//!   corrections, lessons, handoffs, decisions, problems, goals, questions
//!   and findings on this directory,
//!   its repository, or the session itself;
//! - other agents working here now;
//! - task claims observed here with no observed close or release, labelled
//!   as observations rather than verified external tracker state;
//! - what the recent sessions here did: commits, pull requests, files
//!   changed, Beads, what they left open, the last answer;
//! - pending signal notices;
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
//! in memory, and only fitted expansions read stored summaries and requests. Sections
//! run cheapest first within the request's deadline; a section that meets
//! the deadline is left out with reason `deadline`, and the brief answers
//! with what it has.

use crate::daemon::query::{NearFn, annotations};
use crate::daemon::scope::SharedScopeIndex;
use polylogue_model::DeliveryEvent;
use polylogue_model::user::LeftOut;
use polylogue_model::{
    Reference, SessionId, Timestamp,
    annotation::{Annotation, AnnotationKind},
};
use polylogue_store::search;
use polylogue_store::{Reader, ScopeSession, SessionSummary, StoreError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Hits a search brief returns per page.
const SEARCH_PAGE: u64 = 20;
/// The default named-context budget in tokens.
pub const DEFAULT_BUDGET: u32 = 10_000;
const RESUME_BUDGET: u32 = 4_000;

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
    /// A judged file-targeted pointer for a prompt, with no archived body.
    Advisory {
        prompt: String,
        cwd: PathBuf,
        session: SessionId,
    },
    /// Taking over from another session named by its ID or a pasted session link.
    Session {
        session: SessionId,
        #[serde(default)]
        for_session: Option<SessionId>,
    },
    /// Three or more sessions named together: one line each, never a
    /// brief of each.
    Sessions {
        sessions: Vec<SessionId>,
        #[serde(default)]
        for_session: Option<SessionId>,
    },
    /// Past work relevant to a text, with a summary of how it was found.
    /// `cursor` continues from a previous page's `next`.
    Words {
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
            Self::Advisory { session, .. } => Some(session),
            Self::Session { for_session, .. }
            | Self::Sessions { for_session, .. }
            | Self::Words { for_session, .. } => for_session.as_ref(),
        }
    }

    pub fn cwd(&self) -> Option<&Path> {
        match self {
            Self::SessionStart { cwd, .. } | Self::Advisory { cwd, .. } => Some(cwd),
            Self::Words { cwd, .. } => cwd.as_deref(),
            Self::Session { .. } | Self::Sessions { .. } => None,
        }
    }
}

/// The context prepared for a brief step or a hook, with the delivery that
/// records its outcome.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BriefReply {
    /// The delivery's ID, which the hook reports the outcome of; none when no
    /// delivery was queued
    pub delivery: Option<String>,
    pub prepared: Prepared,
}

/// Prepared context, not yet delivered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Prepared {
    pub event: polylogue_model::DeliveryEvent,
    pub text: String,
    /// References included, in order.
    pub refs: Vec<String>,
    pub left_out: Vec<LeftOut>,
    /// A search brief only: how it searched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_summary: Option<SearchSummary>,
    /// Nothing worth sending: the caller sends nothing.
    pub empty: bool,
    /// A search brief only: the cursor of the next page, when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
}

/// How a search brief searched, so a reader can judge what it may have missed.
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
    if let Some(root) = git_identity::repository_of(cwd) {
        let root = root.to_string_lossy().into_owned();
        if !dirs.contains(&root) {
            dirs.push(root);
        }
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

/// The deliveries that set the point a resume or compaction brief reports
/// changes since, once delivered: session briefs, messages and other hand-overs to
/// the session do not.
pub(crate) const BRIEF_EVENTS: [polylogue_model::DeliveryEvent; 3] = [
    polylogue_model::DeliveryEvent::SessionStart,
    polylogue_model::DeliveryEvent::Resume,
    polylogue_model::DeliveryEvent::Compaction,
];

/// The latest brief a session received: when, and the archive's write
/// counter then. Annotations written after it are the next brief's delta.
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

pub struct BriefContext<'a, 'reader> {
    pub reader: &'a Reader<'reader>,
    pub pending_deliveries: &'a [polylogue_store::archive::writer::PendingDelivery],
    pub delivery_id: &'a str,
    pub context: &'a std::sync::Arc<polylogue_reading::operation::Context>,
    pub ranking_identity: Option<&'a str>,
    pub near: Option<&'a NearFn<'a>>,
}

/// Prepares the context `ask` wants. `delivery_id` goes in the text, so a
/// reader of the transcript can find the delivery record. `scope` is the
/// daemon's scope index. `near` ranks a search brief by similarity when vectors exist.
pub fn prepare(
    brief: BriefContext<'_, '_>,
    scope: &SharedScopeIndex,
    pending_advice: &[Vec<String>],
    ask: &Ask,
    budget_tokens: command_surface::TokenBudget,
) -> Result<Prepared, StoreError> {
    let BriefContext {
        reader,
        delivery_id,
        context,
        ..
    } = brief;
    let budget = budget_tokens
        .resolve(Some(ask.default_budget_tokens()))
        .map_or(u64::MAX, u64::from);
    match ask {
        Ask::SessionStart {
            cwd,
            session,
            source,
        } => session_start(&brief, scope, cwd, session.as_ref(), *source, budget),
        Ask::Advisory {
            prompt,
            cwd,
            session,
        } => crate::advisory::prepare(
            reader,
            scope,
            prompt,
            cwd,
            session,
            delivery_id,
            pending_advice,
            context,
        ),
        Ask::Session { session, .. } => {
            session_brief(reader, session, budget, delivery_id, context)
        }
        Ask::Sessions { sessions, .. } => {
            session_index(reader, sessions, budget, delivery_id, context)
        }
        Ask::Words {
            text, cwd, cursor, ..
        } => search_brief(&brief, text, cwd.as_deref(), cursor.as_deref(), budget),
    }
}
/// Notes a brief carries: the operator's, and agents' the operator
/// accepted.
fn operator_note_ids(
    reader: &Reader<'_>,
    targets: Vec<Reference>,
    after_seq: Option<i64>,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
) -> Result<Vec<polylogue_model::RevisionReference>, StoreError> {
    let now: Timestamp = context.clock.now.into();
    let kinds = [
        AnnotationKind::Correction,
        AnnotationKind::Lesson,
        AnnotationKind::Handoff,
        AnnotationKind::Decision,
        AnnotationKind::Problem,
        AnnotationKind::Goal,
        AnnotationKind::Question,
        AnnotationKind::Finding,
    ]
    .map(|k| format!("kind:{}", k.as_str()))
    .join(" OR ");
    let mut conditions = format!(
        "{} ({kinds}) (author:operator OR (agent:true judged:accepted))",
        annotations::about(&targets)
    );
    if let Some(after) = after_seq {
        conditions.push_str(&format!(" write_counter > {after}"));
    }
    let mut eligible = Vec::new();
    for reference in annotations::ids(reader, &conditions, context)? {
        if reader.annotation_is_unexpired(&reference, now)? {
            eligible.push(reference);
        }
    }
    Ok(eligible)
}

/// Recheck the existing selection's eligibility before loading its body.
fn eligible_note(
    reader: &Reader<'_>,
    reference: &polylogue_model::RevisionReference,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
) -> Result<Option<Annotation>, StoreError> {
    if !reader
        .annotation_revision_state(reference)?
        .is_some_and(|state| state.current)
        || !reader.annotation_is_unexpired(reference, context.clock.now.into())?
    {
        return Ok(None);
    }
    let Some(id) = &reference.revision else {
        return Ok(None);
    };
    let eligible = annotations::count(
        reader,
        &format!(
            "id:{} (author:operator OR (agent:true judged:accepted))",
            annotations::quoted(id)
        ),
        context,
    )? > 0;
    if eligible {
        reader.annotation(reference)
    } else {
        Ok(None)
    }
}

/// Bodies of the eligible notes on an already selected session.
fn operator_notes(
    reader: &Reader<'_>,
    targets: Vec<Reference>,
    after_seq: Option<i64>,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
) -> Result<Vec<Annotation>, StoreError> {
    let mut notes = Vec::new();
    for reference in operator_note_ids(reader, targets, after_seq, context)? {
        if let Some(note) = eligible_note(reader, &reference, context)? {
            notes.push(note);
        }
    }
    Ok(notes)
}

/// Agent-written notes here the operator has not judged yet.
fn unjudged_other_annotations(
    reader: &Reader<'_>,
    targets: &[Reference],
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
) -> Result<u64, StoreError> {
    annotations::count(
        reader,
        &format!(
            "{} NOT author:operator judged:unknown",
            annotations::about(targets)
        ),
        context,
    )
}

/// A clock time in the logical-day zone with its offset, to the minute.
fn day(t: Option<Timestamp>, zone: &jiff::tz::TimeZone) -> String {
    t.map(|t| polylogue_model::local_minute(t, zone))
        .unwrap_or_else(|| "unknown time".into())
}

fn note_line(n: &Annotation, zone: &jiff::tz::TimeZone) -> String {
    let label = n.fact.key.as_deref().unwrap_or(n.fact.kind.as_str());
    let targets: Vec<String> = n.fact.about().map(ToString::to_string).collect();
    let by = if n.fact.author.is_agent() {
        format!(", declared author {}", n.fact.author)
    } else {
        String::new()
    };
    format!(
        "- [{label}] {} ({}, {}{by})  {}",
        n.fact.text().unwrap_or(""),
        targets.join(", "),
        day(n.fact.created_at, zone),
        n.fact.reference()
    )
}

fn annotation_item(
    reader: &Reader<'_>,
    note: &Annotation,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
) -> Result<Item, StoreError> {
    use polylogue_model::annotation::LinkRole;
    let reference = note.fact.reference();
    let mut contrary = Vec::new();
    let _ = reader.visit_annotation_references_linked(
        &reference,
        &[LinkRole::Challenges, LinkRole::Corrects],
        |annotation| {
            contrary.push(annotation);
            Ok(std::ops::ControlFlow::Continue(()))
        },
    )?;
    let mut text = String::new();
    let mut refs = Vec::new();
    let judgment = |annotation: &Annotation| -> Result<&'static str, StoreError> {
        if annotation.fact.author == polylogue_model::Author::Operator {
            return Ok("operator-authored");
        }
        if annotation.fact.author.is_agent()
            && annotations::count(
                reader,
                &format!(
                    "id:{} judged:accepted",
                    annotations::quoted(&annotation.fact.id)
                ),
                context,
            )? > 0
        {
            return Ok("accepted by operator");
        }
        Ok("unknown")
    };
    let mut attribution = format!(
        "original author:{} via:{} judgment:{}",
        note.fact.author,
        note.fact.via.as_str(),
        judgment(note)?
    );
    let mut not_inlined = Vec::new();
    for revision in contrary {
        if !reader.annotation_is_unexpired(&revision, context.clock.now.into())? {
            continue;
        }
        let Some(correction) = eligible_note(reader, &revision, context)? else {
            let reference = Reference::Annotation(revision);
            refs.push(reference.to_string());
            not_inlined.push(reference.to_string());
            continue;
        };
        let state = judgment(&correction)?;
        let _ = write!(
            attribution,
            "; contrary {} author:{} via:{} judgment:{}",
            correction.fact.reference(),
            correction.fact.author,
            correction.fact.via.as_str(),
            state
        );
        let _ = writeln!(
            text,
            "Contrary note, author:{} via:{}; judgment {state}: {}",
            correction.fact.author,
            correction.fact.via.as_str(),
            note_line(&correction, context.clock.day.zone())
        );
        refs.push(correction.fact.reference().to_string());
    }
    if !not_inlined.is_empty() {
        let _ = write!(
            attribution,
            "; contrary annotations not inlined: {}; inspect: {}",
            not_inlined.len(),
            not_inlined
                .iter()
                .map(|reference| format!("`polylogue {reference} read`"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    text.push_str(&note_line(note, context.clock.day.zone()));
    refs.push(reference.to_string());
    Ok(Item {
        expansion: None,
        attribution: Some(attribution),
        text,
        refs,
    })
}

/// One line (or a few) of a brief section, with what it cites.
enum Expansion {
    Annotation(polylogue_model::RevisionReference),
    Session(Box<ScopeSession>),
    Notice(String),
}

struct Item {
    expansion: Option<Expansion>,
    attribution: Option<String>,
    text: String,
    refs: Vec<String>,
}

/// Candidate sections are kept until fitting so later candidates get an
/// entry before an earlier passage consumes the expansion budget.
struct Section {
    heading: String,
    items: Vec<Item>,
    what: String,
    pull: String,
}

struct Fitter {
    text: String,
    refs: Vec<String>,
    left_out: Vec<LeftOut>,
    room: usize,
    pointers: Vec<String>,
    late: Option<String>,
    budget: u64,
    sections: Vec<Section>,
}

/// A source prefix ends on a UTF-8 boundary. Quoting it is lossless, even
/// when it contains newlines, quotes, role markers or fence delimiters.
fn prefix(text: &str, bytes: usize) -> &str {
    let mut end = text.len().min(bytes);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Standard JSON escapes preserve every source byte while keeping terminal
/// controls and conversational role markers out of the rendered data fence.
pub(crate) fn archive_json(value: &serde_json::Value) -> String {
    let mut encoded = polylogue_render::terminal::json(value, false);
    for (marker, escaped) in [
        ("<", r"\u003c"),
        (">", r"\u003e"),
        ("Human:", r"Human\u003a"),
        ("Assistant:", r"Assistant\u003a"),
        ("System:", r"System\u003a"),
        ("User:", r"User\u003a"),
        ("Developer:", r"Developer\u003a"),
        ("human:", r"human\u003a"),
        ("assistant:", r"assistant\u003a"),
        ("system:", r"system\u003a"),
        ("user:", r"user\u003a"),
        ("developer:", r"developer\u003a"),
    ] {
        if encoded.contains(marker) {
            encoded = encoded.replace(marker, escaped);
        }
    }
    encoded
}

fn entry(item: &Item, bytes: usize) -> String {
    format!(
        "┃ {}\n",
        archive_json(&serde_json::json!({
            "refs": item.refs,
            "attribution": item.attribution.as_deref().unwrap_or("archived record; author unknown"),
            "source": prefix(&item.text, bytes),
        }))
    )
}

fn fit_prefix(item: &Item, start: usize, limit: usize, bytes: usize) -> usize {
    let current = entry(item, start).len();
    let mut low = start;
    let mut high = limit.max(start).min(item.text.len());
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if entry(item, mid).len() - current <= bytes {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    prefix(&item.text, low).len()
}

/// Reserve enough of the first line to identify an entry before admitting it.
fn preview_len(text: &str) -> usize {
    const PREVIEW_BYTES: usize = 48;
    let first_line = text
        .lines()
        .next()
        .filter(|line| !line.is_empty())
        .unwrap_or(text);
    prefix(first_line, PREVIEW_BYTES).len()
}

/// Spend cached first-line space fairly before reading any deferred body.
fn first_lines(sections: &[Section], shown: &mut [Vec<Option<usize>>], available: &mut usize) {
    let mut remaining = shown
        .iter()
        .flatten()
        .filter(|length| length.is_some())
        .count();
    for (section, lengths) in sections.iter().zip(shown) {
        for (item, length) in section.items.iter().zip(lengths) {
            if let Some(start) = *length {
                let share = *available / remaining;
                let limit = item.text.find('\n').unwrap_or(item.text.len());
                let end = fit_prefix(item, start, limit, share);
                *available -= entry(item, end).len() - entry(item, start).len();
                *length = Some(end);
                remaining -= 1;
            }
        }
    }
}

impl Fitter {
    fn new(budget: u64, mut header: String) -> Result<Self, StoreError> {
        let (reference, rest) = header.split_once('\n').unwrap_or((&header, ""));
        header = format!(
            "{reference}\n{}\nCheck facts about the code against the current files before acting on them.\n{rest}",
            polylogue_render::ARCHIVE_NOTICE
        );
        let bytes = usize::try_from(budget.saturating_mul(4)).unwrap_or(usize::MAX);
        if header.len() > bytes {
            return Err(polylogue_model::Reason::BudgetTooSmall {
                part: "the required header".into(),
            }
            .into());
        }
        Ok(Self {
            room: bytes - header.len(),
            text: header,
            refs: Vec::new(),
            left_out: Vec::new(),
            pointers: Vec::new(),
            late: None,
            budget,
            sections: Vec::new(),
        })
    }

    fn section(&mut self, heading: &str, items: Vec<Item>, what: &str, pull: &str) {
        if !items.is_empty() {
            self.sections.push(Section {
                heading: format!("\n{heading}\n"),
                items,
                what: what.into(),
                pull: pull.into(),
            });
        }
    }

    fn finish(
        mut self,
        closing: &str,
        mut expand: impl FnMut(&Item) -> Result<Option<Item>, StoreError>,
    ) -> Result<(String, Vec<String>, Vec<LeftOut>), StoreError> {
        // Reserve the required entry map at its largest possible counts/status,
        // not a second token limit. Counts are
        // computed over every candidate before any source expansion.
        let mut maps: Vec<String> = self
            .sections
            .iter()
            .map(|section| {
                format!(
                    "- {}: {count} whole / {count} shortened / {count} omitted; details not available (deadline, read failure, budget): {failures}; expand: `{}`\n",
                    section.what,
                    section.pull,
                    count = section.items.len(), failures = section.items.len().saturating_mul(2)
                )
            })
            .collect();
        let mut tail = String::from("\nEntry map (whole / shortened / omitted):\n");
        for map in &maps {
            tail.push_str(map);
        }
        for pointer in &self.pointers {
            let _ = writeln!(tail, "- {pointer}");
        }
        if let Some(late) = &self.late {
            let _ = writeln!(tail, "{late}");
        }
        let _ = writeln!(tail, "{closing}");
        if tail.len() > self.room {
            return Err(polylogue_model::Reason::BudgetTooSmall {
                part: "the required entry map".into(),
            }
            .into());
        }
        let mut available = self.room - tail.len();
        let mut shown: Vec<Vec<Option<usize>>> = Vec::new();
        // Admit a candidate only when its preview and attribution fit.
        for section in &self.sections {
            let mut lengths = vec![None; section.items.len()];
            let mut heading_added = false;
            for (item, length) in section.items.iter().zip(&mut lengths) {
                let first = preview_len(&item.text);
                let cost = entry(item, first).len()
                    + if heading_added {
                        0
                    } else {
                        section.heading.len()
                    };
                if cost <= available {
                    available -= cost;
                    *length = Some(first);
                    heading_added = true;
                }
            }
            shown.push(lengths);
        }
        if !self.sections.is_empty() && shown.iter().flatten().all(Option::is_none) {
            return Err(polylogue_model::Reason::BudgetTooSmall {
                part: "one candidate entry".into(),
            }
            .into());
        }
        first_lines(&self.sections, &mut shown, &mut available);
        // Only selected candidates may read their stored summaries or bodies,
        // outside the scope-index lock and under the caller's existing limit.
        let mut unavailable: Vec<Vec<&'static str>> = vec![Vec::new(); self.sections.len()];
        for (section_index, (section, lengths)) in
            self.sections.iter_mut().zip(&mut shown).enumerate()
        {
            for (item, length) in section.items.iter_mut().zip(lengths) {
                let Some(start) = *length else {
                    continue;
                };
                if available == 0 {
                    continue;
                }
                let expanded = match expand(item) {
                    Ok(Some(expanded)) => expanded,
                    Ok(None) => continue,
                    Err(error) => {
                        unavailable[section_index].push(
                            if matches!(error, StoreError::ReadDeadlineExceeded) {
                                "deadline"
                            } else {
                                "read failure"
                            },
                        );
                        self.left_out.push(LeftOut {
                            what: format!("details of {}", item.refs.join(" ")),
                            reason: format!("not available ({error})"),
                        });
                        Item {
                            expansion: None,
                            attribution: item.attribution.clone(),
                            refs: item.refs.clone(),
                            text: format!("Details not available ({error}).\n{}", item.text),
                        }
                    }
                };
                let first = preview_len(&expanded.text);
                let before = entry(item, start).len();
                let after = entry(&expanded, first).len();
                if after <= before + available {
                    available = available + before - after;
                    *item = expanded;
                    *length = Some(first);
                } else {
                    unavailable[section_index].push("budget");
                    self.left_out.push(LeftOut {
                        what: format!("details of {}", item.refs.join(" ")),
                        reason: format!(
                            "budget cannot fit source attribution; expand: {}",
                            section.pull
                        ),
                    });
                }
            }
        }
        // First give every selected candidate its share of a useful first
        // line; then spend the remaining expansion budget in stable order.
        first_lines(&self.sections, &mut shown, &mut available);
        for (section, lengths) in self.sections.iter().zip(&mut shown) {
            for (item, length) in section.items.iter().zip(lengths) {
                if let Some(start) = *length {
                    let end = fit_prefix(item, start, item.text.len(), available);
                    available -= entry(item, end).len() - entry(item, start).len();
                    *length = Some(end);
                }
            }
        }
        for (section_index, ((section, lengths), map)) in
            self.sections.iter().zip(shown).zip(&mut maps).enumerate()
        {
            let whole = lengths
                .iter()
                .zip(&section.items)
                .filter(|(length, item)| **length == Some(item.text.len()))
                .count();
            let omitted = lengths.iter().filter(|length| length.is_none()).count();
            let shortened = section.items.len() - whole - omitted;
            if omitted < section.items.len() {
                self.text.push_str(&section.heading);
            }
            for (item, length) in section.items.iter().zip(lengths) {
                if let Some(length) = length {
                    self.text.push_str(&entry(item, length));
                    self.refs.extend(item.refs.iter().cloned());
                }
            }
            // Replacing the long reservation with these counts only releases
            // space. Every section keeps its pull command, even when omitted.
            let failures = unavailable[section_index].len();
            unavailable[section_index].sort_unstable();
            unavailable[section_index].dedup();
            let status = if failures == 0 {
                String::new()
            } else {
                format!(
                    "; details not available ({}): {failures}",
                    unavailable[section_index].join(", ")
                )
            };
            *map = format!(
                "- {}: {whole} whole / {shortened} shortened / {omitted} omitted{status}; expand: `{}`\n",
                section.what, section.pull
            );
            if omitted > 0 {
                self.left_out.push(LeftOut {
                    what: format!("{omitted} {}", section.what),
                    reason: format!("budget {} tokens; expand: {}", self.budget, section.pull),
                });
            }
            if shortened > 0 {
                self.left_out.push(LeftOut {
                    what: format!("{shortened} shortened {}", section.what),
                    reason: format!(
                        "source tails omitted for budget {} tokens; expand: {}",
                        self.budget, section.pull
                    ),
                });
            }
        }
        self.text
            .push_str("\nEntry map (whole / shortened / omitted):\n");
        for map in maps {
            self.text.push_str(&map);
        }
        for pointer in self.pointers {
            let _ = writeln!(self.text, "- {pointer}");
        }
        if let Some(late) = self.late {
            let _ = writeln!(self.text, "{late}");
        }
        let _ = writeln!(self.text, "{closing}");
        Ok((self.text, self.refs, self.left_out))
    }
}

/// What the scope index says about a brief's directories, copied out of it
/// so that no archive read happens under its lock.
struct Here {
    repos: Vec<String>,
    no_sessions: bool,
    /// Lightweight scope metadata after the last brief, newest first. Bodies
    /// and session annotations are read only for fitted expansions.
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
    ) -> Self {
        let sessions = index.sessions_under(dirs, None);
        let no_sessions = sessions.is_empty();
        let recent = sessions
            .into_iter()
            .filter(|s| me.is_none_or(|me| s.session.id.to_string() != me))
            .filter(|s| after.is_none_or(|p| s.session.last_activity.is_some_and(|t| t > p)))
            .cloned()
            .collect();
        Self {
            repos: index.repos_under(dirs),
            no_sessions,
            recent,
            live: index
                .live_under(dirs, me, now)
                .into_iter()
                .map(|live| (live.session.clone(), live.state.as_str()))
                .collect(),
            beads: index.open_beads(dirs),
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
    failed: Vec<LeftOut>,
}

impl Deadline {
    fn section<T>(
        &mut self,
        reader: &Reader<'_>,
        what: &'static str,
        work: impl FnOnce() -> Result<T, StoreError>,
    ) -> Option<T> {
        match reader.check().and_then(|()| work()) {
            Ok(value) => Some(value),
            Err(StoreError::ReadDeadlineExceeded) => {
                self.late.push(what);
                None
            }
            Err(error) => {
                self.failed.push(LeftOut {
                    what: what.into(),
                    reason: format!("not available ({error})"),
                });
                None
            }
        }
    }

    fn show_failures(&mut self, fit: &mut Fitter) {
        for failure in self.failed.drain(..) {
            fit.pointers
                .push(format!("{}: {}", failure.what, failure.reason));
            fit.left_out.push(failure);
        }
    }
}

/// Other sessions working, waiting or idle under the brief's directories,
/// leaving out the ones a `hidden` note hides.
fn live_items(
    reader: &Reader<'_>,
    live: &[(ScopeSession, &'static str)],
    now: i64,
    zone: &jiff::tz::TimeZone,
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
                .map_or("untitled".into(), |t| t.to_owned());
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
            if s.edited_files() > 0 {
                let files = s.edited_files();
                let noun = if files == 1 { "file" } else { "files" };
                let _ = write!(line, ", uncommitted edits to {files} {noun}");
            }

            if let Some(todo) = s.todo() {
                let _ = write!(line, ", doing \"{}\"", todo);
            }
            if let Some(at) = s.last_event_at().and_then(Timestamp::from_unix_ms) {
                let _ = write!(line, ", last event {}", day(Some(at), zone));
            }
            Item {
                expansion: None,
                attribution: None,
                text: line,
                refs: vec![format!("session:{id}")],
            }
        })
        .collect())
}

/// What a recent session did, in a few lines, from its stored summary and
/// its live facts.
fn session_item(
    s: &ScopeSession,
    summary: Option<&SessionSummary>,
    now: i64,
    zone: &jiff::tz::TimeZone,
) -> Item {
    let id = s.session.id.to_string();
    let counted = s.session.messages.map_or_else(
        || "messages not counted yet".to_owned(),
        |n| format!("{n} messages"),
    );
    let mut text = format!(
        "- {}  recorded title: {}  ({counted})  {id}",
        day(s.session.last_activity, zone),
        s.session
            .title
            .as_deref()
            .map_or_else(|| "untitled".into(), |t| t.to_owned()),
    );
    let mut did = Vec::new();
    let mut open = Vec::new();
    if let Some(work) = summary {
        if !work.commits.is_empty() {
            did.push(format!(
                "{} commit{} ({})",
                work.commits.len(),
                if work.commits.len() == 1 { "" } else { "s" },
                work.commits.join(", ")
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
        did.push(format!("claimed {task}; no close or release observed"));
    }
    if !did.is_empty() {
        let _ = write!(text, "\n    did: {}", did.join("; "));
    }
    if s.open_tool_call() || summary.is_some_and(|w| w.unfinished_tool_call) {
        open.push("a tool call never finished".to_owned());
    }
    if let Some(w) = s.waiting_on_at(now) {
        open.push(format!("waiting on {w}"));
    }
    if !open.is_empty() {
        let _ = write!(text, "\n    open: {}", open.join("; "));
    }
    if let Some(answer) = summary.and_then(|w| w.last_answer.as_deref()) {
        let _ = write!(text, "\n    last answer: \"{}\"", answer);
    }
    Item {
        expansion: Some(Expansion::Session(Box::new(s.clone()))),
        attribution: Some(format!(
            "scope-index observation; {} messages; last observed {}; current external state unverified",
            s.session
                .messages
                .map_or("unknown".into(), |count| count.to_string()),
            day(s.session.last_activity, zone)
        )),
        text,
        refs: vec![format!("session:{id}")],
    }
}

fn session_start(
    brief: &BriefContext<'_, '_>,
    scope: &SharedScopeIndex,
    cwd: &Path,
    session: Option<&SessionId>,
    source: StartSource,
    budget: u64,
) -> Result<Prepared, StoreError> {
    let BriefContext {
        reader,
        delivery_id,
        context,
        ..
    } = *brief;
    let now: Timestamp = context.clock.now.into();
    let dirs = scope_dirs(cwd);
    let here = cwd.display().to_string();
    let me = session.map(ToString::to_string);

    // On resume and after compaction, only what changed since this
    // session's last delivered brief, committed or still queued.
    let previous = match (source, &me) {
        (StartSource::Resume | StartSource::Compact, Some(recipient)) => {
            use polylogue_model::user::UserWrite;
            let pending = brief.pending_deliveries;
            let delivered = |id: &str| {
                pending.iter().any(|request| {
                    matches!(&request.request.write, UserWrite::DeliveryEvent {
                        delivery, what: polylogue_model::DeliveryOutcome::Delivered, ..
                    } if delivery == id)
                })
            };
            let stored = reader.deliveries_to(recipient)?;
            let committed = stored
                .iter()
                .find(|delivery| {
                    BRIEF_EVENTS.contains(&delivery.event)
                        && (delivery.events.iter().any(|what| what == "delivered")
                            || delivered(&delivery.id))
                })
                .map(|delivery| LastBrief {
                    at: delivery.at,
                    seq: delivery.seq,
                });
            let mut previous = committed;
            for pending in pending.iter().rev() {
                let request = &pending.request;
                if let UserWrite::Deliver(delivery) = &request.write
                    && delivery.recipient_id.as_deref() == Some(recipient.as_str())
                    && BRIEF_EVENTS.contains(&delivery.event)
                    && delivered(&request.id)
                    && let Some(prepared_at) = pending.prepared_at
                    && !stored.iter().any(|delivery| delivery.id == request.id)
                {
                    previous = LastBrief::newest(
                        previous,
                        Some(LastBrief {
                            at: request.at,
                            seq: prepared_at,
                        }),
                    );
                    break;
                }
            }
            previous
        }
        _ => None,
    };
    let mut deadline = Deadline::default();
    let found = scope.read(Some(std::time::Instant::now()), |index| {
        Here::of(
            index,
            &dirs,
            me.as_deref(),
            now.unix_ms(),
            previous.map(|p| p.at),
        )
    });
    let mut targets: Vec<Reference> = dirs.iter().map(|d| Reference::Cwd(d.clone())).collect();
    let recorded = found.as_ref().map(|h| h.repos.clone()).unwrap_or_default();
    targets.extend(repo_names(&dirs, recorded).into_iter().map(Reference::Repo));
    if let Some(s) = session {
        targets.push(Reference::Session(s.clone()));
    }
    let scope_loaded = found.is_some();
    let no_sessions = found.as_ref().is_some_and(|h| h.no_sessions);
    let (recent, live, beads) = match found {
        Some(h) => (h.recent, Some(h.live), Some(h.beads)),
        None => {
            deadline.failed.push(LeftOut {
                what: "live sessions, open Beads, recent sessions".into(),
                reason: "not available (scope index has not loaded); check `polylogue status`"
                    .into(),
            });
            (Vec::new(), None, None)
        }
    };

    let annotation_scope = targets
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
        "polylogue {}",
        polylogue_render::shell_quote(&format!(
            "annotations where author:operator ({annotation_scope})"
        ))
    );
    let other_annotations_command = format!(
        "polylogue {}",
        polylogue_render::shell_quote(&format!(
            "annotations where NOT author:operator judged:unknown ({annotation_scope})"
        ))
    );

    let mut header = format!(
        "[polylogue brief delivery:{delivery_id}: refs in entry map]\nPolylogue brief for {here} (delivery {delivery_id})\n"
    );
    if let Some(p) = &previous {
        let _ = writeln!(
            header,
            "Changes since your last brief at {}.",
            day(Some(p.at), context.clock.day.zone())
        );
    }
    // The operator's own words after compaction are never cut.
    let mut prompt_refs = Vec::new();
    let mut protected_prompts = String::new();
    if source == StartSource::Compact
        && let Some(s) = &me
    {
        let prompts = deadline
            .section(reader, "operator prompts", || reader.prompts(s))
            .unwrap_or_default();
        if !prompts.is_empty() {
            let _ = writeln!(
                protected_prompts,
                "\nEvery prompt the operator typed in this session, verbatim, oldest first:"
            );
            for p in &prompts {
                let _ = writeln!(
                    protected_prompts,
                    "\n[written_by:human author:operator session:{s} message:{} at:{}]\n┃ {}",
                    p.message_id,
                    day(p.at, context.clock.day.zone()),
                    archive_json(&serde_json::json!(&p.text))
                );
                prompt_refs.push(format!("message:{}", p.message_id));
            }
        }
    }
    // Sections run cheapest first, each within the deadline.
    let notes = deadline
        .section(reader, "operator notes", || {
            operator_note_ids(reader, targets.clone(), previous.map(|p| p.seq), context)
        })
        .unwrap_or_default();

    let with_manual = matches!(
        source,
        StartSource::Startup | StartSource::Clear | StartSource::Compact
    );
    if with_manual {
        header.push_str("\nAgent manual:\n");
        header.push_str(&crate::manual::text());
        header.push('\n');
    }
    header.push_str(&protected_prompts);
    if scope_loaded && no_sessions {
        let _ = writeln!(header, "No session is recorded for this directory: {here}.");
    }
    let mut fit = Fitter::new(budget, header)?;
    fit.refs.extend(prompt_refs);

    fit.section(
        "Notes from the operator:",
        notes.into_iter().map(|reference| Item {
            text: format!("Scoped annotation {reference}; content not expanded"),
            refs: vec![Reference::Annotation(reference.clone()).to_string()],
            expansion: Some(Expansion::Annotation(reference)),
            attribution: Some("selected as operator/accepted annotation; exact revision; content not expanded; recheck before use".into()),
        }).collect(),
        "operator notes",
        &operator_notes_command,
    );
    if let Some(live) = live.filter(|live| !live.is_empty())
        && let Some(items) = deadline.section(reader, "live sessions", || {
            live_items(reader, &live, now.unix_ms(), context.clock.day.zone())
        })
    {
        fit.section(
            "Other agents working here now:",
            items,
            "live sessions",
            "polylogue running",
        );
    }
    // Beads come from the scope index and read nothing.
    let beads: Vec<Item> = beads
        .unwrap_or_default()
        .into_iter()
        .map(|b| Item {
                expansion: None,
                attribution: None,
            text: format!(
                "- {}: last observed claim by {}; session last active {}; no close or release observed; current external state unverified",
                b.id,
                b.session,
                day(b.last_active, context.clock.day.zone())
            ),
            refs: vec![format!("session:{}", b.session)],
        })
        .collect();
    fit.section(
        "Bead claims last observed here, with no observed close or release:",
        beads,
        "open Beads",
        "polylogue 'task:*'",
    );
    let sessions = recent
        .iter()
        .map(|s| {
            session_item(
                s,
                None,
                context.clock.now.as_millisecond(),
                context.clock.day.zone(),
            )
        })
        .collect();
    fit.section(
        "Recent sessions here, and what they did:",
        sessions,
        "recent sessions",
        &format!(
            "polylogue {}",
            polylogue_render::shell_quote(&format!("cwd:{}", annotations::quoted(&here))),
        ),
    );
    let notices = deadline.section(reader, "notices", || {
        reader.pending_notice_candidates(&dirs, me.as_deref())
    });
    let (notices, inspect_count) = notices.unwrap_or_default();
    if inspect_count > 0 {
        fit.left_out.push(LeftOut {
            what: format!("{inspect_count} signal records"),
            reason: "recipient/directory scope or historical effect state requires inspection"
                .into(),
        });
        fit.pointers.push(format!("{inspect_count} signal records require recipient/directory scope or effect-state inspection; inspect: `polylogue notices`"));
    }
    let notices = notices
        .into_iter()
        .map(|row| Item {
            expansion: Some(Expansion::Notice(row.id.clone())),
            attribution: Some(format!(
                "sender:signal; last observed {}; no outcome recorded",
                day(Some(row.at), context.clock.day.zone())
            )),
            text: format!("Signal notice {}; contents not expanded", row.id),
            refs: vec![format!("delivery:{}", row.id)],
        })
        .collect();
    fit.section("Signal notices:", notices, "notices", "polylogue notices");

    let other_annotations = deadline
        .section(reader, "annotations by other authors", || {
            unjudged_other_annotations(reader, &targets, context)
        })
        .unwrap_or(0);
    if other_annotations > 0 {
        fit.left_out.push(LeftOut {
            what: format!("{other_annotations} annotations by other authors"),
            reason: "an annotation by another author needs an eligible operator judgment before injection".into(),
        });
        fit.pointers.push(format!(
            "{other_annotations} annotations by other authors, not yet accepted: `{other_annotations_command}`"
        ));
    }
    deadline.show_failures(&mut fit);
    if !deadline.late.is_empty() {
        for what in &deadline.late {
            fit.left_out.push(LeftOut {
                what: (*what).into(),
                reason: "deadline".into(),
            });
        }
        fit.late = Some(if scope_loaded {
            format!(
                "Not available before the deadline: {}; the whole brief: `polylogue brief --cwd {} --deadline none`",
                deadline.late.join(", "),
                polylogue_render::shell_quote(&here)
            )
        } else {
            format!(
                "Not available before the deadline: {}; check `polylogue status` while the scope index loads",
                deadline.late.join(", ")
            )
        });
    }
    let no_candidates = fit.refs.is_empty()
        && fit.sections.is_empty()
        && fit.pointers.is_empty()
        && fit.late.is_none();
    let closing = if no_candidates && previous.is_some() {
        "Nothing changed."
    } else {
        "Earlier work on a topic: `polylogue 'WORDS' brief`."
    };
    let (text, refs, left_out) = fit.finish(closing, |item| {
        if let Some(Expansion::Annotation(reference)) = &item.expansion {
            let note = eligible_note(reader, reference, context)?.ok_or_else(|| StoreError::internal("annotation eligibility changed before expansion"))?;
            return annotation_item(reader, &note, context).map(Some);
        }

        if let Some(Expansion::Notice(delivery)) = &item.expansion {
            let Some((_, text)) = reader.delivery(delivery)? else {
                return Err(StoreError::NotFound(format!("delivery {delivery}")));
            };
            return Ok(Some(Item {
                expansion: None,
                attribution: item.attribution.clone(),
                text,
                refs: item.refs.clone(),
            }));
        }
        let Some(Expansion::Session(session)) = &item.expansion else {
            return Ok(None);
        };
        reader.check()?;
        let mut expanded = session_item(
            session,
            reader.session_summary(session.pk)?.as_ref(),
            context.clock.now.as_millisecond(),
            context.clock.day.zone(),
        );
        let notes = operator_notes(
            reader,
            vec![Reference::Session(session.session.id.clone())],
            previous.map(|p| p.seq),
            context,
        )?;
        let mut text = String::new();
        for note in notes {
            let note = annotation_item(reader, &note, context)?;
            let _ = writeln!(text, "{}", note.text);
            if let Some(attribution) = &note.attribution {
                let target = expanded.attribution.get_or_insert_with(String::new);
                let _ = write!(target, "; {attribution}");
            }
            expanded.refs.extend(note.refs);
        }
        if let Some(prompt) = reader.latest_prompt(&session.session.id.to_string())? {
            text.push_str(&prompt.text);
            text.push_str("\n\nStored session/model record:\n");
            expanded.refs.push(format!("message:{}", prompt.message_id));
            let attribution = expanded.attribution.get_or_insert_with(String::new);
            let _ = write!(attribution, "; latest observed request recorded written_by:{} message:{}; older prompts: `polylogue session:{} read --view dialogue`", prompt.written_by, prompt.message_id, session.session.id);
        }
        text.push_str(&expanded.text);
        expanded.text = text;
        Ok(Some(expanded))
    })?;
    let empty = refs.is_empty() && left_out.is_empty() && !with_manual;
    Ok(Prepared {
        event: match source {
            StartSource::Startup | StartSource::Clear => DeliveryEvent::SessionStart,
            StartSource::Resume => DeliveryEvent::Resume,
            StartSource::Compact => DeliveryEvent::Compaction,
        },
        text,
        refs,
        left_out,
        search_summary: None,
        empty,
        next: None,
    })
}
/// One line per session (title, harness, directory, last activity, live
/// state, reference); the budget decides how many lines, and the entry map
/// counts the rest.
fn session_index(
    reader: &Reader<'_>,
    sessions: &[SessionId],
    budget: u64,
    delivery_id: &str,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
) -> Result<Prepared, StoreError> {
    let zone = context.clock.day.zone();
    let now = context.clock.now.as_millisecond();
    let mut items = Vec::with_capacity(sessions.len());
    for session in sessions {
        reader.check()?;
        let id = session.to_string();
        let harness = session.origin().to_string();
        let text = match reader.session(session)? {
            None => missing_session(session),
            Some(found) => {
                let live = reader.live_state(&id, now)?;
                let last = live
                    .as_ref()
                    .and_then(|live| live.last_activity_at)
                    .and_then(Timestamp::from_unix_ms);
                format!(
                    "{} · {harness} · {} · last active {} · {}",
                    found
                        .title
                        .as_ref()
                        .map_or("untitled", |title| title.text.as_str()),
                    found.cwd.last().map_or("directory unknown", String::as_str),
                    day(last, zone),
                    live.map_or("unknown", |live| live.view.state.as_str()),
                )
            }
        };
        items.push(Item {
            expansion: None,
            attribution: Some(format!(
                "archived session metadata; session:{id}; live state at {}",
                day(Some(context.clock.now.into()), zone)
            )),
            text,
            refs: vec![format!("session:{id}")],
        });
    }
    let mut fit = Fitter::new(
        budget,
        format!(
            "[polylogue session index delivery:{delivery_id}: refs in entry map]\nPolylogue index of {} sessions named together (delivery {delivery_id})\n",
            sessions.len()
        ),
    )?;
    fit.section(
        &format!(
            "Times are in {}.\nOne line per session: title, harness, directory, last activity, live state:",
            context.clock.day.zone_name()
        ),
        items,
        "sessions",
        "polylogue REF brief",
    );
    let (text, refs, left_out) =
        fit.finish("\n`polylogue REF brief` for any of these.", |_| Ok(None))?;
    Ok(Prepared {
        event: DeliveryEvent::SessionBrief,
        text,
        refs,
        left_out,
        search_summary: None,
        empty: false,
        next: None,
    })
}

fn missing_session(session: &SessionId) -> String {
    let mut text = format!("session {session} is not in the archive (yet)");
    if let Some(origin) = polylogue_sources::origin::by_name(session.origin().as_str())
        && origin.web.is_some()
    {
        let _ = write!(text, "; capture it with `polylogue import {}`", origin.name);
    }
    text
}

fn session_brief(
    reader: &Reader<'_>,
    session: &SessionId,
    budget: u64,
    delivery_id: &str,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
) -> Result<Prepared, StoreError> {
    let id = session.to_string();
    let Some(s) = reader.session(session)? else {
        return Ok(Prepared {
            event: DeliveryEvent::SessionBrief,
            text: missing_session(session),
            refs: vec![format!("session:{session}")],
            left_out: Vec::new(),
            search_summary: None,
            empty: false,
            next: None,
        });
    };
    let mut deadline = Deadline::default();
    let prompts = deadline
        .section(reader, "operator prompts", || reader.prompts(&id))
        .unwrap_or_default();
    let (messages, parent_issues) = deadline
        .section(reader, "session messages", || {
            let selected = reader.select_messages(session, true)?;
            let mut messages = Vec::new();
            selected.visit(Default::default(), |message| {
                messages.push(message);
                std::ops::ControlFlow::Continue(())
            })?;
            Ok((messages, selected.parents().issues.clone()))
        })
        .unwrap_or_default();
    // The newest assistant message that has text: a trailing tool-only
    // message does not hide the answer before it.
    let last_answer = messages.iter().rev().find_map(|m| {
        if m.role != polylogue_model::Role::Assistant {
            return None;
        }
        let text = m
            .blocks
            .iter()
            .filter_map(|b| match b {
                polylogue_model::Block::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        (!text.trim().is_empty()).then_some(text)
    });
    let notes = deadline
        .section(reader, "operator notes", || {
            operator_notes(
                reader,
                vec![Reference::Session(session.clone())],
                None,
                context,
            )
        })
        .unwrap_or_default();
    let session_reference = Reference::Session(session.clone());
    let other_annotations = deadline
        .section(reader, "annotations by other authors", || {
            unjudged_other_annotations(reader, std::slice::from_ref(&session_reference), context)
        })
        .unwrap_or_default();
    let open_items = deadline
        .section(reader, "open items", || reader.open_items(&id))
        .flatten()
        .unwrap_or_default();

    let mut text = format!("[polylogue takeover delivery:{delivery_id}: refs in entry map]\n");
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
        reason: "a session brief carries prompts, notes and the last answer".into(),
    }];
    if other_annotations > 0 {
        let _ = writeln!(
            text,
            "Annotations by other authors, not yet accepted: {other_annotations}; list them with `polylogue {}`.",
            polylogue_render::shell_quote(&format!(
                "annotations where NOT author:operator judged:unknown about:{}",
                serde_json::to_string(&session_reference.to_string())
                    .expect("reference serializes")
            ))
        );
        left_out.push(LeftOut {
            what: format!("{other_annotations} annotations by other authors"),
            reason: "an annotation by another author needs an eligible operator judgment before injection".into(),
        });
    }
    for issue in &parent_issues {
        let _ = writeln!(text, "Parents incomplete: {issue}");
        left_out.push(LeftOut {
            what: "inherited session content".into(),
            reason: issue.to_string(),
        });
    }
    if open_items.any() {
        let _ = writeln!(text, "Left open: {}.", open_items.phrases().join("; "));
    }
    let note_items = deadline
        .section(reader, "operator note text", || {
            notes
                .iter()
                .map(|note| annotation_item(reader, note, context))
                .collect::<Result<Vec<_>, StoreError>>()
        })
        .unwrap_or_default();
    let correction_prefix = note_items
        .iter()
        .map(|item| item.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let correction_refs: Vec<String> = note_items
        .iter()
        .flat_map(|item| item.refs.iter().cloned())
        .collect();
    let correction_attribution = note_items
        .iter()
        .filter_map(|item| item.attribution.as_deref())
        .collect::<Vec<_>>()
        .join("; ");
    let mut fit = Fitter::new(budget, text)?;
    deadline.show_failures(&mut fit);
    for what in deadline.late {
        fit.pointers
            .push(format!("{what}: not available (deadline)"));
        fit.left_out.push(LeftOut {
            what: what.into(),
            reason: "deadline".into(),
        });
    }
    fit.refs.append(&mut refs);
    fit.left_out.append(&mut left_out);
    let pull = format!("polylogue {id} read");
    fit.section(
        "Notes from the operator on it:",
        note_items,
        "operator notes",
        &pull,
    );
    fit.section(
        &format!(
            "Times are in {}.\nThe operator's prompts in it, oldest first:",
            context.clock.day.zone_name()
        ),
        prompts
            .iter()
            .map(|p| Item {
                expansion: None,
                attribution: Some(format!(
                    "written_by:human; author:operator; session:{id}; message:{}; at:{}; contrary context: {correction_attribution}",
                    p.message_id,
                    day(p.at, context.clock.day.zone())
                )),
                text: format!("{correction_prefix}\n{}", p.text),
                refs: correction_refs.iter().cloned().chain(std::iter::once(format!("message:{}", p.message_id))).collect(),
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
            .map(|text| Item {
                expansion: None,
                attribution: Some(format!(
                    "written_by:model; last answer; contrary context: {correction_attribution}"
                )),
                text: format!("{correction_prefix}\n{text}"),
                refs: correction_refs.clone(),
            })
            .collect(),
        "last answer",
        &pull,
    );
    let (text, refs, left_out) = fit.finish(
        &format!("\nThe full session: `{pull}`; its dialogue: `{pull} --view dialogue`; its changes: `{pull} --view changes`."),
        |_| Ok(None),
    )?;
    Ok(Prepared {
        event: DeliveryEvent::SessionBrief,
        text,
        refs,
        left_out,
        search_summary: None,
        empty: false,
        next: None,
    })
}

/// One message a search brief found.
struct Hit {
    session: String,
    message: String,
    /// The lists that rank it: `prose`, `output`, `semantic`.
    found_in: Vec<&'static str>,
    snippet: Option<String>,
}

/// A search brief's query: messages with every word in prose, every word in
/// tool output and, with `similarity`, every message ranked by similarity
/// to the whole text, ranked together by reciprocal rank fusion; with
/// `directory`, sessions there rank in one more list. Also the name of
/// each list, in `found_in`'s numbering.
fn search_query(
    words: &[String],
    text: &str,
    similarity: bool,
    directory: Option<&str>,
) -> Option<(String, Vec<&'static str>)> {
    let quoted = |prefix: &str| {
        words
            .iter()
            .map(|w| format!("{prefix}{}", annotations::quoted(w)))
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
            format!("messages where similar:{}", annotations::quoted(text)),
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
            annotations::quoted(dir)
        ));
    }
    Some((query, lists.iter().map(|(name, _)| *name).collect()))
}

fn search_brief(
    brief: &BriefContext<'_, '_>,
    text_in: &str,
    cwd: Option<&Path>,
    cursor: Option<&str>,
    budget: u64,
) -> Result<Prepared, StoreError> {
    let BriefContext {
        reader,
        delivery_id,
        context,
        ranking_identity,
        near,
        ..
    } = *brief;
    let words = search::words(text_in);
    let text = text_in.trim();
    let directory = cwd.and_then(|c| scope_dirs(c).pop());
    let mut similarity = near.is_some() && !text.is_empty();
    let mut similarity_failed = None;
    let found = loop {
        let Some((query, indexes)) = search_query(&words, text, similarity, directory.as_deref())
        else {
            break None;
        };
        let answer = crate::daemon::query::run(
            reader,
            crate::daemon::query::Ask {
                text: &query,
                cursor,
                page_size: Some(SEARCH_PAGE),
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
                let rows = page
                    .value
                    .into_rows()
                    .map_err(|f| search_failure(&query, &f))?;
                break Some((rows, indexes));
            }
            // The provider could not embed the text: the words still search.
            Err(f) if similarity && f.code() == "embed" => {
                similarity = false;
                similarity_failed = Some(f.error.message);
            }
            Err(f) => return Err(search_failure(&query, &f)),
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

    let mut out = format!("[polylogue recall delivery:{delivery_id}: refs in entry map]\n");
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
        if !similarity && similarity_failed.is_none() {
            let _ = writeln!(
                out,
                "No exact matches; semantic search is not enabled, so related work under other wording may exist."
            );
        }
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
        let _ = writeln!(out, "\n{sid}  {}", title);
        refs.push(format!("session:{sid}"));
        for h in page.iter().filter(|h| &h.session == sid) {
            let snippet = match &h.snippet {
                Some(s) => s.clone(),
                None => reader.message_text(&h.message)?.unwrap_or_default(),
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
        let notes = annotations::list(
            reader,
            &annotations::about(&[Reference::Session(id.clone())]),
            context,
        )?;
        let mut prefix = String::new();
        let mut attribution = String::from("session context; transcript source is archived data");
        for note in notes {
            let note = annotation_item(reader, &note, context)?;
            let _ = writeln!(prefix, "{}", note.text);
            if let Some(actor) = note.attribution {
                let _ = write!(attribution, "; {actor}");
            }
            refs.extend(note.refs);
        }
        prefix.push_str(&out);
        out = prefix;
        items.push(Item {
            expansion: None,
            attribution: Some(attribution),
            text: out,
            refs,
        });
    }
    let search_summary = SearchSummary {
        words,
        indexes: indexes.into_iter().map(str::to_owned).collect(),
        hits: page.len(),
        sessions: sessions.len(),
        more: next.is_some(),
    };
    let mut closing = String::new();
    if search_summary.indexes.is_empty() {
        let _ = writeln!(
            closing,
            "\nNo search ran: the text has no words to look for."
        );
    } else {
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
                Some(c) => format!(
                    "; more: `polylogue {} brief --cursor {c}`",
                    polylogue_render::shell_quote(text_in.trim())
                ),
                None => String::new(),
            }
        );
    }
    let mut fit = Fitter::new(budget, out)?;
    fit.section(
        "Stored context:",
        items,
        "matching sessions",
        &format!("polylogue {} brief", polylogue_render::shell_quote(text_in)),
    );
    let (out, refs, left_out) = fit.finish(&closing, |_| Ok(None))?;
    Ok(Prepared {
        event: DeliveryEvent::SearchBrief,
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

/// A search brief query's failure keeps the code it failed with.
fn search_failure(query: &str, failure: &command_surface::Failure) -> StoreError {
    StoreError::Selection {
        code: failure
            .code
            .as_deref()
            .and_then(polylogue_model::Code::from_name)
            .unwrap_or(polylogue_model::Code::Internal),
        message: format!("search brief query {query:?}: {}", failure.message),
    }
}
