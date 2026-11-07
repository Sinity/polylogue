//! Live state: what a session is doing now, folded from evidence.
//!
//! Every live fact comes from stored inputs: hook lines (route `hook`),
//! the daemon's process watcher (route `process`), harness registries
//! (route `registry`) and commits seen in a session's worktree (route
//! `git`). [`fold`] applies one piece of evidence to a
//! session's [`LiveFacts`] and returns the intervals it closed; folding the
//! stored evidence of a session in (time, line) order always gives the same
//! facts and the same intervals, so a rebuild reproduces every past state.
//! [`view`] turns facts and `now` into the state shown, with the instant at
//! which it would change with no new evidence. Nothing here reads a clock,
//! prose or shell-command text.

use crate::Json;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub mod channel;

#[cfg(test)]
mod tests;

/// With no process evidence for a session, its hook evidence shows its
/// activity for this long; afterwards the session is `stale`. A transcript
/// that grew this recently with no other evidence makes a session
/// `unknown`.
pub const LIVE_WINDOW_MS: i64 = 15 * 60 * 1000;
/// A tool call open this long is a `long_call` stall.
pub const LONG_CALL_MS: i64 = 30 * 60 * 1000;
/// The live table holds sessions with a live root process or evidence in
/// the last day: a `stale` session whose latest evidence is this old is
/// `ended`, so stale sessions age out rather than accumulate.
pub const STALE_FOR_MS: i64 = 24 * 60 * 60 * 1000;

/// The live state shown for a session at an instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveState {
    Working,
    Waiting,
    Idle,
    Unknown,
    Stale,
    Ended,
}

stored_names!(LiveState {
    Working => "working",
    Waiting => "waiting",
    Idle => "idle",
    Unknown => "unknown",
    Stale => "stale",
    Ended => "ended",
});

impl LiveState {
    /// Working, waiting, idle or unknown: listed as running.
    pub fn running(self) -> bool {
        matches!(
            self,
            Self::Working | Self::Waiting | Self::Idle | Self::Unknown
        )
    }

    /// Peers count sessions working, waiting or idle.
    pub fn active(self) -> bool {
        matches!(self, Self::Working | Self::Waiting | Self::Idle)
    }

    /// The running list's order: waiting first.
    pub fn rank(self) -> u8 {
        match self {
            Self::Waiting => 0,
            Self::Working => 1,
            Self::Idle => 2,
            Self::Unknown => 3,
            Self::Stale => 4,
            Self::Ended => 5,
        }
    }
}

/// What a session's turn is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activity {
    Working,
    Idle,
    Waiting,
}

stored_names!(Activity { Working => "working", Idle => "idle", Waiting => "waiting" });

/// What a waiting session waits for from the operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitingOn {
    Permission,
    Question,
    Plan,
    Input,
}

stored_names!(WaitingOn {
    Permission => "permission",
    Question => "question",
    Plan => "plan",
    Input => "input",
});

/// A session waiting on something other than the operator. Never counts
/// as waiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stall {
    RateLimit,
    ApiError,
    LongCall,
    Queue,
}

stored_names!(Stall {
    RateLimit => "rate_limit",
    ApiError => "api_error",
    LongCall => "long_call",
    Queue => "queue",
});

/// The part of a session's live facts a live interval measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aspect {
    Activity,
    WaitingOn,
    Stall,
}

stored_names!(Aspect { Activity => "activity", WaitingOn => "waiting_on", Stall => "stall" });

/// A process as the kernel names it: pid alone is reused, so a process is
/// its boot, its pid and its start time (field 22 of `/proc/<pid>/stat`,
/// clock ticks after boot).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProcessIdentity {
    pub boot: String,
    pub pid: u32,
    pub start: u64,
}

impl ProcessIdentity {
    /// The identity an evidence line carries in `boot_id`, `pid_key` and
    /// `start_key`; `None` when any part is missing.
    fn from_line(data: &Json, pid_key: &str, start_key: &str) -> Option<Self> {
        Some(Self {
            boot: data.get("boot_id")?.as_str()?.to_owned(),
            pid: u32::try_from(number(data.get(pid_key)?)?).ok()?,
            start: number(data.get(start_key)?)?,
        })
    }
}

fn number(v: &Json) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

/// A value and the instant it began.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Since<T> {
    pub value: T,
    pub since: i64,
}

/// A tool call that started and has not finished.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenCall {
    pub tool: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub at: i64,
}

/// A permission request a session waits on: the tool and its input,
/// which an answer given ahead ("the same tool and input next time")
/// matches exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingRequest {
    pub tool: String,
    #[serde(default, skip_serializing_if = "Json::is_null")]
    pub input: Json,
    pub at: i64,
}

/// A harness registry's own word for a session's status, verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessStatus {
    pub word: String,
    pub at: i64,
}

impl HarnessStatus {
    /// The activity the harness's word states, when it states one. Claude
    /// Code's registry says `busy`, `idle` or `blocked`; Codex's thread
    /// status says `active` or `idle` with flags for waits.
    fn activity(&self) -> Option<(Activity, Option<WaitingOn>)> {
        match self.word.as_str() {
            "busy" | "active" => Some((Activity::Working, None)),
            "idle" => Some((Activity::Idle, None)),
            "blocked" | "waitingOnApproval" => {
                Some((Activity::Waiting, Some(WaitingOn::Permission)))
            }
            "waitingOnUserInput" => Some((Activity::Waiting, Some(WaitingOn::Question))),
            _ => None,
        }
    }
}

/// Everything known about one session's liveness, from its evidence alone.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveFacts {
    /// The harness process the session runs in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<ProcessIdentity>,
    /// When the process watcher saw the root exit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_gone_at: Option<i64>,
    /// When the harness reported the session's end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<i64>,
    /// The turn's activity from hook events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hook_activity: Option<Since<Activity>>,
    /// What the hook events say the session waits for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hook_waiting_on: Option<Since<WaitingOn>>,
    /// The harness registry's latest word, which outranks hook inference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness_status: Option<HarnessStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_call: Option<OpenCall>,
    /// The permission request the session waits on, as the hook saw it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_request: Option<PendingRequest>,
    /// A stall the evidence states (rate limit, API error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stall: Option<Since<Stall>>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub open_subagents: u32,
    /// Files changed through structured edit tools and not yet committed.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub footprint: BTreeSet<String>,
    /// The live task, from explicit task evidence only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    /// The harness's in-progress todo item, as text; never a task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub todo: Option<String>,
    /// The terminal identity the latest hook line carried.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<Json>,
    /// The time of the latest evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_at: Option<i64>,
    /// The intervals open now, one per aspect at most.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_intervals: Vec<OpenInterval>,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// An interval that evidence has opened and not yet closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenInterval {
    pub aspect: Aspect,
    pub value: String,
    pub start: i64,
}

/// A closed interval of one aspect: it depends only on evidence, never on
/// the clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveInterval {
    pub aspect: Aspect,
    pub value: String,
    pub start: i64,
    pub end: i64,
    /// The kind of the evidence that closed it.
    pub closed_by: String,
}

/// The routes live evidence arrives by; an event's kind starts with its
/// route's prefix.
pub const HOOK_PREFIX: &str = "hook_";
pub const PROCESS_PREFIX: &str = "process_";
pub const REGISTRY_PREFIX: &str = "registry_";
pub const GIT_PREFIX: &str = "git_";

/// Whether an event of this kind is live evidence.
pub fn is_evidence(kind: &str) -> bool {
    kind.starts_with(HOOK_PREFIX)
        || kind.starts_with(PROCESS_PREFIX)
        || kind.starts_with(REGISTRY_PREFIX)
        || kind.starts_with(GIT_PREFIX)
}

/// Tools whose structured input names the one file they change.
const EDIT_TOOLS: &[(&str, &str)] = &[
    ("Edit", "/tool_input/file_path"),
    ("Write", "/tool_input/file_path"),
    ("MultiEdit", "/tool_input/file_path"),
    ("NotebookEdit", "/tool_input/notebook_path"),
    ("write_file", "/tool_input/file_path"),
    ("replace", "/tool_input/file_path"),
];

/// Codex's patch format names each changed file on a header line.
const PATCH_HEADERS: &[&str] = &[
    "*** Add File: ",
    "*** Update File: ",
    "*** Delete File: ",
    "*** Move to: ",
];

/// The files a finished tool call changed, from its structured input.
fn edited_paths(data: &Json) -> Vec<String> {
    let tool = data.get("tool_name").and_then(Json::as_str).unwrap_or("");
    if let Some((_, pointer)) = EDIT_TOOLS.iter().find(|(name, _)| *name == tool) {
        return data
            .pointer(pointer)
            .and_then(Json::as_str)
            .map(|p| vec![p.to_owned()])
            .unwrap_or_default();
    }
    if tool == "apply_patch" {
        let patch = data
            .pointer("/tool_input/input")
            .or_else(|| data.pointer("/tool_input/patch"))
            .or_else(|| data.get("tool_input"))
            .and_then(Json::as_str)
            .unwrap_or("");
        return patch
            .lines()
            .filter_map(|line| {
                PATCH_HEADERS
                    .iter()
                    .find_map(|h| line.strip_prefix(h))
                    .map(|p| p.trim().to_owned())
            })
            .filter(|p| !p.is_empty())
            .collect();
    }
    Vec::new()
}

/// The in-progress item of a `TodoWrite` call, as text.
fn todo_in_progress(data: &Json) -> Option<String> {
    if data.get("tool_name").and_then(Json::as_str) != Some("TodoWrite") {
        return None;
    }
    data.pointer("/tool_input/todos")?
        .as_array()?
        .iter()
        .find(|t| t.get("status").and_then(Json::as_str) == Some("in_progress"))
        .and_then(|t| {
            t.get("activeForm")
                .or_else(|| t.get("content"))
                .and_then(Json::as_str)
        })
        .map(str::to_owned)
}

/// The stall a `StopFailure` states, from its structured error type.
fn stop_failure(data: &Json) -> Stall {
    let kind = data
        .get("error_type")
        .or_else(|| data.pointer("/error/type"))
        .and_then(Json::as_str)
        .unwrap_or("");
    if matches!(kind, "rate_limit" | "rate_limit_error") {
        Stall::RateLimit
    } else {
        Stall::ApiError
    }
}

fn continue_open(open: &mut Vec<OpenInterval>, aspect: Aspect, value: Option<&str>, at: i64) {
    if let Some(value) = value {
        open.push(OpenInterval {
            aspect,
            value: value.to_owned(),
            start: at,
        });
    }
}

impl LiveFacts {
    fn set_activity(&mut self, activity: Activity, at: i64) {
        if self.hook_activity.map(|a| a.value) != Some(activity) {
            self.hook_activity = Some(Since {
                value: activity,
                since: at,
            });
        }
    }

    fn set_waiting(&mut self, on: WaitingOn, at: i64) {
        if self.hook_waiting_on.map(|w| w.value) != Some(on) {
            self.hook_waiting_on = Some(Since {
                value: on,
                since: at,
            });
        }
        self.set_activity(Activity::Waiting, at);
    }

    /// The turn moved on: no wait, no stall.
    fn resume_work(&mut self, at: i64) {
        self.hook_waiting_on = None;
        self.pending_request = None;
        self.stall = None;
        self.set_activity(Activity::Working, at);
    }

    fn stop(&mut self, at: i64) {
        self.hook_waiting_on = None;
        self.pending_request = None;
        self.open_call = None;
        self.set_activity(Activity::Idle, at);
    }

    /// Takes the process a hook or registry line names as the root. A new
    /// process for the same session (a native resume) replaces the root
    /// and its exit; a line naming no process leaves the root as it was.
    fn adopt_root(&mut self, identity: Option<ProcessIdentity>) {
        if let Some(identity) = identity
            && self.root.as_ref() != Some(&identity)
        {
            self.root = Some(identity);
            self.root_gone_at = None;
            // The registry word belonged to the earlier process.
            self.harness_status = None;
        }
    }

    /// Whether a line naming `identity` may speak for this session's root:
    /// a line about another process (pid reuse, another boot) never does.
    fn same_root(&self, identity: Option<&ProcessIdentity>) -> bool {
        match (self.root.as_ref(), identity) {
            (Some(root), Some(identity)) => root == identity,
            _ => true,
        }
    }

    fn apply_hook(&mut self, event: &str, at: i64, data: &Json) {
        self.adopt_root(ProcessIdentity::from_line(
            data,
            "harness_pid",
            "harness_start",
        ));
        if let Some(terminal) = data.get("terminal_identity") {
            self.terminal = Some(terminal.clone());
        }
        match event {
            "session_start" => {
                self.ended_at = None;
                self.stop(at);
            }
            "user_prompt_submit" | "before_agent" | "pre_llm_call" => self.resume_work(at),
            "pre_tool_use" | "before_tool" => {
                self.resume_work(at);
                let tool = data.get("tool_name").and_then(Json::as_str).unwrap_or("");
                self.open_call = Some(OpenCall {
                    tool: tool.to_owned(),
                    call_id: data
                        .get("tool_use_id")
                        .and_then(Json::as_str)
                        .map(str::to_owned),
                    at,
                });
                match tool {
                    "AskUserQuestion" => self.set_waiting(WaitingOn::Question, at),
                    "ExitPlanMode" => self.set_waiting(WaitingOn::Plan, at),
                    _ => {}
                }
                if let Some(todo) = todo_in_progress(data) {
                    self.todo = Some(todo);
                }
            }
            "post_tool_use" | "post_tool_use_failure" | "after_tool" => {
                self.resume_work(at);
                self.open_call = None;
                if event != "post_tool_use_failure" {
                    self.footprint.extend(edited_paths(data));
                }
            }
            "permission_request" => {
                self.set_waiting(WaitingOn::Permission, at);
                self.pending_request = Some(PendingRequest {
                    tool: data
                        .get("tool_name")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_owned(),
                    input: data.get("tool_input").cloned().unwrap_or(Json::Null),
                    at,
                });
            }
            "permission_denied" | "elicitation_result" => self.resume_work(at),
            "elicitation" => self.set_waiting(WaitingOn::Question, at),
            "notification" => match data.get("notification_type").and_then(Json::as_str) {
                Some("permission_prompt") => self.set_waiting(WaitingOn::Permission, at),
                Some("elicitation_dialog") => self.set_waiting(WaitingOn::Question, at),
                Some("idle_prompt") => self.stop(at),
                _ => {}
            },
            "stop" | "after_agent" => {
                self.stop(at);
                if data.get("ending").and_then(Json::as_str) == Some("question") {
                    self.set_waiting(WaitingOn::Question, at);
                }
            }
            "stop_failure" => {
                self.stop(at);
                self.stall = Some(Since {
                    value: stop_failure(data),
                    since: at,
                });
            }
            "subagent_start" => {
                self.open_subagents += 1;
                self.resume_work(at);
            }
            "subagent_stop" => self.open_subagents = self.open_subagents.saturating_sub(1),
            "session_end" => {
                self.ended_at = Some(at);
                self.open_call = None;
                self.pending_request = None;
                self.open_subagents = 0;
                self.footprint.clear();
            }
            "task_claimed" => {
                self.task = data.get("task").and_then(Json::as_str).map(str::to_owned);
            }
            "task_released" if self.task.as_deref() == data.get("task").and_then(Json::as_str) => {
                self.task = None;
            }
            _ => {}
        }
    }

    fn apply_process(&mut self, event: &str, at: i64, data: &Json) {
        if event == "root_gone"
            && let Some(identity) = ProcessIdentity::from_line(data, "pid", "start")
            && self.root.as_ref() == Some(&identity)
            && self.root_gone_at.is_none()
        {
            // A root gone mid-turn is stale, not ended: its footprint stays
            // until a commit takes the files or the session ends.
            self.root_gone_at = Some(at);
            self.open_call = None;
        }
    }

    /// A commit in the session's worktree, with the files it contains:
    /// those leave the footprint, since they are no longer uncommitted.
    fn apply_git(&mut self, event: &str, data: &Json) {
        if event != "commit" {
            return;
        }
        for path in data
            .get("files")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
            .filter_map(Json::as_str)
        {
            self.footprint.remove(path);
        }
    }

    fn apply_registry(&mut self, event: &str, at: i64, data: &Json) {
        let identity = ProcessIdentity::from_line(data, "pid", "start");
        if event.starts_with("hosted_") {
            // Polylogue's own host started the harness process, so its
            // lines name the root with certainty.
            self.adopt_root(identity);
        } else if !self.same_root(identity.as_ref()) {
            // A registry reader names the process that owns the pid when it
            // reads the file, so its line never chooses the root; it only
            // speaks for a root the hooks established, or for none.
            return;
        }
        if event == "hosted_ended" {
            // The host saw the harness process exit after it ended its
            // last turn: the session ended.
            self.ended_at = Some(at);
            self.open_call = None;
            self.pending_request = None;
            self.open_subagents = 0;
            self.footprint.clear();
            return;
        }
        if let Some(word) = data.get("status").and_then(Json::as_str) {
            self.harness_status = Some(HarnessStatus {
                word: word.to_owned(),
                at,
            });
        }
    }

    /// The activity and wait shown while the session's process lives: the
    /// registry's word when it states one, else the hooks'.
    fn shown(&self) -> Option<(Activity, Option<WaitingOn>)> {
        if self.ended_at.is_some() || self.root_gone_at.is_some() {
            return None;
        }
        if let Some((activity, waiting_on)) =
            self.harness_status.as_ref().and_then(|s| s.activity())
        {
            // The hooks may say more precisely what a blocked session waits for.
            let waiting_on =
                waiting_on.map(|registry| self.hook_waiting_on.map_or(registry, |hook| hook.value));
            return Some((activity, waiting_on));
        }
        let activity = self.hook_activity.map(|a| a.value)?;
        let waiting_on = self.hook_waiting_on.map(|w| w.value);
        Some((
            activity,
            waiting_on.filter(|_| activity == Activity::Waiting),
        ))
    }

    /// Closes and opens intervals so the open set matches what is shown now.
    fn settle_intervals(&mut self, at: i64, closed_by: &str) -> Vec<LiveInterval> {
        let shown = self.shown();
        let wanted: [(Aspect, Option<&'static str>); 3] = [
            (Aspect::Activity, shown.map(|(a, _)| a.as_str())),
            (
                Aspect::WaitingOn,
                shown.and_then(|(_, w)| w).map(WaitingOn::as_str),
            ),
            (
                Aspect::Stall,
                self.stall
                    .filter(|_| shown.is_some())
                    .map(|s| s.value.as_str()),
            ),
        ];
        let mut closed = Vec::new();
        for (aspect, value) in wanted {
            let open = self.open_intervals.iter().position(|o| o.aspect == aspect);
            let current = open.map(|i| self.open_intervals[i].value.as_str());
            if current == value {
                continue;
            }
            if let Some(i) = open {
                let o = self.open_intervals.remove(i);
                // Evidence at one instant that passes through a value
                // measures no time and leaves no interval.
                if o.start == at {
                    continue_open(&mut self.open_intervals, aspect, value, at);
                    continue;
                }
                closed.push(LiveInterval {
                    aspect,
                    value: o.value,
                    start: o.start,
                    end: at,
                    closed_by: closed_by.to_owned(),
                });
            }
            if let Some(value) = value {
                self.open_intervals.push(OpenInterval {
                    aspect,
                    value: value.to_owned(),
                    start: at,
                });
            }
        }
        self.open_intervals.sort_by_key(|o| o.aspect as u8);
        closed
    }
}

/// Applies one piece of evidence (an event of `kind` at `at` with `data`)
/// and returns the intervals it closed. Evidence of other kinds changes
/// nothing.
pub fn fold(facts: &mut LiveFacts, kind: &str, at: i64, data: &Json) -> Vec<LiveInterval> {
    if let Some(event) = kind.strip_prefix(HOOK_PREFIX) {
        if event == "context_emitted" {
            return Vec::new();
        }
        facts.apply_hook(event, at, data);
    } else if let Some(event) = kind.strip_prefix(PROCESS_PREFIX) {
        facts.apply_process(event, at, data);
    } else if let Some(event) = kind.strip_prefix(REGISTRY_PREFIX) {
        facts.apply_registry(event, at, data);
    } else if let Some(event) = kind.strip_prefix(GIT_PREFIX) {
        facts.apply_git(event, data);
    } else {
        return Vec::new();
    }
    facts.last_at = Some(facts.last_at.map_or(at, |last| last.max(at)));
    facts.settle_intervals(at, kind)
}

/// The live state at an instant and what is shown with it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LiveView {
    pub state: LiveState,
    /// When the state began, as far as the evidence says.
    pub since: Option<i64>,
    /// The next instant at which the state or the stall changes with no
    /// new evidence; `None` when only new evidence can change them.
    pub until: Option<i64>,
    pub waiting_on: Option<WaitingOn>,
    pub stall: Option<Stall>,
    /// Whether the session's process is known alive (`Some(true)`), known
    /// gone (`Some(false)`), or unobserved (`None`).
    pub process_alive: Option<bool>,
}

/// The view of `facts` at `now`. `grew_at` is when the session's
/// transcript last grew, which alone can make a session `unknown`. Total:
/// every input gives a view, and between `until` instants with no new
/// evidence the view is constant. Evidence dated after `now` counts as at
/// `now`.
pub fn view(facts: Option<&LiveFacts>, grew_at: Option<i64>, now: i64) -> LiveView {
    let ended = |since| LiveView {
        state: LiveState::Ended,
        since,
        until: None,
        waiting_on: None,
        stall: None,
        process_alive: None,
    };
    let Some(facts) = facts.filter(|f| f.last_at.is_some()) else {
        // Only the transcript's growth: the inference by time that remains.
        return match grew_at.map(|g| g.min(now)) {
            Some(g) if now - g <= LIVE_WINDOW_MS => LiveView {
                state: LiveState::Unknown,
                since: Some(g),
                until: Some(g + LIVE_WINDOW_MS + 1),
                waiting_on: None,
                stall: None,
                process_alive: None,
            },
            Some(_) => ended(None),
            // Nothing recorded a time: nothing says it ended.
            None => LiveView {
                state: LiveState::Unknown,
                ..ended(None)
            },
        };
    };
    let process_alive = facts.root.as_ref().map(|_| facts.root_gone_at.is_none());
    if let Some(at) = facts.ended_at {
        return LiveView {
            process_alive,
            ..ended(Some(at))
        };
    }
    let last = facts.last_at.unwrap_or(i64::MIN).min(now);
    // A stale session is ended once its latest evidence is a day old.
    let aged_out = last.saturating_add(STALE_FOR_MS);
    let stale = |since: i64| {
        if now >= aged_out {
            // No evidence says when it ended; its latest evidence is when
            // it was last known to run.
            LiveView {
                process_alive,
                ..ended(Some(last))
            }
        } else {
            LiveView {
                state: LiveState::Stale,
                since: Some(since),
                until: Some(aged_out),
                waiting_on: None,
                stall: None,
                process_alive,
            }
        }
    };
    if let Some(gone) = facts.root_gone_at {
        // Gone after a stop is an end; gone mid-turn is stale.
        let stopped = facts.hook_activity.map(|a| a.value) == Some(Activity::Idle)
            && facts.hook_waiting_on.is_none();
        if stopped {
            return LiveView {
                process_alive,
                ..ended(Some(gone))
            };
        }
        return stale(gone);
    }
    // Without process evidence, hook evidence holds only for the window.
    let window_end = facts.root.is_none().then(|| last + LIVE_WINDOW_MS + 1);
    if let Some(end) = window_end.filter(|end| now >= *end) {
        return stale(end);
    }
    let shown = facts.shown();
    let state = match shown.map(|(a, _)| a) {
        Some(Activity::Waiting) => LiveState::Waiting,
        Some(Activity::Working) => LiveState::Working,
        Some(Activity::Idle) => LiveState::Idle,
        None => LiveState::Unknown,
    };
    let since = facts
        .open_intervals
        .iter()
        .find(|o| o.aspect == Aspect::Activity)
        .map(|o| o.start);
    let long_call_at = facts
        .open_call
        .as_ref()
        .filter(|_| state == LiveState::Working)
        .map(|c| c.at + LONG_CALL_MS);
    let stall = facts.stall.map(|s| s.value).or_else(|| {
        long_call_at
            .filter(|at| now >= *at)
            .map(|_| Stall::LongCall)
    });
    let until = [window_end, long_call_at.filter(|at| now < *at)]
        .into_iter()
        .flatten()
        .min();
    LiveView {
        state,
        since,
        until,
        waiting_on: shown.and_then(|(_, w)| w),
        stall,
        process_alive,
    }
}
