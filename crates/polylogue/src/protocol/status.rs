//! Status answer types shared by discovery, transports and rendering.
use command_surface::JsonSchema;
use polylogue_render::short_duration;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    /// One line for status bars: running sessions and the longest wait
    #[serde(default)]
    pub line: bool,
    /// Count stored vectors per model (reads the vector store)
    #[serde(default)]
    pub vectors: bool,
    /// Check the stored bytes against their hashes (CLI only)
    #[serde(default)]
    pub verify: bool,
    /// Write the checked revisions to this manifest
    pub manifest: Option<PathBuf>,
    /// Every log record the daemon keeps (200), not only the last 20
    #[serde(default)]
    pub log: bool,
    /// Include per-pattern capture counts and unread source fields
    #[serde(default)]
    pub details: bool,
    /// Redraw the view every 2 seconds until interrupted (CLI on a color terminal only)
    #[serde(default)]
    pub watch: bool,
    /// Answer once the daemon has caught up, or when this much time has passed (30s, 10m)
    pub wait: Option<String>,
    /// With --wait, answer once it has caught up after this many changes (the `changes` of an earlier answer)
    pub after_changes: Option<u64>,
}

/// How many log records an ordinary status answer carries.
pub const RECENT_LOG: usize = 20;

/// How long a SQL statement runs before status lists it.
pub const STATEMENTS_AFTER: std::time::Duration = std::time::Duration::from_secs(3);

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Answer {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archive: Option<PathBuf>,
    pub pid: u32,
    pub uptime_seconds: u64,
    /// What the archive's figures in this reply come from.
    pub answers_from: AnswersFrom,
    /// Changes to what status reports since the daemon started: the loop's
    /// work, and the snapshot read after a commit.
    pub changes: u64,
    /// The daemon has finished the work it has: no index being built, no
    /// pass running or due, derived figures and search text current, and
    /// this answer read after the latest commit.
    pub caught_up: bool,
    /// The archive totals, from the snapshot; absent when none is read
    /// (`answers_from`) or it could not be read (`totals_error`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<Store>,
    /// The rebuild of `index.db` in progress, with its reason and counts.
    /// Read with `build_progress`, so both are present or neither.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_build: Option<polylogue_store::archive::index_build::IndexBuild>,
    /// Which index files exist and what they mean: the installed index
    /// (`none`, `current` or `unusable` with why) and the rebuild (`no`,
    /// `writing` with its reason, or `ready`), with a failed install's error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<polylogue_store::archive::index_build::IndexStatus>,
    /// The build's steps, rates and estimate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_progress: Option<polylogue_store::archive::build_progress::BuildProgress>,
    /// Why the snapshot holds no archive totals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub totals_error: Option<String>,
    /// Each subscription limit's current window, from the snapshot.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quota: Vec<polylogue_store::measure::quota::CurrentWindow>,
    /// Why the snapshot holds no quota windows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quota_error: Option<String>,
    /// Exact stored counts, requested with --vectors or --verify.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vectors: Option<polylogue_store::vectors::VectorStatus>,
    /// Why requested vector counts could not be read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vectors_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification: Option<polylogue_store::verify::Verification>,
    pub pending_writes: PendingWrites,
    /// Session starts without context after the latest successful start.
    pub session_starts_without_context: Option<MissingContext>,
    #[serde(flatten)]
    pub loop_status: polylogue_engine::LoopStatus,
    pub signals: Option<Signals>,
    /// Absent until a snapshot is read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notices: Option<Notices>,
    /// Absent until a snapshot is read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub running: Option<Running>,
    pub warm: Warm,
    pub mode: Mode,
    pub freshness: polylogue_model::freshness::ReadFreshness,
    /// The daemon's latest log records, oldest first: the last
    /// [`RECENT_LOG`], or every kept one with `--log`.
    #[serde(default)]
    pub recent_log: Vec<polylogue_store::log::Record>,
    /// Each daemon thread that is doing something, with what, for how long
    /// and how long since it last made progress, and the SQL statements
    /// running longer than [`STATEMENTS_AFTER`], by thread name.
    #[serde(default)]
    pub threads: Vec<polylogue_store::activity::Thread>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slow_statements: Vec<polylogue_store::activity::SlowStatement>,
}

/// What a status reply's archive figures come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AnswersFrom {
    /// The status snapshot read at `read_at`. An index build beside a
    /// readable index is shown with it; the figures are the installed
    /// index's.
    Snapshot { read_at: polylogue_model::Timestamp },
    /// No index can be read while one is built: `build_progress` is the
    /// content.
    Build,
    /// No snapshot has been read since `since`.
    NotReadYet { since: NotReadSince },
}

polylogue_model::stored_names! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum NotReadSince {
        /// The daemon started.
        Start => "start",
        /// A built index was installed.
        Install => "install",
    }
}

impl AnswersFrom {
    /// The one wording of a reply without figures, used by the reply's note,
    /// the view's headline and `--line`; none for a snapshot, and none for a
    /// build, which the build's progress describes.
    pub fn words(&self) -> Option<&'static str> {
        match self {
            Self::Snapshot { .. } | Self::Build => None,
            Self::NotReadYet {
                since: NotReadSince::Start,
            } => Some("figures not read yet since the daemon started"),
            Self::NotReadYet {
                since: NotReadSince::Install,
            } => Some("figures not read yet since the new index was installed"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct MissingContext {
    pub since: Option<polylogue_model::Timestamp>,
    pub causes: std::collections::BTreeMap<String, u64>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PendingWrites {
    pub delivery_outcomes: u64,
    pub delivery_write_failures: u64,
    pub mailbox_failures: u64,
    pub host_reattachment_failures: u64,
    pub live_event_lines: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum Mode {
    Standalone,
    Blocked {
        configured: Option<String>,
        path: PathBuf,
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum Signals {
    Observed {
        rules: Vec<SignalRule>,
        unusable: Vec<UnusableRule>,
        #[serde(default)]
        held: Vec<HeldSignal>,
    },
    Held {
        held: Vec<HeldSignal>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HeldSignal {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<command_surface::Answer<()>>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SignalRule {
    pub id: String,
    pub name: String,
    pub query: String,
    pub author: String,
    /// When a notice from the rule was last produced; none if it never fired.
    pub last_fired: Option<polylogue_model::Timestamp>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UnusableRule {
    pub id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Notices {
    Measured {
        last_day: usize,
        latest: Vec<Notice>,
    },
    Unavailable {
        error: String,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Notice {
    pub id: String,
    pub at: polylogue_model::Timestamp,
    pub text: String,
    pub events: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Warm {
    /// Whether the scope index briefs answer from has been read since the
    /// daemon started.
    pub scope_index_loaded: bool,
    pub vectors: Option<Vectors>,
    pub vector_loads: u64,
    pub query_vectors_reused: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Vectors {
    pub key: String,
    pub vectors: usize,
}

#[cfg(test)]
mod tests;

/// The sessions running now, waiting first; or why they could not be read.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Running {
    Measured {
        /// How many of them wait on the operator.
        waiting: usize,
        sessions: Vec<polylogue_store::RunningSession>,
    },
    Unavailable {
        error: String,
    },
}

impl Running {
    /// The status line's running part: `3 running · 1 waiting 7m`, the
    /// wait being the longest; empty with nothing running.
    pub fn line(&self, now_ms: i64) -> String {
        let Self::Measured { waiting, sessions } = self else {
            return "running unavailable".into();
        };
        if sessions.is_empty() {
            return String::new();
        }
        let mut line = format!("{} running", sessions.len());
        if *waiting > 0 {
            line.push_str(&format!(" · {waiting} waiting"));
            let longest = sessions
                .iter()
                .filter(|s| s.state == polylogue_model::live::LiveState::Waiting)
                .filter_map(|s| s.since)
                .min();
            if let Some(since) = longest {
                line.push(' ');
                line.push_str(&short_duration(now_ms.saturating_sub(since)));
            }
        }
        line
    }
}

/// Archive figures with capture summaries for the status reply.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct Store {
    /// Visible logical sessions with a transcript.
    pub sessions: u64,
    /// Visible physical sessions with a transcript, counted separately.
    #[serde(default)]
    pub sessions_physical: u64,
    pub messages: u64,
    pub documents: u64,
    pub tool_calls: u64,
    pub inputs: u64,
    pub revisions: u64,
    /// Bytes of input content stored, decoded: what the inputs hold, not
    /// what the segment log takes on disk.
    pub stored_bytes: u64,
    /// Input chains by how they are shown: `parsing`, `repeated`, `no_parser`,
    /// `done`, `waiting`, `refused`, or `failed`.
    pub parse_states: BTreeMap<String, u64>,
    /// The 20 most frequent unread fields, with --details only.
    pub unused: Vec<(String, u64)>,
    /// How many record kinds and fields parsers did not read, all of them.
    pub unused_fields: u64,
    /// Texts stored but not yet in the search index: how far search lags
    /// the writer.
    pub search_pending: u64,
    /// The count stopped at its bound. More texts are waiting when this is true.
    pub search_pending_is_minimum: bool,
    /// Sessions waiting for their summaries to be folded.
    pub summaries_due: u64,
    /// The latest confirmed full discovery walk from each source machine.
    pub capture_coverage: Vec<Coverage>,
    /// Why this index replaced the one before it, as its rebuild recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_rebuild: Option<polylogue_store::index::file::IndexRebuild>,
}

/// The last scan on one machine.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Coverage {
    pub machine: String,
    pub observed_at: polylogue_model::Timestamp,
    pub roots: Vec<Root>,
}

/// One watched root, with pattern counts only when requested.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Root {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_bytes: Option<Vec<u8>>,
    pub status: polylogue_model::capture::Accessibility,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub collected: u64,
    pub ignored: u64,
    pub unrecognized: u64,
    pub unreadable: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entries: Vec<polylogue_model::capture::Entry>,
}

impl Store {
    pub fn from_snapshot(
        store: polylogue_store::Status,
        details: bool,
        errors: &[polylogue_model::Failure],
        machine: &str,
    ) -> Self {
        use polylogue_model::capture::Classification;
        Self {
            sessions: store.sessions,
            sessions_physical: store.sessions_physical,
            messages: store.messages,
            documents: store.documents,
            tool_calls: store.tool_calls,
            inputs: store.inputs,
            revisions: store.revisions,
            stored_bytes: store.stored_bytes,
            parse_states: store.parse_states,
            unused: if details { store.unused } else { Vec::new() },
            unused_fields: store.unused_fields,
            search_pending: store.search_pending,
            search_pending_is_minimum: store.search_pending_is_minimum,
            summaries_due: store.summaries_due,
            capture_coverage: store
                .capture_coverage
                .into_iter()
                .map(|coverage| {
                    let errors = if coverage.machine == machine {
                        errors
                    } else {
                        &[]
                    };
                    Coverage {
                        machine: coverage.machine,
                        observed_at: coverage.observed_at,
                        roots: coverage
                            .roots
                            .into_iter()
                            .map(|root| {
                                let path = String::from_utf8_lossy(&root.path_bytes).into_owned();
                                let count = |class| {
                                    root.entries
                                        .iter()
                                        .filter(|entry| entry.class == class)
                                        .map(|entry| entry.files)
                                        .sum()
                                };
                                Root {
                                    error: errors
                                        .iter()
                                        .find(|error| {
                                            error.subject == format!("source root {path}")
                                        })
                                        .map(ToString::to_string),
                                    path,
                                    path_bytes: std::str::from_utf8(&root.path_bytes)
                                        .is_err()
                                        .then_some(root.path_bytes),
                                    status: root.status,
                                    collected: count(Classification::Collected),
                                    ignored: count(Classification::Ignored),
                                    unrecognized: count(Classification::Unrecognized),
                                    unreadable: root.unavailable_paths.len(),
                                    entries: if details { root.entries } else { Vec::new() },
                                }
                            })
                            .collect(),
                    }
                })
                .collect(),
            index_rebuild: store.index_rebuild,
        }
    }
}
