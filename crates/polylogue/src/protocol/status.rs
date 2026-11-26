//! Status answer types shared by discovery, transports and rendering.
use command_surface::JsonSchema;
use polylogue_render::short_duration;
use serde::{Deserialize, Serialize};
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
    /// Redraw the view every 2 seconds until interrupted (CLI on a color terminal only)
    #[serde(default)]
    pub watch: bool,
    /// Answer once the daemon has caught up, or when this much time has passed (30s, 10m)
    pub wait: Option<String>,
    /// With --wait, answer once it has caught up after this many changes (the `changes` of an earlier answer)
    pub after: Option<u64>,
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
    /// The first reads that status and briefs answer from are not ready yet.
    pub warming: bool,
    /// Changes to what status reports since the daemon started: the loop's
    /// work, and the snapshot read after a commit.
    pub changes: u64,
    /// The daemon has finished the work it has: no index being built, no
    /// pass running or due, derived figures and search text current, and
    /// this answer read after the latest commit.
    pub caught_up: bool,
    /// The archive totals; none when they could not be read (`totals_error`).
    pub store: Option<polylogue_store::Status>,
    /// The build of `index.db` in progress, with its progress.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_build: Option<polylogue_store::archive::index_build::IndexBuild>,
    /// The build's phase, rates and estimate, beside its counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_progress: Option<polylogue_store::archive::build_progress::BuildProgress>,
    pub totals_error: Option<String>,
    /// Exact stored counts, requested with --vectors or --verify.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vectors: Option<polylogue_store::vectors::VectorStatus>,
    /// Why requested vector counts could not be read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vectors_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification: Option<polylogue_store::verify::Verification>,
    pub pending_writes: PendingWrites,
    #[serde(flatten)]
    pub loop_status: polylogue_engine::LoopStatus,
    pub signals: Option<Signals>,
    pub notices: Notices,
    pub running: Running,
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
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PendingWrites {
    pub delivery_outcomes: u64,
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
