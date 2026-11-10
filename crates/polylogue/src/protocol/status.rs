//! Status answer types shared by discovery, transports and rendering.
use command_surface::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    /// Print one line for status bars instead of the whole answer.
    #[serde(default)]
    pub line: bool,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Answer {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archive: Option<PathBuf>,
    pub pid: u32,
    pub uptime_seconds: u64,
    /// The archive totals; none when they could not be read (`totals_error`).
    pub store: Option<polylogue_store::Status>,
    pub totals_error: Option<String>,
    /// Why the last check of whether the index caught up failed.
    pub caught_up_error: Option<String>,
    #[serde(flatten)]
    pub loop_status: polylogue_engine::LoopStatus,
    pub signals: Option<Signals>,
    pub notices: Notices,
    pub running: Running,
    pub warm: Warm,
    pub mode: Mode,
    pub currency: polylogue_model::currency::IndexCurrency,
    #[cfg(feature = "sinex")]
    pub primary: Option<Primary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum Mode {
    Standalone,
    Primary,
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

#[cfg(feature = "sinex")]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Primary {
    Observed {
        #[serde(flatten)]
        producer: Box<polylogue_sinex::Status>,
        applied_position: Option<i64>,
        lag_positions: Option<i64>,
        position_error: Option<String>,
        /// Confirmed events Polylogue could not read; they stay in Sinex.
        refused_events: u64,
        last_refused: Option<String>,
    },
    Unavailable {
        error: String,
    },
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

/// `45s`, `7m`, `3h12m`, `2d4h`.
pub fn short_duration(ms: i64) -> String {
    let s = ms.max(0) / 1000;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m", s / 60),
        3600..86400 => format!("{}h{}m", s / 3600, s % 3600 / 60),
        _ => format!("{}d{}h", s / 86400, s % 86400 / 3600),
    }
}
