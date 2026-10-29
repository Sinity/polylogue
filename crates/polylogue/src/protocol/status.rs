//! Status answer types shared by discovery, transports and rendering.
use command_surface::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Answer {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archive: Option<PathBuf>,
    pub pid: u32,
    pub uptime_seconds: u64,
    pub store: Option<polylogue_store::Status>,
    pub count_error: Option<String>,
    pub counted: Counted,
    #[serde(flatten)]
    pub loop_status: polylogue_engine::LoopStatus,
    pub signals: Option<Signals>,
    pub notices: Notices,
    pub warm: Warm,
    pub mode: Mode,
    pub currency: polylogue_model::currency::IndexCurrency,
    #[cfg(feature = "sinex")]
    pub primary: Option<Primary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Counted {
    Counting {
        state: Counting,
    },
    Measured {
        position: String,
        current: bool,
        seconds_ago: f64,
        seconds_to_count: f64,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Counting {
    Counting,
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
#[serde(deny_unknown_fields)]
pub struct Signals {
    pub rules: Vec<SignalRule>,
    pub unusable: Vec<UnusableRule>,
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
    pub answers_reused: u64,
    pub answers_computed: u64,
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
        producer: polylogue_sinex::Status,
        applied_position: Option<i64>,
        lag_positions: Option<i64>,
        position_error: Option<String>,
    },
    Unavailable {
        error: String,
    },
}
