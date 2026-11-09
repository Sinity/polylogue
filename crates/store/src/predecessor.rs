//! The Python predecessor's own stores, each a declared older input format
//! of `polylogue import --format NAME PATH`: its `user.db` of authored
//! records ([`user_db`]) and its vector stores ([`vectors`]). Its source
//! catalog (`source.db` with `blob/`) is read by [`crate::legacy_import`].
//! Each store is only read: opened read-only and immutable, or, when its
//! write-ahead log holds frames, copied with the log into the importing
//! archive's folder and the copy read.

pub mod user_db;
pub mod vectors;

/// The origin here of each origin name the predecessor stored.
pub fn origin(name: &str) -> Option<&'static str> {
    Some(match name {
        "claude-code-session" => "claude-code",
        "codex-session" => "codex",
        "gemini-cli-session" => "gemini-cli",
        "hermes-session" => "hermes",
        "antigravity-session" => "antigravity",
        "grok-export" => "grok",
        "chatgpt-export" => "chatgpt",
        "claude-ai-export" => "claude-ai",
        "claude-design-session" => "claude-design",
        "aistudio-drive" => "ai-studio",
        "otel-genai" => "otel-genai",
        _ => return None,
    })
}
#[cfg(any(test, feature = "fixture"))]
pub mod fixture;
