use crate::{SessionId, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionKind {
    /// A conversation or agent run the operator started.
    #[default]
    Main,
    /// A session another session spawned (a Claude Code subagent).
    Subagent,
    /// A side conversation kept inside its parent's file.
    Sidechain,
    /// The model call that compacted another session's history (a Claude
    /// Code `agent-acompact` file): the summarize prompt and the summary.
    Compaction,
}

stored_names!(SessionKind {
    Main => "main",
    Subagent => "subagent",
    Sidechain => "sidechain",
    Compaction => "compaction",
});

impl SessionKind {
    /// Sessions another session started. A message copied into one of
    /// these from its parent still belongs to the parent.
    pub fn is_spawned(self) -> bool {
        !matches!(self, SessionKind::Main)
    }
}

/// Where a title came from, highest precedence first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TitleSource {
    /// The operator renamed it.
    Custom,
    /// The tool generated it.
    Generated,
    /// The tool's summary line.
    Summary,
    /// The agent's name, for a spawned session.
    AgentName,
    /// The imported file's name, used when no recorded title is available.
    FileName,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Title {
    pub text: String,
    pub source: TitleSource,
}

/// A session's own facts. Counts and times derived from its messages live
/// in the index, not here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    pub id: SessionId,
    pub kind: SessionKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<Title>,
    /// Working directories in the order the session used them. A later
    /// correction from the tool (Claude Code `relocated`) is appended.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cwd: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_branch: Option<String>,
    /// The repository the session worked in, as the tool names it (a
    /// remote URL), when it records one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Version of the tool that wrote the session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started: Option<Timestamp>,
    /// How the harness ran the session, when not interactively (Claude
    /// Code `bg` for a background session).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

impl Session {
    pub fn new(id: SessionId, kind: SessionKind) -> Self {
        Self {
            id,
            kind,
            title: None,
            cwd: Vec::new(),
            git_branch: None,
            repo: None,
            tool_version: None,
            started: None,
            mode: None,
        }
    }

    /// Keeps the title with the highest-precedence source; ties go to the
    /// later one.
    pub fn offer_title(&mut self, text: &str, source: TitleSource) {
        if text.trim().is_empty() {
            return;
        }
        if self.title.as_ref().is_none_or(|t| source <= t.source) {
            self.title = Some(Title {
                text: text.to_owned(),
                source,
            });
        }
    }

    /// Records a working directory unless it is already the latest one.
    pub fn offer_cwd(&mut self, dir: &str) {
        if !dir.is_empty() && self.cwd.last().map(String::as_str) != Some(dir) {
            self.cwd.push(dir.to_owned());
        }
    }

    pub fn offer_start(&mut self, at: Timestamp) {
        if self.started.is_none_or(|s| at < s) {
            self.started = Some(at);
        }
    }

    /// Merges facts another input (or a later step of the same input) holds
    /// about this session (architecture 5.5a). Idempotent: merging the same
    /// facts twice changes nothing, because a parser resends a session's
    /// facts in full after every step. The kind first recorded stays; the
    /// title goes by precedence; working directories keep their order;
    /// branch and tool version take the newer value; the earliest start wins.
    pub fn merge(&mut self, other: &Session) {
        if let Some(t) = &other.title {
            self.offer_title(&t.text, t.source);
        }
        if other.cwd.starts_with(&self.cwd) {
            self.cwd.clone_from(&other.cwd);
        } else if !self.cwd.starts_with(&other.cwd) {
            for dir in &other.cwd {
                if !self.cwd.contains(dir) {
                    self.cwd.push(dir.clone());
                }
            }
        }
        if other.git_branch.is_some() {
            self.git_branch.clone_from(&other.git_branch);
        }
        if other.repo.is_some() {
            self.repo.clone_from(&other.repo);
        }
        if other.tool_version.is_some() {
            self.tool_version.clone_from(&other.tool_version);
        }
        if let Some(at) = other.started {
            self.offer_start(at);
        }
        if other.mode.is_some() {
            self.mode.clone_from(&other.mode);
        }
    }
}

#[cfg(test)]
mod tests;
