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
    /// Version of the tool that wrote the session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started: Option<Timestamp>,
}

impl Session {
    pub fn new(id: SessionId, kind: SessionKind) -> Self {
        Self {
            id,
            kind,
            title: None,
            cwd: Vec::new(),
            git_branch: None,
            tool_version: None,
            started: None,
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
        if other.tool_version.is_some() {
            self.tool_version.clone_from(&other.tool_version);
        }
        if let Some(at) = other.started {
            self.offer_start(at);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_is_idempotent_and_keeps_the_earliest_start() {
        let mut a = session();
        a.offer_cwd("/a");
        a.offer_start(Timestamp::from_unix_ms(50).unwrap());
        let mut b = session();
        b.offer_cwd("/a");
        b.offer_cwd("/b");
        b.offer_title("t", TitleSource::Generated);
        b.offer_start(Timestamp::from_unix_ms(10).unwrap());
        a.merge(&b);
        let once = a.clone();
        a.merge(&b);
        assert_eq!(a, once);
        assert_eq!(a.cwd, ["/a", "/b"]);
        assert_eq!(a.started, Timestamp::from_unix_ms(10));
        assert_eq!(a.title.unwrap().text, "t");
    }

    #[test]
    fn merge_of_unrelated_directories_appends_new_ones() {
        let mut a = session();
        a.offer_cwd("/x");
        let mut b = session();
        b.offer_cwd("/y");
        b.offer_cwd("/x");
        a.merge(&b);
        assert_eq!(a.cwd, ["/x", "/y"]);
    }

    fn session() -> Session {
        Session::new(
            SessionId::parse("claude-code:s").unwrap(),
            SessionKind::Main,
        )
    }

    #[test]
    fn custom_title_beats_generated_and_summary_in_any_order() {
        let mut s = session();
        s.offer_title("summary", TitleSource::Summary);
        s.offer_title("custom", TitleSource::Custom);
        s.offer_title("generated", TitleSource::Generated);
        assert_eq!(s.title.unwrap().text, "custom");
    }

    #[test]
    fn later_title_of_same_source_wins() {
        let mut s = session();
        s.offer_title("first", TitleSource::Generated);
        s.offer_title("second", TitleSource::Generated);
        assert_eq!(s.title.unwrap().text, "second");
    }

    #[test]
    fn cwd_keeps_order_without_repeats_in_a_row() {
        let mut s = session();
        for d in ["/a", "/a", "/b", "/a"] {
            s.offer_cwd(d);
        }
        assert_eq!(s.cwd, ["/a", "/b", "/a"]);
    }
}
