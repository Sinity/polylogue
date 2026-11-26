//! What the session summaries read from a parsed message or event, taken
//! once from the parsed value: a summary folds these heads in session order
//! and never decodes stored content to find them. Each is one pure function
//! of the value, so the head of a message read back from the index equals
//! the head of the message the parser produced.

use polylogue_model::{Block, CompactionBoundary, Event, Message, MicrocompactionBoundary};
use serde::{Deserialize, Serialize};

/// Characters kept of a prompt's first line as a title.
const TITLE_LENGTH: usize = 80;

/// What the summaries read from one message.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageHead {
    /// The title the message gives its session when it is the session's
    /// first prose prompt a person typed: the first line of its first text
    /// block. `None` when it has no text block, or when that block is a
    /// context-forked subagent's instructions, which name nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The artifacts each tool result reports, for the call that made it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub results: Vec<ResultEvidence>,
}

/// The commits and pull requests a tool result's output reports. The
/// links summary takes them when the result's call ran `git commit` or
/// `gh pr create`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultEvidence {
    /// The result's block among the message's blocks.
    pub block: u32,
    pub call_id: Option<String>,
    /// Hashes of `[branch abc1234] subject` lines, lowercased.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commits: Vec<String>,
    /// Pull request URLs: `(owner/repo, number, url)`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pulls: Vec<(String, i64, String)>,
}

/// What the summaries read from one event.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EventHead {
    /// The data of a live event with a time (hook, process, registry and git
    /// lines), which the live summary folds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub live: Option<polylogue_model::Json>,
    /// A compaction's boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<CompactionBoundary>,
    /// The calls a microcompaction cleared.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed_calls: Vec<String>,
}

/// The head of a message.
pub fn message_head(message: &Message) -> MessageHead {
    let title = message
        .blocks
        .iter()
        .find_map(|block| match block {
            Block::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .and_then(prompt_title);
    let results = message
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(n, block)| {
            let Block::ToolResult {
                call_id, output, ..
            } = block
            else {
                return None;
            };
            let output = output.as_deref().unwrap_or("");
            let evidence = ResultEvidence {
                block: n as u32,
                call_id: call_id.clone(),
                commits: commits_in(output),
                pulls: pulls_in(output),
            };
            (!evidence.commits.is_empty() || !evidence.pulls.is_empty()).then_some(evidence)
        })
        .collect();
    MessageHead { title, results }
}

/// The head of an event. Data that does not have its kind's shape gives no
/// boundary, as a summary reading it would find none.
pub fn event_head(event: &Event) -> EventHead {
    EventHead {
        live: (event.at.is_some() && polylogue_model::live::is_live_event(&event.kind))
            .then(|| event.data.clone()),
        compaction: (event.kind == "compaction")
            .then(|| serde_json::from_value::<CompactionBoundary>(event.data.clone()).ok())
            .flatten(),
        removed_calls: if event.kind == "microcompaction" {
            serde_json::from_value::<MicrocompactionBoundary>(event.data.clone())
                .map(|boundary| boundary.removed_calls)
                .unwrap_or_default()
        } else {
            Vec::new()
        },
    }
}

/// The title a prompt gives: its first non-empty line, cut after 80
/// characters with an ellipsis; none for the harness's instructions to a
/// context-forked subagent.
pub(crate) fn prompt_title(prompt: &str) -> Option<String> {
    if prompt.trim_start().starts_with("<fork-boilerplate>") {
        return None;
    }
    let line = prompt.lines().map(str::trim).find(|l| !l.is_empty())?;
    let mut t: String = line.chars().take(TITLE_LENGTH).collect();
    if line.chars().count() > TITLE_LENGTH {
        t.push('…');
    }
    Some(t)
}

/// `[branch abc1234] subject` lines of `git commit` output.
pub(crate) fn commits_in(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|l| {
            let inner = l.trim().strip_prefix('[')?.split_once(']')?.0;
            let hash = inner.rsplit(' ').next()?;
            (hash.len() >= 7 && hash.chars().all(|c| c.is_ascii_hexdigit()))
                .then(|| hash.to_ascii_lowercase())
        })
        .collect()
}

/// Pull request URLs in `gh pr create` output: `(owner/repo, number, url)`.
/// The URL is the pull request's own, so a link does not depend on which
/// mention of it (a `/files` suffix, trailing punctuation) was read first.
pub(crate) fn pulls_in(output: &str) -> Vec<(String, i64, String)> {
    output
        .split_whitespace()
        .filter_map(|w| {
            let rest = w.strip_prefix("https://github.com/")?;
            let mut parts = rest.split('/');
            let (owner, repo, pull, number) =
                (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
            let n: i64 = number
                .trim_end_matches(|c: char| !c.is_ascii_digit())
                .parse()
                .ok()?;
            (pull == "pull").then(|| {
                let repo = format!("{owner}/{repo}");
                let url = format!("https://github.com/{repo}/pull/{n}");
                (repo, n, url)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;
