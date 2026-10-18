//! Semantic cards: a compact summary of a session built from outcomes
//! (old `ap7`): what was run and how it ended, which files were written,
//! the last failure. Facts of the records only; nothing here reads meaning
//! into the text (the "code selects and structures" rule).

use crate::{is_prompt, outcome_label};
use polylogue_model::{Block, Message, Session, ToolOutcome};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Card {
    pub title: String,
    pub lines: Vec<String>,
}

pub fn card(session: &Session, messages: &[Message]) -> Card {
    let title = session
        .title
        .as_ref()
        .map(|t| t.text.clone())
        .unwrap_or_else(|| session.id.to_string());
    let prompts = messages.iter().filter(|m| is_prompt(m)).count();
    let mut calls = 0usize;
    let mut tools: BTreeMap<String, usize> = BTreeMap::new();
    let mut commands: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut written: Vec<String> = vec![];
    let mut outcomes: BTreeMap<String, &ToolOutcome> = BTreeMap::new();
    let mut call_command: BTreeMap<String, String> = BTreeMap::new();
    let mut last_failure: Option<String> = None;

    for m in messages {
        for b in &m.blocks {
            match b {
                Block::ToolCall {
                    call_id,
                    tool,
                    command,
                    path,
                    ..
                } => {
                    calls += 1;
                    let tool = tool.clone().unwrap_or_else(|| "tool".into());
                    *tools.entry(tool.clone()).or_default() += 1;
                    if let Some(c) = command {
                        let key = c.split_whitespace().take(2).collect::<Vec<_>>().join(" ");
                        commands.entry(key.clone()).or_default().0 += 1;
                        if let Some(id) = call_id {
                            call_command.insert(id.clone(), key);
                        }
                    }
                    if let Some(p) = path
                        && matches!(
                            tool.as_str(),
                            "Write" | "Edit" | "MultiEdit" | "apply_patch" | "NotebookEdit"
                        )
                        && !written.contains(p)
                    {
                        written.push(p.clone());
                    }
                }
                Block::ToolResult {
                    call_id: Some(id),
                    outcome,
                    ..
                } => {
                    outcomes.insert(id.clone(), outcome);
                    if matches!(outcome, ToolOutcome::Error { .. }) {
                        let what = call_command.get(id).cloned().unwrap_or_else(|| id.clone());
                        last_failure = Some(format!("{what}: {}", outcome_label(outcome)));
                        if let Some(c) = call_command.get(id) {
                            commands.entry(c.clone()).or_default().1 += 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let mut lines = vec![format!(
        "{} messages, {prompts} prompts, {calls} tool calls",
        messages.len()
    )];
    if !tools.is_empty() {
        let mut t: Vec<(String, usize)> = tools.into_iter().collect();
        t.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        lines.push(format!(
            "tools: {}",
            t.iter()
                .take(6)
                .map(|(n, c)| format!("{n} {c}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !commands.is_empty() {
        let mut c: Vec<(String, (usize, usize))> = commands.into_iter().collect();
        c.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(a.0.cmp(&b.0)));
        lines.push(format!(
            "ran: {}",
            c.iter()
                .take(6)
                .map(|(n, (runs, failed))| if *failed > 0 {
                    format!("{n} ×{runs} ({failed} failed)")
                } else {
                    format!("{n} ×{runs}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !written.is_empty() {
        let shown: Vec<&str> = written.iter().take(5).map(String::as_str).collect();
        let more = written.len().saturating_sub(shown.len());
        lines.push(format!(
            "wrote: {}{}",
            shown.join(", "),
            if more > 0 {
                format!(" and {more} more")
            } else {
                String::new()
            }
        ));
    }
    let unanswered = messages
        .iter()
        .flat_map(|m| &m.blocks)
        .filter(|b| matches!(b, Block::ToolCall { call_id: Some(id), .. } if !outcomes.contains_key(id)))
        .count();
    if unanswered > 0 {
        lines.push(format!("{unanswered} calls without a result"));
    }
    if let Some(f) = last_failure {
        lines.push(format!("last failure: {f}"));
    }
    Card { title, lines }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::conversation;
    use polylogue_model::{SessionId, SessionKind};

    #[test]
    fn a_card_counts_outcomes_not_words() {
        let mut s = Session::new(
            SessionId::parse("claude-code:s").unwrap(),
            SessionKind::Main,
        );
        s.offer_title("Fix the tests", polylogue_model::TitleSource::Summary);
        let c = card(&s, &conversation());
        assert_eq!(c.title, "Fix the tests");
        assert_eq!(c.lines[0], "3 messages, 1 prompts, 1 tool calls");
        assert!(c.lines.contains(&"tools: Bash 1".to_owned()));
        assert!(
            c.lines
                .contains(&"ran: cargo test ×1 (1 failed)".to_owned()),
            "{:?}",
            c.lines
        );
        assert!(
            c.lines
                .contains(&"last failure: cargo test: error (exit 101)".to_owned())
        );
    }
}
