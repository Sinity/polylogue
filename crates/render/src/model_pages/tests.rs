use super::*;
use crate::tests::conversation;

mod attachments;

fn all_pages(budget: usize) -> Vec<Page> {
    let ms = conversation();
    let mut pages = vec![];
    let mut c = None;
    loop {
        let mut pager = Pager::new(View::Full, budget, c);
        for m in &ms {
            if pager.push(m.clone()).unwrap().is_break() {
                break;
            }
        }
        let p = pager.finish().unwrap();
        c = p.next;
        pages.push(p);
        if c.is_none() {
            break;
        }
        assert!(pages.len() < 10_000, "pages must make progress");
    }
    pages
}

#[test]
fn pages_cover_the_whole_session_once_at_any_budget() {
    let whole = all_pages(1_000_000);
    assert_eq!(whole.len(), 1);
    let full = whole[0].text.clone();
    assert!(full.contains("→ Bash: cargo test"));
    assert!(full.contains("← error (exit 101)"));
    assert!(
        !full.contains("sk-ant-api03-AAAA"),
        "masked before measuring"
    );
    for budget in [1, 2, 5, 13, 40] {
        let joined: String = all_pages(budget).into_iter().map(|p| p.text).collect();
        assert_eq!(joined, full, "budget {budget}");
    }
}

#[test]
fn a_page_stays_within_its_budget() {
    for p in all_pages(10) {
        assert!(p.text.chars().count() <= 40, "{:?}", p.text);
        assert!(!p.text.is_empty());
    }
}

fn joined(messages: &[Message], view: View, budget: usize) -> String {
    let mut position = None;
    let mut out = String::new();
    for _ in 0..10000 {
        let mut pager = Pager::new(view, budget, position);
        for message in messages {
            if pager.push(message.clone()).unwrap().is_break() {
                break;
            }
        }
        let page = pager.finish().unwrap();
        assert!(page.text.chars().count() <= budget.max(1).saturating_mul(4));
        out.push_str(&page.text);
        position = page.next;
        if position.is_none() {
            return out;
        }
    }
    panic!("pages made no progress");
}

#[test]
fn pages_preserve_utf8_authorship_views_and_mask_pem_across_blocks_and_messages() {
    use crate::tests::message;
    use polylogue_model::{Block, PromptKind, Role};
    let messages = vec![
        message(
            "1",
            Role::Assistant,
            Some(PromptKind::Harness),
            vec![Block::Text {
                text: "λ🙂 prefix -----BEGIN PRIVATE KEY-----private-one".into(),
            }],
        ),
        message(
            "2",
            Role::Assistant,
            None,
            vec![Block::Text {
                text: "private-two-----END PRIVATE KEY----- visible λ🙂".into(),
            }],
        ),
        message(
            "3",
            Role::User,
            Some(PromptKind::Model),
            vec![Block::Text {
                text: "model prompt λ🙂".into(),
            }],
        ),
    ];
    let whole = joined(&messages, View::Full, usize::MAX);
    assert!(whole.contains("[harness]"));
    assert!(whole.contains("[model]"));
    assert!(whole.contains("visible λ🙂"));
    assert!(!whole.contains("private-one"));
    assert!(!whole.contains("private-two"));
    for view in [View::Full, View::Dialogue, View::Prompts, View::Actions] {
        let expected = joined(&messages, view, usize::MAX);
        for budget in [0, 1, 2, 5, 19] {
            assert_eq!(joined(&messages, view, budget), expected);
        }
    }
}

#[test]
fn a_hidden_block_still_masks_a_visible_continuation_of_a_private_key() {
    use crate::tests::message;
    use polylogue_model::{Block, Role, ToolOutcome};
    let messages = vec![message(
        "1",
        Role::Assistant,
        None,
        vec![
            Block::ToolResult {
                call_id: None,
                outcome: ToolOutcome::Ok,
                output: Some("-----BEGIN PRIVATE KEY-----hidden".into()),
                data: None,
                attachments: vec![],
            },
            Block::Text {
                text: "secret-tail-----END PRIVATE KEY----- visible".into(),
            },
        ],
    )];
    let output = joined(&messages, View::Dialogue, 1);
    assert!(!output.contains("secret-tail"));
    assert!(output.contains("visible"));
}

#[test]
fn masking_context_continues_from_a_tool_label_into_its_body() {
    use crate::tests::message;
    use polylogue_model::{Block, Role, ToolAction};
    let messages = vec![message(
        "1",
        Role::Assistant,
        None,
        vec![Block::ToolCall {
            call_id: None,
            tool: Some("-----BEGIN PRIVATE KEY-----".into()),
            action: ToolAction::Shell,
            input: serde_json::json!({}),
            command: Some("private-tail-----END PRIVATE KEY----- visible".into()),
            path: None,
        }],
    )];
    for budget in [1, usize::MAX] {
        let output = joined(&messages, View::Full, budget);
        assert!(!output.contains("private-tail"));
        assert!(output.contains("visible"));
    }
}
