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

fn joined_with_checkpoints(messages: &[Message], view: View, budget: usize) -> String {
    let mut position = None;
    let mut page_end: Option<PageEnd> = None;
    let mut out = String::new();
    for _ in 0..10000 {
        let from = page_end.as_ref().map_or(0, PageEnd::message);
        let mut pager = match (position, page_end.take()) {
            (Some(at), Some(saved)) => Pager::resume(view, budget, at, saved),
            _ => Pager::new(view, budget, position),
        };
        for message in &messages[from..] {
            if pager.push(message.clone()).unwrap().is_break() {
                break;
            }
        }
        page_end = pager.page_end();
        let page = pager.finish().unwrap();
        out.push_str(&page.text);
        position = page.next;
        if position.is_none() {
            return out;
        }
    }
    panic!("page_end pages made no progress");
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
            assert_eq!(joined_with_checkpoints(&messages, view, budget), expected);
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
                reported_duration_ms: None,
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
            run_by_script: false,
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

#[test]
fn structured_slices_reconstruct_masked_blocks_with_source_indices_at_every_budget() {
    use crate::tests::message;
    use polylogue_model::{Block, Role};
    let text = " λ🙂\n\n   visible <script> ".repeat(100);
    let messages = vec![message(
        "slice",
        Role::Assistant,
        None,
        vec![
            Block::Thinking {
                text: Some("reason".into()),
                redacted: false,
                signature: None,
            },
            Block::Text { text: text.clone() },
        ],
    )];
    for budget in [1, 7, 4096] {
        let mut position = None;
        let mut collected = String::new();
        let mut offset = 0;
        loop {
            let mut pager = Pager::new(View::Full, budget, position);
            for message in &messages {
                if pager.push(message.clone()).unwrap().is_break() {
                    break;
                }
            }
            let page = pager.finish().unwrap();
            for slice in page.slices.iter().filter(|s| s.kind == "text") {
                assert_eq!(slice.block, 1);
                assert_eq!(slice.message.as_ref().unwrap().id, messages[0].id);
                assert_eq!(slice.from, offset);
                assert_eq!(slice.until - slice.from, slice.text.chars().count());
                offset = slice.until;
                collected.push_str(&slice.text);
            }
            position = page.next;
            if position.is_none() {
                break;
            }
        }
        assert_eq!(collected, text);
    }
}

#[test]
fn one_large_tool_result_is_reachable_without_embedding_it_in_one_page() {
    use crate::tests::message;
    use polylogue_model::{Block, Role, ToolOutcome};
    let text = "large output λ🙂\n".repeat(65536);
    let message = message(
        "large",
        Role::Assistant,
        None,
        vec![Block::ToolResult {
            reported_duration_ms: None,
            call_id: None,
            outcome: ToolOutcome::Ok,
            output: Some(text.clone()),
            data: None,
            attachments: vec![],
        }],
    );
    let mut position = None;
    let mut collected = String::new();
    let mut pages = 0;
    loop {
        let mut pager = Pager::new(View::Full, 8192, position);
        let _ = pager.push(message.clone()).unwrap();
        let page = pager.finish().unwrap();
        assert!(
            page.slices
                .iter()
                .map(|s| s.text.chars().count())
                .sum::<usize>()
                <= 32768
        );
        for slice in &page.slices {
            collected.push_str(&slice.text);
        }
        position = page.next;
        pages += 1;
        if position.is_none() {
            break;
        }
    }
    assert!(pages > 20);
    assert_eq!(collected, text);
}

#[test]
fn latest_selects_a_bounded_tail_after_masking_all_source_blocks() {
    use crate::tests::message;
    use polylogue_model::{Block, Role};
    let source = message(
        "tail",
        Role::Assistant,
        None,
        vec![
            Block::Text {
                text: "-----BEGIN PRIVATE KEY-----hidden-first".into(),
            },
            Block::Text {
                text: format!(
                    "hidden-tail-----END PRIVATE KEY----- {}visible-tail λ🙂",
                    "visible ".repeat(1000)
                ),
            },
        ],
    );
    let mut pager = Pager::new(View::Full, 16, None).at_latest();
    let _ = pager.push(source.clone()).unwrap();
    let position = pager.first_position().unwrap();
    let page = pager.finish().unwrap();
    assert!(page.text.chars().count() <= 64);
    assert!(page.text.ends_with("visible-tail λ🙂\n"));
    assert!(!page.text.contains("hidden"));
    assert_eq!(page.slices[0].block, 1);
    assert!(page.slices[0].from > 0);
    assert!(page.next.is_none());
    let mut reopened = Pager::resume(
        View::Full,
        16,
        position,
        PageEnd::new(0, MessageMasker::default()),
    );
    let _ = reopened.push(source).unwrap();
    assert_eq!(reopened.finish().unwrap(), page);
}
