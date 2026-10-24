use super::*;
use crate::model_pages::{BlockPager, Position};
use polylogue_model::{Block, DocumentId, DocumentPurpose};
use std::ops::ControlFlow;

fn source_document() -> Document {
    let mut doc = Document::new(
        DocumentId::parse("claude-ai:doc:a").unwrap(),
        DocumentPurpose::Memory,
    );
    doc.blocks = vec![
        Block::ToolResult {
            call_id: None,
            outcome: polylogue_model::ToolOutcome::Ok,
            output: Some("-----BEGIN PRIVATE KEY-----\n".into()),
            data: None,
            attachments: Vec::new(),
        },
        Block::Text {
            text: "synthetic-secret-tail\n-----END PRIVATE KEY-----\nVisible λ🙂".into(),
        },
    ];
    doc
}
#[test]
fn role_free_formats_mask_before_hidden_block_projection_and_page_splitting() {
    let doc = source_document();
    let structured = document(doc.clone(), View::Full, Format::Json).unwrap();
    assert!(!structured.to_string().contains("synthetic-secret-tail"));
    assert!(structured["document"].get("role").is_none());
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        let value = document(doc.clone(), View::Dialogue, format).unwrap();
        assert!(!value.to_string().contains("synthetic-secret-tail"));
        assert!(value.to_string().contains("Visible λ🙂"));
    }
    let mut position = None;
    let mut text = String::new();
    loop {
        let mut pager = BlockPager::new(View::Dialogue, 2, position, "[memory]\n".into());
        for block in doc.blocks.clone() {
            if pager.push(block).unwrap() == ControlFlow::Break(()) {
                break;
            }
        }
        let page = pager.finish().unwrap();
        text.push_str(&page.text);
        position = page.next;
        if position.is_none() {
            break;
        }
    }
    assert!(!text.contains("synthetic-secret-tail"));
    assert!(text.contains("Visible λ🙂"));
    let mut pager = BlockPager::new(
        View::Full,
        2,
        Some(Position {
            message: 99,
            piece: 0,
            character: 0,
        }),
        String::new(),
    );
    for block in doc.blocks {
        let _ = pager.push(block).unwrap();
    }
    assert!(pager.finish().is_err());
}
