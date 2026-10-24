use super::*;
use polylogue_model::{Attachment, Block, ToolOutcome};

#[test]
fn unicode_attachment_labels_are_charged_before_splitting_and_described_only_when_visible() {
    let mut message = conversation().remove(0);
    message.blocks = vec![Block::ToolResult {
        call_id: None,
        outcome: ToolOutcome::Ok,
        output: Some("Earlier output without attachment labels. ".repeat(4)),
        data: None,
        attachments: vec![Attachment::default(), Attachment::default()],
    }];
    let label = "label λ🙂字 ".repeat(60);
    let whole = pages(&message, &label, 100000);
    assert_eq!(whole.len(), 1);
    for budget in [1, 3, 17] {
        let pages = pages(&message, &label, budget);
        assert_eq!(
            pages
                .iter()
                .map(|page| page.text.as_str())
                .collect::<String>(),
            whole[0].text
        );
        assert!(
            pages
                .iter()
                .all(|page| page.text.chars().count() <= budget * 4)
        );
        assert!(
            pages[0].attachments.is_empty(),
            "message header alone does not expose attachment metadata"
        );
        assert!(pages.iter().any(|page| {
            page.attachments
                .iter()
                .any(|reference| reference.attachment == 1)
        }));
    }
}

fn pages(message: &polylogue_model::Message, label: &str, budget: usize) -> Vec<Page> {
    let mut result = Vec::new();
    let mut position = None;
    loop {
        let mut pager = Pager::new(View::Full, budget, position);
        let _ = pager
            .push_with_attachments(message.clone(), |_| {
                vec![
                    crate::AttachmentLabel {
                        block: 0,
                        attachment: 0,
                        text: label.to_owned(),
                    },
                    crate::AttachmentLabel {
                        block: 0,
                        attachment: 1,
                        text: "Second attachment".into(),
                    },
                ]
            })
            .unwrap();
        let page = pager.finish().unwrap();
        position = page.next;
        result.push(page);
        if position.is_none() {
            break;
        }
    }
    result
}
