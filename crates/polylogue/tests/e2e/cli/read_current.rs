//! Real typed read delivery, independent of the retired rendered-session wire.
use super::*;
use polylogue_model::BlobHash;
use polylogue_model::read::{ItemKind, ReadItem, ReadPage, SelectionLocator};
use std::collections::BTreeMap;

fn typed_page(output: &Output) -> ReadPage {
    assert!(output.status.success(), "{output:?}");
    read_page(json(output))
}
#[derive(Default)]
pub(super) struct Items {
    values: Vec<ReadItem>,
    partial: BTreeMap<(SelectionLocator, u64, BlobHash), (String, u64)>,
}
impl Items {
    pub(super) fn push(&mut self, page: &ReadPage) {
        self.push_document(&page.document);
    }
    pub(super) fn push_document(&mut self, document: &polylogue_model::read::ReadDocument) {
        for item in &document.items {
            if let ItemKind::Fragment {
                text,
                position,
                until,
                total,
            } = &item.kind
            {
                let key = (position.selection, position.item, position.item_hash);
                let (encoded, characters) = self.partial.entry(key).or_default();
                assert_eq!(
                    position.character, *characters,
                    "fragment gap or repeated prefix"
                );
                assert!(*until > *characters && *until <= *total);
                assert_eq!(*until - *characters, text.chars().count() as u64);
                encoded.push_str(text);
                *characters = *until;
                if until == total {
                    let (encoded, _) = self.partial.remove(&key).unwrap();
                    assert_eq!(BlobHash::of(encoded.as_bytes()), position.item_hash);
                    let item: ReadItem = serde_json::from_str(&encoded).unwrap();
                    assert!(!matches!(item.kind, ItemKind::Fragment { .. }));
                    assert!(!self.values.contains(&item), "duplicate source item");
                    self.values.push(item);
                }
            } else {
                assert!(!self.values.contains(item), "duplicate source item");
                self.values.push(item.clone());
            }
        }
    }
    pub(super) fn into_values(self) -> Vec<ReadItem> {
        assert!(
            self.partial.is_empty(),
            "delivery ended within an encoded item"
        );
        self.values
    }
    pub(super) fn text(&self) -> String {
        assert!(
            self.partial.is_empty(),
            "delivery ended within an encoded item"
        );
        self.values
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
}
fn words(page: &ReadPage) -> String {
    let mut items = Items::default();
    items.push(page);
    items.text()
}

#[test]
fn typed_read_pages_unicode_once_and_writes_a_bundle() {
    let directory = tempfile::tempdir().unwrap().keep();
    eprintln!("typed read fixture: {}", directory.display());
    let home = directory.as_path().join("home");
    let archive = directory.as_path().join("archive");
    let config = directory.as_path().join("config");
    let fixture = crate::Fixture::new(&archive, &home, &config);
    let project = home.join(".claude/projects/-typed-read");
    std::fs::create_dir_all(&project).unwrap();
    let expected = "A synthetic λ🙂 prompt across source and transport boundaries.";
    std::fs::write(project.join("typed.jsonl"), line("u1", "typed", expected)).unwrap();
    let _daemon = start_daemon(&fixture);
    wait_status(&fixture, |status| status["store"]["messages"] == 1);
    let target = "session:claude-code:typed";
    let whole = typed_page(&cli(
        &fixture,
        &[
            "--json",
            target,
            "read",
            "--view",
            "transcript",
            "--budget",
            "10000",
        ],
    ));
    assert_eq!(words(&whole), expected);
    let output = cli(
        &fixture,
        &[
            "--json",
            target,
            "read",
            "--view",
            "transcript",
            "--budget",
            "1",
        ],
    );
    let mut reconstructed = Items::default();
    let mut answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&output.stdout).unwrap();
    for page_number in 0..1000 {
        assert!(
            serde_json::to_vec(&answer).unwrap().len() < 25_000,
            "page exceeds the transfer bound"
        );
        assert_eq!(answer.exit_code(), 0, "{answer:?}");
        let page = read_page(answer.value().unwrap().clone());
        reconstructed.push(&page);
        let Some(next) = page.document.footer.next else {
            break;
        };
        assert!(page_number < 999, "continuation failed to reach the end");
        answer = follow(&fixture, &next);
    }
    assert_eq!(reconstructed.text(), expected);
    let destination = directory.as_path().join("bundle");
    std::fs::create_dir(&destination).unwrap();
    wait_caught_up(&fixture);
    let delivered = cli(
        &fixture,
        &[
            "--json",
            target,
            "read",
            "--view",
            "transcript",
            "--format",
            "jsonl",
            "--to",
            destination.to_str().unwrap(),
        ],
    );
    assert_eq!(delivered.status.code(), Some(0), "{delivered:?}");
    let records = std::fs::read_to_string(destination.join("records.jsonl")).unwrap();
    let mut values = records
        .lines()
        .map(|line| serde_json::from_str::<polylogue_model::bundle::Record>(line).unwrap());
    let Some(polylogue_model::bundle::Record::Manifest(manifest)) = values.next() else {
        panic!("manifest must lead the bundle");
    };
    manifest.validate().unwrap();
    assert!(manifest.freshness.up_to_date(), "{manifest:?}");
    assert!(manifest.incomplete.is_empty(), "{manifest:?}");
    assert!(manifest.coverage.is_empty(), "{manifest:?}");

    let mut messages = 0;
    for record in values {
        if let polylogue_model::bundle::Record::Message(record) = record {
            record.validate().unwrap();
            assert_eq!(
                record.message.blocks,
                vec![polylogue_model::Block::Text {
                    text: expected.into()
                }]
            );
            messages += 1;
        }
    }
    assert_eq!(messages, 1);
    drop(_daemon);
    std::fs::remove_dir_all(directory).unwrap();
}
