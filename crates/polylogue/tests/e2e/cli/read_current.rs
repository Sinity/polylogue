//! Real typed read delivery, independent of the retired rendered-session wire.
use super::*;
use polylogue_model::BlobHash;
use polylogue_model::read::{ItemKind, ReadItem, ReadPage, SelectionLocator};
use std::collections::BTreeMap;

fn typed_page(output: &Output) -> ReadPage {
    assert!(output.status.success(), "{output:?}");
    serde_json::from_value(json(output)).unwrap()
}
#[derive(Default)]
pub(super) struct Items {
    values: Vec<ReadItem>,
    partial: BTreeMap<(SelectionLocator, u64, BlobHash), (String, u64)>,
}
impl Items {
    pub(super) fn push(&mut self, page: &ReadPage) {
        for item in &page.document.items {
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
fn typed_read_real_cli_pages_unicode_once_and_publishes_a_canonical_bundle() {
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
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        let output = cli(&fixture, &["--deadline", "none", "--json", "status"]);
        assert!(output.status.success(), "{output:?}");
        let status = json(&output);
        if status["store"]["messages"] == 1 && status["busy"] == false {
            break;
        }
        assert!(
            Instant::now() < until,
            "synthetic source never became ready: {status}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let target = "session:claude-code:typed";
    let whole = typed_page(&cli(
        &fixture,
        &[
            "--json",
            "read",
            target,
            "--view",
            "transcript",
            "--page-tokens",
            "10000",
        ],
    ));
    assert_eq!(words(&whole), expected);
    assert!(whole.records.is_empty());
    let mut output = cli(
        &fixture,
        &[
            "--json",
            "read",
            target,
            "--view",
            "transcript",
            "--page-tokens",
            "1",
        ],
    );
    let mut reconstructed = Items::default();
    for page_number in 0..1000 {
        assert!(
            output.stdout.len() <= 25_000,
            "page exceeds the transfer bound"
        );
        let page = typed_page(&output);
        reconstructed.push(&page);
        let Some(next) = page.document.footer.next else {
            break;
        };
        assert!(page_number < 999, "continuation failed to reach the end");
        let stages = polylogue_model::command::effective_stages(&next.cli);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].program, "polylogue");
        let mut arguments = vec!["--json"];
        arguments.extend(stages[0].args.iter().map(String::as_str));
        output = cli(&fixture, &arguments);
    }
    assert_eq!(reconstructed.text(), expected);
    let destination = directory.as_path().join("bundle");
    std::fs::create_dir(&destination).unwrap();
    let delivered = cli(
        &fixture,
        &[
            "--json",
            "read",
            target,
            "--view",
            "transcript",
            "--format",
            "jsonl",
            "--to",
            destination.to_str().unwrap(),
        ],
    );
    assert!(
        matches!(delivered.status.code(), Some(0 | 5)),
        "{delivered:?}"
    );
    let records = std::fs::read_to_string(destination.join("records.jsonl")).unwrap();
    let mut values = records
        .lines()
        .map(|line| serde_json::from_str::<polylogue_model::bundle::Record>(line).unwrap());
    let Some(polylogue_model::bundle::Record::Manifest(manifest)) = values.next() else {
        panic!("manifest must lead the bundle");
    };
    manifest.validate().unwrap();
    eprintln!(
        "bundle snapshot: positions={:?}, caught_up={}, currency={:?}, incomplete={:?}, coverage={:?}",
        manifest.read_at,
        manifest.caught_up,
        manifest.currency,
        manifest.incomplete,
        manifest.coverage
    );
    let degraded = !manifest.caught_up
        || manifest.currency.degraded()
        || !manifest.incomplete.is_empty()
        || !manifest.coverage.is_empty();
    assert_eq!(delivered.status.code(), Some(if degraded { 5 } else { 0 }));

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
