//! Real typed read delivery, independent of the retired rendered-session wire.
use super::*;
use polylogue_model::read::{FragmentEncoding, ItemKind, ReadPage};

fn typed_page(output: &Output) -> ReadPage {
    assert!(output.status.success(), "{output:?}");
    serde_json::from_value(json(output)).unwrap()
}
fn words(page: &ReadPage) -> String {
    page.document
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Text { text }
            | ItemKind::Fragment {
                encoding: FragmentEncoding::Text,
                text,
                ..
            } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn typed_read_real_cli_pages_unicode_once_and_publishes_a_canonical_bundle() {
    let directory = tempfile::tempdir().unwrap();
    let home = directory.path().join("home");
    let archive = directory.path().join("archive");
    let config = directory.path().join("config");
    let fixture = crate::Fixture::new(&archive, &home, &config);
    let project = home.join(".claude/projects/-typed-read");
    std::fs::create_dir_all(&project).unwrap();
    let expected = "A synthetic λ🙂 prompt across source and transport boundaries.";
    std::fs::write(project.join("typed.jsonl"), line("u1", "typed", expected)).unwrap();
    let _daemon = start_daemon(&fixture);
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        let output = cli(&fixture, &["--json", "status"]);
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
            "--budget",
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
            "--budget",
            "1",
        ],
    );
    let mut reconstructed = String::new();
    for page_number in 0..1000 {
        assert!(
            output.stdout.len() <= 25_000,
            "page exceeds the transfer bound"
        );
        let page = typed_page(&output);
        reconstructed.push_str(&words(&page));
        let Some(next) = page.document.footer.next else {
            break;
        };
        assert!(page_number < 999, "continuation failed to reach the end");
        let request = serde_json::to_string(&next.http.arguments).unwrap();
        output = cli(&fixture, &["--json", "read", "--request", &request]);
    }
    assert_eq!(reconstructed, expected);
    let destination = directory.path().join("bundle");
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
    assert!(delivered.status.success(), "{delivered:?}");
    let records = std::fs::read_to_string(destination.join("records.jsonl")).unwrap();
    let mut values = records
        .lines()
        .map(|line| serde_json::from_str::<polylogue_model::bundle::Record>(line).unwrap());
    assert!(matches!(
        values.next(),
        Some(polylogue_model::bundle::Record::Manifest(_))
    ));
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
}
