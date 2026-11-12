use super::*;
use polylogue_model::{Block, Document, DocumentId, DocumentPurpose, Timestamp, ToolOutcome};

#[test]
fn document_pages_preserve_exact_prefix_and_refuse_cross_content_stale_and_tampered_continuations()
{
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let mut doc = Document::new(
        DocumentId::parse("claude-ai:manual").unwrap(),
        DocumentPurpose::Context,
    );
    doc.set_time(
        Timestamp::from_unix_ms(1),
        polylogue_model::TimeEvidence::intrinsic(polylogue_model::Precision::Instant),
    )
    .unwrap();
    doc.blocks = vec![
        Block::ToolResult {
            reported_duration_ms: None,
            call_id: None,
            outcome: ToolOutcome::Ok,
            output: Some("-----BEGIN PRIVATE KEY-----\n".into()),
            data: None,
            attachments: Vec::new(),
        },
        Block::Text {
            text: "synthetic-secret-tail\n-----END PRIVATE KEY-----\nVisible λ🙂 tail".into(),
        },
        Block::Text {
            text: "Later block".into(),
        },
        Block::Text {
            text: "<script>alert('synthetic')</script>\u{1b}[31m\u{7}".into(),
        },
    ];
    let mut other = doc.clone();
    other.id = DocumentId::parse("claude-ai:other").unwrap();
    let source = dir.path().join("documents.json");
    write_documents(&source, &[doc.clone(), other]);
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let _daemon = start_daemon(&env);
    let imported = cli(&env, &["--json", "import", source.to_str().unwrap()]);
    let _ = import_report(&imported);
    let reference = "document:claude-ai:manual";
    let whole = reading(&env, reference, &["--page-tokens", "100000"]);
    let encoded = serde_json::to_string(&whole).unwrap();
    assert!(encoded.contains("synthetic-secret-tail"));
    assert!(encoded.contains("Visible λ🙂 tail"));
    assert_eq!(joined_items(&env, reference, "1"), whole.document.items);
    assert_eq!(
        document(&env, reference),
        serde_json::to_value(&doc).unwrap()
    );
    let terminal = cli(&env, &["read", reference, "--format", "terminal"]);
    assert!(terminal.status.success(), "{terminal:?}");
    assert!(!terminal.stdout.contains(&0x1b) && !terminal.stdout.contains(&7));
    assert!(
        String::from_utf8(terminal.stdout)
            .unwrap()
            .contains("synthetic-secret-tail")
    );
    let html_path = dir.path().join("document.html");
    let html = cli(
        &env,
        &[
            "--json",
            "read",
            reference,
            "--format",
            "html",
            "--to",
            html_path.to_str().unwrap(),
        ],
    );
    assert!(matches!(html.status.code(), Some(0 | 5)), "{html:?}");
    let html = std::fs::read_to_string(html_path).unwrap();
    assert!(!html.contains("<script>alert('synthetic')</script>"));
    assert!(html.contains("&lt;script&gt;") && html.contains("synthetic-secret-tail"));
    let first = reading(&env, reference, &["--page-tokens", "1"]);
    let next = first.document.footer.next.unwrap();
    let args = next_args(&next);
    let cursor = args.iter().position(|arg| arg == "--cursor").unwrap() + 1;
    assert!(!args[cursor].is_empty());
    assert!(!args[cursor].contains("synthetic-secret-tail"));
    let other_page = reading(&env, "document:claude-ai:other", &["--page-tokens", "1"]);
    let other_args = next_args(&other_page.document.footer.next.unwrap());
    let other_cursor = other_args.iter().position(|arg| arg == "--cursor").unwrap() + 1;
    for token in [
        other_args[other_cursor].clone(),
        "broken".into(),
        String::new(),
        format!("{}x", args[cursor]),
    ] {
        let mut changed = args.clone();
        changed[cursor] = token;
        let result = cli(
            &env,
            &changed.iter().map(String::as_str).collect::<Vec<_>>(),
        );
        assert!(!result.status.success(), "{result:?}");
    }
    let notes = serde_json::to_string(&[polylogue_model::assertion::NewAssertion {
        kind: polylogue_model::assertion::AssertionKind::Note,
        key: None,
        value: Some(json!("Unrelated write")),
        valid_until: None,
        links: vec![polylogue_model::assertion::NewAssertionLink {
            role: polylogue_model::assertion::LinkRole::About,
            reference: polylogue_model::Reference::parse("document:claude-ai:other").unwrap(),
        }],
    }])
    .unwrap();
    let note = cli(&env, &["--json", "assert", &notes]);
    assert!(note.status.success(), "{note:?}");
    let continued = cli(&env, &args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(continued.status.success(), "{continued:?}");
    let index = rusqlite::Connection::open(archive.join("index.db")).unwrap();
    index.execute("UPDATE document_block SET body=x'00' WHERE idx=2 AND document_pk=(SELECT pk FROM document WHERE id='claude-ai:manual')",[]).unwrap();
    drop(index);
    let prefix = reading(&env, reference, &["--page-tokens", "1"]);
    assert!(!prefix.document.items.is_empty());
    let full = cli(
        &env,
        &["--json", "read", reference, "--page-tokens", "100000"],
    );
    assert!(!full.status.success(), "{full:?}");
    doc.set_time(
        Timestamp::from_unix_ms(2),
        polylogue_model::TimeEvidence::intrinsic(polylogue_model::Precision::Instant),
    )
    .unwrap();
    doc.blocks = vec![Block::Text {
        text: "A changed document version".into(),
    }];
    let changed = dir.path().join("changed.json");
    write_documents(&changed, &[doc]);
    let imported = cli(&env, &["--json", "import", changed.to_str().unwrap()]);
    let _ = import_report(&imported);
    let stale = cli(&env, &args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(!stale.status.success(), "{stale:?}");
    assert_eq!(json(&stale)["error"]["kind"], "stale_cursor");
}

fn write_documents(path: &Path, documents: &[Document]) {
    let manifest = json!({"record":"manifest","data":{
        "format":"polylogue-bundle","version":3,"query":format!("query:{}", "0".repeat(64)),"query_text":"synthetic documents",
        "ran_at":"2026-01-01T00:00:00Z","read_at":{"generation":"synthetic","index":10,"archive":10},"freshness":{"local":{"parsing":{"state":"complete"},"derivation":{"state":"complete"},"search":{"state":"complete"},"projection":{"state":"not_applicable"},"kept_chains":0,"raw_only_chains":0},"owner":{"mode":"standalone"}},
        "counts":{"sessions":0,"messages":0,"documents":documents.len(),"events":0,"links":0,"authored":0,"assertions":0,"variants":0,"variant_parts":0,"text_fragments":0,"blobs":0},
        "flags":{"format":"jsonl","with_blobs":false,"read":{"view":"transcript","inherited":false,"assertions":true}},"enabled_variants":[],"author":"operator"}});
    let mut records = vec![serde_json::from_value::<Record>(manifest).unwrap()];
    for (ordinal, document) in documents.iter().enumerate() {
        records.push(Record::Document(polylogue_model::bundle::DocumentRecord {
            reference: format!("document:{}", document.id),
            commit: ordinal as u64 + 1,
            document: document.clone(),
        }));
    }
    let mut raw = String::new();
    for record in records {
        record.validate().unwrap();
        raw.push_str(&serde_json::to_string(&record).unwrap());
        raw.push('\n');
    }
    std::fs::write(path, raw).unwrap();
}
