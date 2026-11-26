use super::*;
use polylogue_model::{Block, Document, DocumentId, DocumentPurpose, Timestamp, ToolOutcome};

#[test]
fn document_pages_preserve_prefix_and_refuse_cross_content_stale_and_tampered_continuations() {
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
        polylogue_model::TimeAccuracy::intrinsic(polylogue_model::Precision::Instant),
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
    eprintln!("document pages: initial import");
    let imported = cli(&env, &["--json", "import", source.to_str().unwrap()]);
    let _ = import_report(&imported);
    let reference = "document:claude-ai:manual";
    eprintln!("document pages: whole reading");
    let whole = reading(&env, reference, &["--budget", "100000"]);
    let encoded = serde_json::to_string(&whole).unwrap();
    assert!(encoded.contains("synthetic-secret-tail"));
    assert!(encoded.contains("Visible λ🙂 tail"));
    eprintln!("document pages: concatenate one-token pages");
    assert_eq!(joined_items(&env, reference, "1"), whole.document.items);
    eprintln!("document pages: portable document");
    assert_eq!(
        document(&env, reference),
        serde_json::to_value(&doc).unwrap()
    );
    eprintln!("document pages: terminal and HTML");
    let terminal = cli(&env, &[reference, "read", "--format", "terminal"]);
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
            reference,
            "read",
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
    eprintln!("document pages: refuse invalid continuation tokens");
    let first = reading(&env, reference, &["--budget", "1"]);
    let next = first.document.footer.next.unwrap();
    let cursor = next.request.arguments["steps"][0]["cursor"]
        .as_str()
        .unwrap();
    assert!(!cursor.is_empty());
    assert!(!cursor.contains("synthetic-secret-tail"));
    let other_page = reading(&env, "document:claude-ai:other", &["--budget", "1"]);
    let other = other_page.document.footer.next.unwrap();
    for token in [
        other.request.arguments["steps"][0]["cursor"]
            .as_str()
            .unwrap()
            .to_owned(),
        "broken".into(),
        String::new(),
        format!("{cursor}x"),
    ] {
        let mut changed = next.clone();
        changed.request.arguments["steps"][0]["cursor"] = token.into();
        read_failure(follow(&env, &changed));
    }
    eprintln!("document pages: unrelated write and continuation");
    let before_unrelated = followed_page(&env, &next);
    let note = cli(
        &env,
        &[
            "--json",
            "document:claude-ai:other",
            "note",
            "Unrelated write",
        ],
    );
    assert!(note.status.success(), "{note:?}");
    let continued = followed_page(&env, &next);
    assert_eq!(continued.document.items, before_unrelated.document.items);
    assert_eq!(continued.document.header, before_unrelated.document.header);
    eprintln!("document pages: corrupt body and preserve readable prefix");
    let index = rusqlite::Connection::open(archive.join("index.db")).unwrap();
    index.execute("UPDATE document_block SET body=x'00' WHERE idx=2 AND document_pk=(SELECT pk FROM document WHERE id='claude-ai:manual')",[]).unwrap();
    drop(index);
    let prefix = reading(&env, reference, &["--budget", "1"]);
    assert!(!prefix.document.items.is_empty());
    let full = cli(&env, &["--json", reference, "read", "--budget", "100000"]);
    assert!(!full.status.success(), "{full:?}");
    doc.set_time(
        Timestamp::from_unix_ms(2),
        polylogue_model::TimeAccuracy::intrinsic(polylogue_model::Precision::Instant),
    )
    .unwrap();
    doc.blocks = vec![Block::Text {
        text: "A changed document version".into(),
    }];
    let changed = dir.path().join("changed.json");
    write_documents(&changed, &[doc]);
    eprintln!("document pages: replacement import and stale continuation");
    let imported = cli(&env, &["--json", "import", changed.to_str().unwrap()]);
    let _ = import_report(&imported);
    let stale = read_failure(follow(&env, &next));
    assert_eq!(serde_json::to_value(stale).unwrap()["kind"], "stale_cursor");
}

fn write_documents(path: &Path, documents: &[Document]) {
    let manifest = json!({"record":"manifest","data":{
        "format":"polylogue-bundle","version":4,"query":format!("query:{}", "0".repeat(64)),"query_text":"synthetic documents",
        "ran_at":"2026-01-01T00:00:00Z","read_at":{"generation":"synthetic","index":10,"archive":10},"freshness":{"local":{"parsing":{"state":"complete"},"derivation":{"state":"complete"},"search":{"state":"complete"},"confirmed_stream":{"state":"not_applicable"},"kept_chains":0,"raw_only_chains":0},"sinex":{"mode":"standalone"}},
        "counts":{"sessions":0,"messages":0,"documents":documents.len(),"events":0,"links":0,"authored":0,"annotations":0,"alternates":0,"alternate_parts":0,"text_fragments":0,"blobs":0},
        "flags":{"format":"jsonl","with_blobs":false,"read":{"view":"transcript","inherited":false,"annotations":true}},"enabled_alternates":[],"author":"operator"}});
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
