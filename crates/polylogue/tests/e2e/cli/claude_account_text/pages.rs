use super::*;
use base64::Engine as _;
use polylogue_model::{Block, Document, DocumentId, DocumentPurpose, Timestamp, ToolOutcome};

#[test]
fn document_pages_mask_hidden_prefix_and_refuse_cross_content_stale_and_invalid_positions() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let mut doc = Document::new(
        DocumentId::parse("claude-ai:manual").unwrap(),
        DocumentPurpose::Context,
    );
    doc.source_time = Timestamp::from_unix_ms(1);
    doc.blocks = vec![
        Block::ToolResult {
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
    ];
    let mut other = doc.clone();
    other.id = DocumentId::parse("claude-ai:other").unwrap();
    let source = dir.path().join("documents.json");
    let export =
        polylogue_model::export::Document::new(Vec::new()).with_documents(vec![doc.clone(), other]);
    std::fs::write(&source, export.to_json().unwrap()).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    let imported = cli(&env, &["--json", "import", source.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    let reference = "document:claude-ai:manual";
    let whole = cli(
        &env,
        &[
            "--json", "read", reference, "--view", "dialogue", "--budget", "100000",
        ],
    );
    assert!(whole.status.success(), "{whole:?}");
    let expected = json(&whole)["text"].as_str().unwrap().to_owned();
    assert!(!expected.contains("synthetic-secret-tail"));
    assert!(expected.contains("Visible λ🙂 tail"));
    let mut next = None::<String>;
    let mut text = String::new();
    let mut first = None;
    for _ in 0..200 {
        let mut args = vec![
            "--json", "read", reference, "--view", "dialogue", "--budget", "1",
        ];
        if let Some(cursor) = &next {
            args.extend(["--cursor", cursor]);
        }
        let result = cli(&env, &args);
        assert!(result.status.success(), "{result:?}");
        let page = json(&result);
        text.push_str(page["text"].as_str().unwrap());
        next = page["next"].as_str().map(str::to_owned);
        if first.is_none() {
            first = next.clone();
        }
        if next.is_none() {
            break;
        }
    }
    assert!(next.is_none());
    assert_eq!(text, expected);
    let token = first.unwrap();
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(&token)
        .unwrap();
    let cursor: serde_json::Value = serde_json::from_slice(&decoded).unwrap();
    assert!(!String::from_utf8(decoded).unwrap().contains("synthetic"));
    for (id, view) in [
        ("document:claude-ai:other", "dialogue"),
        (reference, "full"),
    ] {
        let result = cli(
            &env,
            &[
                "--json", "read", id, "--view", view, "--budget", "1", "--cursor", &token,
            ],
        );
        assert!(!result.status.success());
        assert_eq!(json(&result)["error"]["kind"], "bad_request");
    }
    let mut malformed = vec!["broken".into()];
    for position in [[999, 0, 0], [1, 999, 0], [1, 0, 999999]] {
        let mut changed = cursor.clone();
        changed["position"] = json!(position);
        malformed.push(
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(serde_json::to_vec(&changed).unwrap()),
        );
    }
    let mut old = cursor;
    old["version"] = json!(1);
    malformed.push(
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&old).unwrap()),
    );
    for token in malformed {
        let result = cli(
            &env,
            &[
                "--json", "read", reference, "--view", "dialogue", "--budget", "1", "--cursor",
                &token,
            ],
        );
        assert!(!result.status.success(), "{result:?}");
        assert_eq!(json(&result)["error"]["kind"], "bad_request");
    }
    let note = cli(
        &env,
        &[
            "--json",
            "note",
            "add",
            "document:claude-ai:other",
            "--body",
            "Unrelated write",
        ],
    );
    assert!(note.status.success(), "{note:?}");
    let continued = cli(
        &env,
        &[
            "--json", "read", reference, "--view", "dialogue", "--budget", "2", "--cursor", &token,
        ],
    );
    assert!(continued.status.success(), "{continued:?}");
    let index = rusqlite::Connection::open(archive.join("index.db")).unwrap();
    index.execute("UPDATE document_block SET body=x'00' WHERE idx=2 AND document_pk=(SELECT pk FROM document WHERE id='claude-ai:manual')",[]).unwrap();
    drop(index);
    let prefix = cli(
        &env,
        &[
            "--json", "read", reference, "--view", "dialogue", "--budget", "1",
        ],
    );
    assert!(prefix.status.success(), "{prefix:?}");
    let full = cli(&env, &["--json", "read", reference]);
    assert!(!full.status.success(), "{full:?}");
    doc.source_time = Timestamp::from_unix_ms(2);
    doc.blocks = vec![Block::Text {
        text: "A changed document version".into(),
    }];
    let changed = dir.path().join("changed.json");
    std::fs::write(
        &changed,
        polylogue_model::export::Document::new(Vec::new())
            .with_documents(vec![doc])
            .to_json()
            .unwrap(),
    )
    .unwrap();
    let imported = cli(&env, &["--json", "import", changed.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    let stale = cli(
        &env,
        &[
            "--json", "read", reference, "--view", "dialogue", "--budget", "2", "--cursor", &token,
        ],
    );
    assert!(!stale.status.success(), "{stale:?}");
    assert_eq!(json(&stale)["error"]["kind"], "stale_cursor");
}
