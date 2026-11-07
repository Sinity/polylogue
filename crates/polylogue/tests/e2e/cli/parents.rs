use super::*;
use polylogue_model::bundle::{self, Record};
use serde_json::json;

fn exported(name: &str, text: &str, parent: Option<(&str, Option<&str>)>) -> Vec<Record> {
    let id = format!("claude-code:{name}");
    let session: polylogue_model::Session =
        serde_json::from_value(json!({"id":id,"kind":"main"})).unwrap();
    let message: polylogue_model::Message = serde_json::from_value(json!({
        "id":format!("claude-code:{name}-message"),"session":id,"role":"user",
        "blocks":[{"type":"text","text":text}]
    }))
    .unwrap();
    let mut records = vec![
        Record::Session(bundle::SessionRecord {
            reference: format!("session:{}", session.id),
            commit: 1,
            session,
        }),
        Record::Message(Box::new(bundle::MessageRecord {
            reference: format!("message:{}", message.id),
            commit: 1,
            order: 0,
            source_version: message.content_hash().unwrap(),
            block_count: 1,
            block_ordinals: vec![0],
            injections: vec![],
            bindings: vec![],
            message,
        })),
    ];
    if let Some((parent, anchor)) = parent {
        let link: polylogue_model::Link = serde_json::from_value(json!({
            "kind":"fork","parent":format!("claude-code:{parent}"),"child":id,
            "at":anchor.map(|s|format!("claude-code:{s}")),"evidence":"synthetic fork marker"
        }))
        .unwrap();
        records.push(Record::Link(bundle::LinkRecord {
            reference: bundle::LinkRecord::reference(&link),
            commit: 1,
            link,
        }));
    }
    records
}

fn write_bundle(path: &Path, sessions: Vec<Vec<Record>>) -> Vec<u8> {
    let records: Vec<_> = sessions.into_iter().flatten().collect();
    let manifest = bundle::Manifest {
        format: bundle::FORMAT.into(),
        version: bundle::VERSION,
        query: format!("query:{}", "a".repeat(64)),
        query_text: "sessions".into(),
        ran_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        read_at: polylogue_model::ChangePosition {
            generation: "synthetic".into(),
            index: 1,
            archive: 1,
        },
        caught_up: true,
        currency: polylogue_model::currency::IndexCurrency::Standalone,
        counts: bundle::Counts {
            sessions: records
                .iter()
                .filter(|r| matches!(r, Record::Session(_)))
                .count() as u64,
            messages: records
                .iter()
                .filter(|r| matches!(r, Record::Message(_)))
                .count() as u64,
            links: records
                .iter()
                .filter(|r| matches!(r, Record::Link(_)))
                .count() as u64,
            ..Default::default()
        },
        flags: bundle::Flags {
            format: bundle::Format::Jsonl,
            with_blobs: false,
            read: Default::default(),
        },
        enabled_variants: vec![],
        author: polylogue_model::Author::Operator,
        notes: vec![],
        incomplete: vec![],
        inventory: None,
        coverage: vec![],
    };
    manifest.validate().unwrap();
    let mut bytes = Vec::new();
    for record in std::iter::once(Record::Manifest(Box::new(manifest))).chain(records) {
        record.validate().unwrap();
        serde_json::to_writer(&mut bytes, &record).unwrap();
        bytes.push(b'\n');
    }
    std::fs::write(path, &bytes).unwrap();
    bytes
}

struct Reading {
    items: Vec<polylogue_model::read::ReadItem>,
    pages: usize,
}
fn read(
    env: &crate::Fixture,
    session: &str,
    inherited: bool,
    format: &str,
    budget: &str,
) -> Reading {
    use polylogue_model::read::ReadPage;
    let target = format!("session:{session}");
    let mut args = vec![
        "--deadline".into(),
        "none".into(),
        "--json".into(),
        "read".into(),
        target,
        "--format".into(),
        format.into(),
        "--page-tokens".into(),
        budget.into(),
    ];
    if inherited {
        args.push("--show".into());
        args.push("inherited".into());
    }
    let mut items = super::read_current::Items::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut pages = 0;
    loop {
        let out = cli(env, &args.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(matches!(out.status.code(), Some(0 | 5)), "{out:?}");
        let page: ReadPage = serde_json::from_value(json(&out)).unwrap();
        assert!(!page.document.items.is_empty(), "empty continuation");
        items.push(&page);
        pages += 1;
        let Some(next) = page.document.footer.next else {
            break;
        };
        assert!(seen.insert(next.cli.clone()), "repeated continuation");
        let stages = polylogue_model::command::effective_stages(&next.cli);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].program, "polylogue");
        args = vec!["--json".into()];
        args.extend(stages[0].args.clone());
    }
    Reading {
        items: items.into_values(),
        pages,
    }
}
fn messages(reading: &Reading) -> Vec<String> {
    reading
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            polylogue_model::read::ItemKind::MessageHeader { .. } => {
                let polylogue_model::Reference::Message(id) =
                    polylogue_model::Reference::parse(&item.source.reference).unwrap()
                else {
                    panic!("message header has a non-message source: {:?}", item.source);
                };
                Some(id.to_string())
            }
            _ => None,
        })
        .collect()
}
fn texts(reading: &Reading) -> Vec<&str> {
    reading
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            polylogue_model::read::ItemKind::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}
fn issues(reading: &Reading) -> Vec<polylogue_model::read::ParentIssue> {
    reading
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            polylogue_model::read::ItemKind::LinkIssue { issue } => Some(issue.clone()),
            _ => None,
        })
        .collect()
}
fn lineage(reading: &Reading) -> &polylogue_model::read::ValueState<bool> {
    reading
        .items
        .iter()
        .find_map(|item| match &item.kind {
            polylogue_model::read::ItemKind::Card { card } => match card.as_ref() {
                polylogue_model::read::Card::Session {
                    lineage_complete, ..
                } => Some(lineage_complete),
                _ => None,
            },
            _ => None,
        })
        .expect("selected session has explicit lineage completeness")
}
fn sid(name: &str) -> polylogue_model::SessionId {
    polylogue_model::SessionId::parse(&format!("claude-code:{name}")).unwrap()
}

#[test]
fn ordinary_cli_reads_deep_parents_and_reports_incomplete_parent_prefixes() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let _daemon = start_daemon(&env);
    let mut sessions = Vec::new();
    for i in 0usize..1030 {
        let parent = format!("s{}", i.saturating_sub(1));
        let anchor = format!("{parent}-message");
        sessions.push(exported(
            &format!("s{i}"),
            &format!("message {i} λ🙂"),
            (i > 0).then_some((parent.as_str(), Some(anchor.as_str()))),
        ));
    }
    sessions.push(exported(
        "prefix-parent",
        "unselected parent tail",
        Some(("s1", Some("s1-message"))),
    ));
    sessions.push(exported(
        "prefix-child",
        "selected child",
        Some(("prefix-parent", Some("s0-message"))),
    ));
    sessions.push(exported(
        "missing-anchor",
        "available λ🙂 --token=synthetic-parents-secret",
        Some(("s0", Some("absent-message"))),
    ));
    sessions.push(exported(
        "absent-anchor",
        "available child",
        Some(("s0", None)),
    ));
    sessions.push(exported(
        "orphan",
        "available orphan",
        Some(("unavailable", Some("unavailable-message"))),
    ));
    sessions.push(exported(
        "cycle-a",
        "cycle a",
        Some(("cycle-b", Some("cycle-b-message"))),
    ));
    sessions.push(exported(
        "cycle-b",
        "cycle b",
        Some(("cycle-a", Some("cycle-a-message"))),
    ));
    let path = dir.path().join("synthetic-bundle.jsonl");
    let original = write_bundle(&path, sessions);
    // This root imports the complete deep-lineage reconstruction corpus.
    let imported = cli(
        &env,
        &[
            "--deadline",
            "none",
            "--json",
            "import",
            path.to_str().unwrap(),
        ],
    );
    assert!(imported.status.success(), "{imported:?}");
    wait_status(&env, |status| status["store"]["messages"] == 1037);
    use polylogue_model::read::{ParentIssue, ValueState};
    let deep = read(&env, "claude-code:s1029", true, "json", "10000");
    assert_eq!(lineage(&deep), &ValueState::known(true));
    assert!(issues(&deep).is_empty());
    assert_eq!(
        messages(&deep),
        (0..1030)
            .map(|i| format!("claude-code:s{i}-message"))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        texts(&deep),
        (0..1030)
            .map(|i| format!("message {i} λ🙂"))
            .collect::<Vec<_>>()
    );
    let prefix = read(&env, "claude-code:prefix-child", true, "json", "10000");
    assert_eq!(lineage(&prefix), &ValueState::known(true));
    assert!(issues(&prefix).is_empty());
    assert_eq!(
        messages(&prefix),
        ["claude-code:s0-message", "claude-code:prefix-child-message"]
    );
    assert_eq!(texts(&prefix), ["message 0 λ🙂", "selected child"]);
    for (name, issue, count) in [
        (
            "missing-anchor",
            ParentIssue::MissingAnchor {
                child: sid("missing-anchor"),
                parent: sid("s0"),
                anchor: Some("claude-code:absent-message".into()),
            },
            1,
        ),
        (
            "absent-anchor",
            ParentIssue::MissingAnchor {
                child: sid("absent-anchor"),
                parent: sid("s0"),
                anchor: None,
            },
            1,
        ),
        (
            "orphan",
            ParentIssue::MissingParent {
                child: sid("orphan"),
                parent: sid("unavailable"),
            },
            1,
        ),
        (
            "cycle-a",
            ParentIssue::Cycle {
                session: sid("cycle-a"),
            },
            2,
        ),
    ] {
        let reading = read(&env, &format!("claude-code:{name}"), true, "json", "10000");
        assert_eq!(lineage(&reading), &ValueState::known(false));
        assert_eq!(issues(&reading), [issue]);
        let mut selected = messages(&reading);
        assert_eq!(selected.len(), count);
        selected.sort();
        let mut expected = if name == "cycle-a" {
            vec![
                "claude-code:cycle-a-message".to_owned(),
                "claude-code:cycle-b-message".to_owned(),
            ]
        } else {
            vec![format!("claude-code:{name}-message")]
        };
        expected.sort();
        assert_eq!(
            selected, expected,
            "only the available proven messages may appear"
        );
    }
    let own = read(&env, "claude-code:missing-anchor", false, "json", "10000");
    assert_eq!(lineage(&own), &ValueState::NotApplicable);
    assert!(issues(&own).is_empty());
    assert_eq!(messages(&own), ["claude-code:missing-anchor-message"]);
    for format in ["terminal", "markdown", "html"] {
        let reading = read(&env, "claude-code:missing-anchor", true, format, "10000");
        assert_eq!(lineage(&reading), &ValueState::known(false));
        assert_eq!(
            issues(&reading),
            [ParentIssue::MissingAnchor {
                child: sid("missing-anchor"),
                parent: sid("s0"),
                anchor: Some("claude-code:absent-message".into())
            }]
        );
        assert_eq!(
            texts(&reading),
            ["available λ🙂 --token=synthetic-parents-secret"]
        );
        assert_eq!(
            messages(&reading),
            ["claude-code:missing-anchor-message"],
            "unproven parent included"
        );
    }
    let paged = read(&env, "claude-code:missing-anchor", true, "json", "2");
    assert!(paged.pages > 1);
    let issue_position = paged
        .items
        .iter()
        .position(|item| matches!(item.kind, polylogue_model::read::ItemKind::LinkIssue { .. }))
        .unwrap();
    let message_position = paged
        .items
        .iter()
        .position(|item| {
            matches!(
                item.kind,
                polylogue_model::read::ItemKind::MessageHeader { .. }
            )
        })
        .unwrap();
    assert!(
        issue_position < message_position,
        "lineage issue family must precede message family across resume"
    );
    assert_eq!(lineage(&paged), &ValueState::known(false));
    assert_eq!(
        issues(&paged),
        [ParentIssue::MissingAnchor {
            child: sid("missing-anchor"),
            parent: sid("s0"),
            anchor: Some("claude-code:absent-message".into())
        }]
    );
    assert_eq!(messages(&paged), ["claude-code:missing-anchor-message"]);
    assert_eq!(
        texts(&paged),
        ["available λ🙂 --token=synthetic-parents-secret"]
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let context = cli(&env, &["--json", "takeover", "claude-code:missing-anchor"]);
    assert!(context.status.success(), "{context:?}");
    let context = json(&context);
    assert!(
        context["prepared"]["text"]
            .as_str()
            .unwrap()
            .contains("Parents incomplete")
    );
    assert!(
        context["prepared"]["left_out"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["what"] == "inherited session content")
    );
}
