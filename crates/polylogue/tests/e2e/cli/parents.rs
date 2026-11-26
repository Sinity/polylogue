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
            copied_prefixes: Vec::new(),
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
            copied_for: Vec::new(),
            message,
        })),
    ];
    if let Some((parent, branch_point)) = parent {
        let link: polylogue_model::Link = serde_json::from_value(json!({
            "kind":"fork","parent":format!("claude-code:{parent}"),"child":id,
            "branch_point":branch_point.map(|s|format!("claude-code:{s}")),"evidence":"synthetic fork marker"
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
        read_position: polylogue_model::ChangePosition {
            index_instance: "synthetic".into(),
            index: 1,
            archive: 1,
        },
        freshness: polylogue_model::freshness::ReadFreshness {
            local: polylogue_model::freshness::ReadCompleteness::empty(),
            sinex: polylogue_model::freshness::IndexFreshness::Standalone,
        },
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
        enabled_alternates: vec![],
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
    exit_code: Option<i32>,
}
fn read(
    env: &crate::Fixture,
    session: &str,
    inherited: bool,
    format: &str,
    budget: &str,
) -> Reading {
    let target = format!("session:{session}");
    let mut args = vec![
        "--deadline".into(),
        "none".into(),
        "--json".into(),
        target,
        "read".into(),
        "--format".into(),
        format.into(),
        "--budget".into(),
        budget.into(),
    ];
    if inherited {
        args.push("--show".into());
        args.push("inherited".into());
    }
    let out = cli(env, &args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(matches!(out.status.code(), Some(0 | 5)), "{out:?}");
    let exit_code = out.status.code();
    let mut page = read_page(json(&out));
    let mut items = super::read_current::Items::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut pages = 0;
    loop {
        assert!(!page.document.items.is_empty(), "empty continuation");
        items.push(&page);
        pages += 1;
        let Some(next) = page.document.footer.next else {
            break;
        };
        assert!(
            seen.insert(serde_json::to_string(&next.request).unwrap()),
            "repeated continuation"
        );
        page = followed_page(env, &next);
    }
    Reading {
        items: items.into_values(),
        pages,
        exit_code,
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
fn cli_reads_deep_parents_and_reports_incomplete_parent_prefixes() {
    // Exercise a read with more than 65 ancestors.
    const CHAIN_SESSIONS: usize = 67;
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let _daemon = start_daemon(&env);
    let mut sessions = Vec::new();
    for i in 0..CHAIN_SESSIONS {
        let parent = format!("s{}", i.saturating_sub(1));
        let branch_point = format!("{parent}-message");
        sessions.push(exported(
            &format!("s{i}"),
            &format!("message {i} λ🙂"),
            (i > 0).then_some((parent.as_str(), Some(branch_point.as_str()))),
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
        "missing-branch-point",
        "available λ🙂 --token=synthetic-parents-secret",
        Some(("s0", Some("absent-message"))),
    ));
    sessions.push(exported(
        "absent-branch-point",
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
    assert_eq!(
        wait_caught_up(&env)["store"]["messages"],
        CHAIN_SESSIONS + 7
    );
    use polylogue_model::read::{ParentIssue, ValueState};
    let leaf = format!("claude-code:s{}", CHAIN_SESSIONS - 1);
    let deep = read(&env, &leaf, true, "json", "10000");
    assert_eq!(deep.exit_code, Some(0));
    assert_eq!(lineage(&deep), &ValueState::known(true));
    assert!(issues(&deep).is_empty());
    assert_eq!(
        messages(&deep),
        (0..CHAIN_SESSIONS)
            .map(|i| format!("claude-code:s{i}-message"))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        texts(&deep),
        (0..CHAIN_SESSIONS)
            .map(|i| format!("message {i} λ🙂"))
            .collect::<Vec<_>>()
    );
    let prefix = read(&env, "claude-code:prefix-child", true, "json", "10000");
    assert_eq!(prefix.exit_code, Some(0));
    assert_eq!(lineage(&prefix), &ValueState::known(true));
    assert!(issues(&prefix).is_empty());
    assert_eq!(
        messages(&prefix),
        ["claude-code:s0-message", "claude-code:prefix-child-message"]
    );
    assert_eq!(texts(&prefix), ["message 0 λ🙂", "selected child"]);
    for (name, issue, count) in [
        (
            "missing-branch-point",
            ParentIssue::MissingBranchPoint {
                child: sid("missing-branch-point"),
                parent: sid("s0"),
                branch_point: Some("claude-code:absent-message".into()),
            },
            1,
        ),
        (
            "absent-branch-point",
            ParentIssue::MissingBranchPoint {
                child: sid("absent-branch-point"),
                parent: sid("s0"),
                branch_point: None,
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
    let own = read(
        &env,
        "claude-code:missing-branch-point",
        false,
        "json",
        "10000",
    );
    assert_eq!(lineage(&own), &ValueState::NotApplicable);
    assert!(issues(&own).is_empty());
    assert_eq!(messages(&own), ["claude-code:missing-branch-point-message"]);
    for format in ["terminal", "markdown", "html"] {
        let reading = read(
            &env,
            "claude-code:missing-branch-point",
            true,
            format,
            "10000",
        );
        assert_eq!(lineage(&reading), &ValueState::known(false));
        assert_eq!(
            issues(&reading),
            [ParentIssue::MissingBranchPoint {
                child: sid("missing-branch-point"),
                parent: sid("s0"),
                branch_point: Some("claude-code:absent-message".into())
            }]
        );
        assert_eq!(
            texts(&reading),
            ["available λ🙂 --token=synthetic-parents-secret"]
        );
        assert_eq!(
            messages(&reading),
            ["claude-code:missing-branch-point-message"],
            "unproven parent included"
        );
    }
    let paged = read(&env, "claude-code:missing-branch-point", true, "json", "2");
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
        [ParentIssue::MissingBranchPoint {
            child: sid("missing-branch-point"),
            parent: sid("s0"),
            branch_point: Some("claude-code:absent-message".into())
        }]
    );
    assert_eq!(
        messages(&paged),
        ["claude-code:missing-branch-point-message"]
    );
    assert_eq!(
        texts(&paged),
        ["available λ🙂 --token=synthetic-parents-secret"]
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let context = cli(
        &env,
        &[
            "--json",
            "session:claude-code:missing-branch-point",
            "brief",
        ],
    );
    assert!(context.status.success(), "{context:?}");
    let context = json(&context)["reply"]["steps"][0]["outcome"]["reply"]["reply"].clone();
    let issue = "fork from claude-code:s0 to claude-code:missing-branch-point has no available branch point (claude-code:absent-message)";
    assert!(
        context["prepared"]["text"]
            .as_str()
            .unwrap()
            .lines()
            .any(|line| line == format!("Parents incomplete: {issue}")),
        "{context}"
    );
    let inherited: Vec<_> = context["prepared"]["left_out"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["what"] == "inherited session content")
        .collect();
    assert_eq!(
        inherited,
        [&json!({"what": "inherited session content", "reason": issue})]
    );
}
