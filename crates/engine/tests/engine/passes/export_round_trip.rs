//! An export imported into a fresh archive gives the same sessions and IDs,
//! and the same user state: notes with their revisions, withdrawals and
//! judgments, settings, kept results, deliveries, chat actions, backfill
//! jobs, restored as frozen historical effects through the canonical bundle.
use super::*;
use polylogue_model::assertion::{
    AssertionKind, AssertionPatch, Change, LinkRole, NewAssertion, NewAssertionLink,
};
use polylogue_model::bundle::{
    AuthoredRecord, Counts, Flags, Format, LinkRecord, Manifest, MessageRecord, ReadSnapshot,
    Record, SessionRecord,
};
use polylogue_model::user::NewChatAction;
use polylogue_model::user::{BackfillCounts, BackfillReport, NewBackfillJob};
use polylogue_model::user::{Delivery, UserRequest, UserWrite};
use polylogue_model::{Author, DeliveryOutcome, Reference, Timestamp};
use std::io::Write;

/// The user-state tables, row by row, without the write counter, which
/// counts this archive's own transactions. Execution eligibility is checked
/// separately: imported chat and backfill histories are historical.
const USER_STATE: &[(&str, &str)] = &[
    ("client_request", "id, content_hash, created_at, request"),
    (
        "assertion",
        "id,root,kind,key,value,author,via,machine,created_at,valid_until",
    ),
    ("link", "record,role,ref,seen_hash,quote"),
    ("revision_state", "*"),
    ("kept_result", "*"),
    (
        "delivery",
        "id, recipient_kind, recipient_id, sender, event, cwd, at, content_hash, bytes, body, refs_json, left_out_json",
    ),
    ("delivery_event", "delivery, what, at, detail"),
    (
        "chat_action",
        "id, request_key, request_hash, request_json, stage_only, needs_approval, created_at",
    ),
    ("chat_action_attachment", "*"),
    (
        "chat_action_event",
        "action, n, at, kind, phase, detail, by, receipt_json",
    ),
    ("backfill_job", "id, provider, since, created_at"),
    (
        "backfill_job_event",
        "job, n, at, kind, by, position, listed, captured, unchanged, failed, detail",
    ),
];

fn user_state(w: &ClaudeHome) -> Vec<(String, serde_json::Value)> {
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    USER_STATE
        .iter()
        .map(|(table, columns)| {
            let mut rows: Vec<String> = reader
                .query(&format!("SELECT {columns} FROM a.{table}"), &[])
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .collect();
            rows.sort();
            ((*table).to_owned(), serde_json::json!(rows))
        })
        .collect()
}

fn request(n: u32, write: UserWrite, author: Author) -> UserRequest {
    UserRequest {
        authorship: polylogue_model::authorship::Authorship {
            author,
            via: polylogue_model::authorship::Channel::Cli,
        },
        id: format!("0199d8a0-0000-7000-8000-{n:012}"),
        machine: "synthetic-machine".into(),
        at: Timestamp::from_unix_ms(1_790_000_000_000 + i64::from(n) * 1000).unwrap(),
        write,
    }
}

fn note(kind: AssertionKind, target: &str, revises: Option<String>) -> UserWrite {
    let value = serde_json::json!(format!("synthetic {}", kind.as_str()));
    if let Some(base) = revises {
        return UserWrite::ReviseAssertions {
            bases: vec![polylogue_model::RevisionReference::exact(
                base.clone(),
                base,
            )],
            patch: AssertionPatch {
                value: Change::Set(value),
                ..Default::default()
            },
        };
    }
    UserWrite::Assert {
        assertions: vec![NewAssertion {
            kind,
            key: if kind == AssertionKind::Judgment {
                Some("accepted".into())
            } else {
                None
            },
            value: Some(value),
            valid_until: None,
            links: vec![NewAssertionLink {
                role: LinkRole::About,
                reference: Reference::parse(target).unwrap(),
            }],
        }],
    }
}

fn frozen_authored(w: &ClaudeHome) -> Vec<polylogue_model::authored::PreparedRequest> {
    let mut records = Vec::new();
    w.reads()
        .reader()
        .unwrap()
        .visit_authored(&mut |prepared| {
            records.push(prepared.clone());
            Ok(())
        })
        .unwrap();
    records
}

fn export_file(w: &ClaudeHome, sessions: &[&str]) -> std::path::PathBuf {
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let mut records = Vec::new();
    let mut counts = Counts::default();
    let mut seen_links = std::collections::BTreeSet::new();
    for id in sessions {
        let id = SessionId::parse(id).unwrap();
        let session = reader.session(&id).unwrap().unwrap();
        records.push(Record::Session(SessionRecord {
            copied_prefixes: reader.copied_prefixes(&id).unwrap(),
            reference: format!("session:{id}"),
            commit: reader.session_commit(&id).unwrap().unwrap(),
            session,
        }));
        counts.sessions += 1;
        for message in reader.messages(&id, false).unwrap() {
            let (message, projection) = reader
                .message_with_projection(&message.id)
                .unwrap()
                .unwrap();
            records.push(Record::Message(Box::new(MessageRecord {
                reference: format!("message:{}", message.id),
                commit: reader.message_admission(&message.id).unwrap().unwrap(),
                order: reader.message_order(&message.id).unwrap().unwrap(),
                source_version: projection.source_version,
                block_count: projection.block_count,
                block_ordinals: projection.block_ordinals,
                injections: Vec::new(),
                bindings: Vec::new(),
                copied_for: reader.copied_for(&message.id).unwrap(),
                message,
            })));
            counts.messages += 1;
        }
        for event in reader.event_records(&id).unwrap() {
            records.push(Record::Event(event));
            counts.events += 1;
        }
        for link in reader.session_export(&id).unwrap().unwrap().links {
            let reference = LinkRecord::reference(&link);
            if seen_links.insert(reference.clone()) {
                records.push(Record::Link(LinkRecord {
                    reference,
                    commit: reader.link_commit(&link).unwrap().unwrap(),
                    link,
                }));
                counts.links += 1;
            }
        }
    }
    let commits: std::collections::BTreeMap<_, _> = reader
        .query("SELECT id,commit_seq FROM source_client_request", &[])
        .unwrap()
        .into_iter()
        .map(|row| {
            (
                row["id"].as_str().unwrap().to_owned(),
                row["commit_seq"].as_u64().unwrap(),
            )
        })
        .collect();
    reader
        .visit_authored(&mut |prepared| {
            records.push(Record::Authored(Box::new(AuthoredRecord {
                reference: format!("request:{}", prepared.request.id),
                commit: commits[&prepared.request.id],
                prepared: prepared.clone(),
            })));
            counts.authored += 1;
            Ok(())
        })
        .unwrap();
    records.sort_by(|a, b| (a.commit(), a.reference()).cmp(&(b.commit(), b.reference())));
    let manifest = Manifest::from_snapshot(
        ReadSnapshot {
            query: format!("query:{}", "0".repeat(64)),
            query_text: "synthetic session and authored state".into(),
            ran_at: Timestamp::from_unix_ms(1_790_000_000_000).unwrap(),
            read_at: reader.change_position().unwrap(),
            freshness: polylogue_model::currency::ReadFreshness {
                local: polylogue_model::currency::ReadCompleteness::empty(),
                owner: polylogue_model::currency::IndexCurrency::Standalone,
            },
            enabled_variants: Vec::new(),
            author: Author::Operator,
            notes: Vec::new(),
            incomplete: Vec::new(),
        },
        counts,
        Flags {
            format: Format::Jsonl,
            with_blobs: false,
            read: polylogue_model::read::ReadOptions::default(),
        },
        None,
    );
    let (mut file, path) = tempfile::Builder::new()
        .prefix("records-")
        .suffix(".jsonl")
        .tempfile_in(&w.dir)
        .unwrap()
        .keep()
        .unwrap();
    for record in std::iter::once(Record::Manifest(Box::new(manifest))).chain(records) {
        record.validate().unwrap();
        serde_json::to_writer(&mut file, &record).unwrap();
        file.write_all(b"\n").unwrap();
    }
    path
}

#[test]
fn an_export_imported_into_a_fresh_archive_gives_the_same_sessions_ids_and_user_state() {
    let w = ClaudeHome::new();
    w.append(
        "s1.jsonl",
        &(user("u1", "s1", "hello") + &assistant("a1", "s1", "hi") + &user("u2", "s1", "more")),
    );
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.run_pass().unwrap().errors.is_empty());
    let writes = e.writer().unwrap();
    let agent = Author::Agent(Some(SessionId::parse("claude-code:s1").unwrap()));
    let n = std::cell::Cell::new(0);
    let write_as = |w: UserWrite, author: Author| {
        n.set(n.get() + 1);
        writes.user(request(n.get(), w, author)).unwrap().id
    };
    let write = |w| write_as(w, Author::Operator);
    let write_note =
        |kind, author, target: &str, revises| write_as(note(kind, target, revises), author);
    let first = write_note(
        AssertionKind::Note,
        Author::Operator,
        "session:claude-code:s1",
        None,
    );
    write_note(
        AssertionKind::Note,
        Author::Operator,
        "session:claude-code:s1",
        Some(first),
    );
    let finding = write_note(
        AssertionKind::Finding,
        agent.clone(),
        "session:claude-code:s1",
        None,
    );
    write_note(
        AssertionKind::Judgment,
        Author::Operator,
        &format!("assertion:{finding}@{finding}"),
        None,
    );
    let withdrawn = write_note(AssertionKind::Lesson, agent, "repo:synthetic", None);
    write(UserWrite::WithdrawAssertions {
        targets: vec![Reference::Assertion(polylogue_model::RevisionReference {
            root: withdrawn,
            revision: None,
        })],
        reason: Some("superseded".into()),
    });
    // Authored notes retain their explicit session targets.
    write_note(
        AssertionKind::Note,
        Author::Operator,
        "session:claude-code:s2",
        None,
    );
    write(UserWrite::SetSetting {
        key: "browser.actions".into(),
        value: "on".into(),
        machine: None,
    });
    write(UserWrite::KeepResult {
        name: Some("kept".into()),
        query: "sessions".into(),
        rows_json: "{\"unit\":\"sessions\",\"columns\":[],\"rows\":[]}".into(),
        read_at: "{}".into(),
    });
    let delivery = write(UserWrite::Deliver(Delivery {
        recipient_kind: "session".into(),
        recipient_id: Some("claude-code:s1".into()),
        sender: "hook".into(),
        event: "session_start".into(),
        cwd: Some("/synthetic".into()),
        text: "a synthetic brief".into(),
        refs: vec!["session:claude-code:s1".into()],
        left_out: vec![],
    }));
    write(UserWrite::DeliveryEvent {
        delivery,
        what: DeliveryOutcome::Delivered,
        detail: None,
    });
    let chat_request = serde_json::json!({"provider": "chatgpt", "text": "synthetic"});
    write(UserWrite::ChatEnqueue(NewChatAction {
        id: "0199d8a0-0000-7000-8000-00000000c001".into(),
        request_key: "synthetic-key".into(),
        request_hash: polylogue_model::BlobHash::of(chat_request.to_string().as_bytes()),
        request: chat_request,
        stage_only: true,
        needs_approval: false,
        attachments: vec![],
    }));
    let action = write(UserWrite::ChatClaim {
        by: "extension-1".into(),
    });
    write(UserWrite::ChatReport {
        action,
        by: "extension-1".into(),
        outcome: "drafted".into(),
        phase: "composer".into(),
        detail: None,
        receipt: None,
    });
    write(UserWrite::BackfillStart(NewBackfillJob {
        id: "0199d8a0-0000-7000-8000-00000000b001".into(),
        provider: "chatgpt".into(),
        since: None,
    }));
    let job = write(UserWrite::BackfillClaim {
        by: "extension-1".into(),
    });
    write(UserWrite::BackfillReport {
        job,
        by: "extension-1".into(),
        report: BackfillReport {
            kind: "finished".into(),
            position: None,
            counts: BackfillCounts {
                listed: 2,
                captured: 1,
                unchanged: 1,
                failed: 0,
            },
            detail: None,
        },
    });
    write_note(
        AssertionKind::Note,
        Author::Operator,
        "session:claude-code:s1",
        None,
    );
    let original = w.export("claude-code:s1");
    let source_state = user_state(&w);
    let source_authored = frozen_authored(&w);
    let file = export_file(&w, &["claude-code:s1"]);

    let fresh = ClaudeHome::new();
    let (mut e2, _lock2) = fresh.engine(64 << 20);
    let report = e2.import(std::slice::from_ref(&file)).unwrap();
    assert_eq!(report.imported, std::slice::from_ref(&file), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    let back = fresh.export("claude-code:s1");
    assert_eq!(back.session.id, original.session.id);
    assert_eq!(back.session.title, original.session.title);
    assert_eq!(back.messages, original.messages);
    assert_eq!(back.events, original.events);
    assert_eq!(back.links, original.links);
    assert_eq!(user_state(&fresh), source_state);
    assert_eq!(frozen_authored(&fresh), source_authored);
    assert_eq!(
        fresh.count("SELECT count(*) AS n FROM a.chat_action WHERE execution='historical'"),
        1
    );
    assert_eq!(
        fresh.count("SELECT count(*) AS n FROM a.backfill_job WHERE execution='historical'"),
        1
    );

    // Importing it again changes nothing.
    let again = e2.import(std::slice::from_ref(&file)).unwrap();
    assert_eq!(again.user_state.applied, 0, "{again:?}");
    assert_eq!(again.user_state.present, report.user_state.applied);
    assert_eq!(frozen_authored(&fresh), source_authored);
    assert_eq!(user_state(&fresh), source_state);
}

#[test]
fn conflicting_fork_statements_survive_canonical_bundle_export_and_import() {
    let original = ClaudeHome::new();
    original.append(
        "parent.jsonl",
        &(user("first", "parent", "first") + &user("second", "parent", "second")),
    );
    for (message, anchor) in [("child-first", "first"), ("child-second", "second")] {
        let mut record: serde_json::Value =
            serde_json::from_str(&user(message, "child", "own child")).unwrap();
        record["forkedFrom"] = serde_json::json!({"sessionId":"parent","messageUuid":anchor});
        original.append("child.jsonl", &format!("{record}\n"));
    }
    let (mut source, _lock) = original.engine(64 << 20);
    let pass = source.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let child = SessionId::parse("claude-code:child").unwrap();
    let expected = original.reads().reader().unwrap().links(&child).unwrap();
    assert_eq!(expected.len(), 2);
    assert_ne!(
        LinkRecord::reference(&expected[0]),
        LinkRecord::reference(&expected[1])
    );
    let path = export_file(&original, &["claude-code:child"]);
    let receiving = ClaudeHome::new();
    let (mut target, _target_lock) = receiving.engine(64 << 20);
    let report = target.import(&[path]).unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    let actual = receiving.reads().reader().unwrap().links(&child).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn copied_subset_survives_whole_parent_bundle_and_later_parent_import() {
    fn conversation(id: &str, time: u64, messages: &[(&str, &str)]) -> serde_json::Value {
        let mapping: serde_json::Map<String, serde_json::Value> = messages.iter().enumerate().map(|(i,(id,text))| {
            (id.to_string(), serde_json::json!({"id":id,"parent":if i==0 {None}else{Some(messages[i-1].0)},"children":[],"message":{"id":id,"author":{"role":"user"},"content":{"content_type":"text","parts":[text]},"create_time":i+1}}))
        }).collect();
        serde_json::json!({"id":id,"create_time":time,"current_node":messages.last().map(|(id,_)|*id),"mapping":mapping})
    }
    fn inherited(home: &ClaudeHome) -> Vec<String> {
        let pool = home.reads();
        let reader = pool.reader().unwrap();
        let selection = reader
            .select_messages(&SessionId::parse("chatgpt:child").unwrap(), true)
            .unwrap();
        selection.require_complete().unwrap();
        let mut texts = Vec::new();
        selection
            .visit(|message| {
                texts.push(message.text());
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
        texts
    }
    let original = ClaudeHome::new();
    let path = original.dir.path().join("conversations.json");
    std::fs::write(
        &path,
        serde_json::to_vec(&vec![
            conversation(
                "parent",
                1,
                &[
                    ("first", "first"),
                    ("uncopied", "uncopied parent middle"),
                    ("third", "third"),
                    ("later", "later parent tail"),
                ],
            ),
            conversation(
                "child",
                2,
                &[("first", "first"), ("third", "third"), ("own", "own child")],
            ),
        ])
        .unwrap(),
    )
    .unwrap();
    let (mut source, _source_lock) = original.engine(64 << 20);
    let report = source.import(&[path]).unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert_eq!(inherited(&original), ["first", "third", "own child"]);
    let whole = export_file(&original, &["chatgpt:parent", "chatgpt:child"]);
    let later = export_file(&original, &["chatgpt:parent"]);
    let receiving = ClaudeHome::new();
    let (mut target, _target_lock) = receiving.engine(64 << 20);
    for file in [whole, later] {
        let report = target.import(&[file]).unwrap();
        assert!(report.refused.is_empty(), "{report:?}");
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert_eq!(inherited(&receiving), ["first", "third", "own child"]);
    }
    assert_eq!(
        receiving
            .reads()
            .reader()
            .unwrap()
            .messages(&SessionId::parse("chatgpt:parent").unwrap(), false)
            .unwrap()
            .len(),
        4
    );
    let pool = receiving.reads();
    let reader = pool.reader().unwrap();
    let id = polylogue_model::MessageId::parse("chatgpt:first").unwrap();
    assert_eq!(
        reader.copied_for(&id).unwrap(),
        original.reads().reader().unwrap().copied_for(&id).unwrap()
    );
}

#[test]
fn native_copied_file_subset_survives_export_with_whole_parent() {
    let original = ClaudeHome::new();
    original.append(
        "parent.jsonl",
        &(user("first", "parent", "copied first") + &user("later", "parent", "uncopied later")),
    );
    original.append(
        "child.jsonl",
        &(user("first", "parent", "copied first") + &user("own", "child", "own child")),
    );
    let (mut source, _source_lock) = original.engine(64 << 20);
    let pass = source.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let child = SessionId::parse("claude-code:child").unwrap();
    let bodies = |home: &ClaudeHome| {
        let pool = home.reads();
        let reader = pool.reader().unwrap();
        let selected = reader.select_messages(&child, true).unwrap();
        selected.require_complete().unwrap();
        let mut texts = Vec::new();
        selected
            .visit(|m| {
                texts.push(m.text());
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
        texts
    };
    assert_eq!(bodies(&original), ["copied first", "own child"]);
    let file = export_file(&original, &["claude-code:parent", "claude-code:child"]);
    let target = ClaudeHome::new();
    let (mut engine, _target_lock) = target.engine(64 << 20);
    let report = engine.import(&[file]).unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert_eq!(bodies(&target), ["copied first", "own child"]);
}

#[test]
fn later_copy_annotations_preserve_declared_parent_without_expanding_earlier_bundles() {
    fn view(home: &ClaudeHome) -> (polylogue_model::read::Parents, Vec<String>) {
        let pool = home.reads();
        let reader = pool.reader().unwrap();
        let selected = reader
            .select_messages(&SessionId::parse("claude-code:child").unwrap(), true)
            .unwrap();
        let mut texts = Vec::new();
        selected
            .visit(|m| {
                texts.push(m.text());
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
        (selected.parents().clone(), texts)
    }
    for declared in ["parent", "other"] {
        let original = ClaudeHome::new();
        original.append(
            "parent.jsonl",
            &(user("shared", "parent", "shared") + &user("uncopied", "parent", "uncopied")),
        );
        original.append("other.jsonl", &user("other-own", "other", "other own"));
        original.append("child.jsonl", &user("own", "child", "own"));
        let (mut source, _source_lock) = original.engine(64 << 20);
        assert!(source.run_pass().unwrap().errors.is_empty());
        let sessions = [
            "claude-code:parent",
            "claude-code:other",
            "claude-code:child",
        ];
        let base = export_file(&original, &sessions);
        let annotated = export_file(&original, &sessions);
        let mut records: Vec<Record> = fs::read_to_string(&annotated)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let claim = polylogue_model::bundle::CopiedFor {
            parent: SessionId::parse(&format!("claude-code:{declared}")).unwrap(),
            child: SessionId::parse("claude-code:child").unwrap(),
            source: polylogue_model::PrefixSource {
                input: "supplied-native-reading".into(),
                chain: 3,
                reading_version: "copy-reader=1".into(),
            },
            position: [42, 7],
        };
        for record in &mut records {
            match record {
                Record::Message(record) if record.message.id.native().as_str() == "shared" => {
                    record.copied_for = vec![claim.clone()]
                }
                Record::Session(record) if record.session.id.native().as_str() == "child" => {
                    record.copied_prefixes = vec![claim.parent.clone()]
                }
                _ => {}
            }
        }
        let mut file = fs::File::create(&annotated).unwrap();
        for record in records {
            record.validate().unwrap();
            serde_json::to_writer(&mut file, &record).unwrap();
            file.write_all(b"\n").unwrap();
        }
        drop(file);
        let receiving = ClaudeHome::new();
        let (mut target, _target_lock) = receiving.engine(64 << 20);
        let report = target.import(&[base]).unwrap();
        assert!(report.refused.is_empty(), "{report:?}");
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert_eq!(view(&receiving).1, ["own"]);
        let report = target.import(&[annotated]).unwrap();
        assert!(report.refused.is_empty(), "{report:?}");
        assert!(report.pass.errors.is_empty(), "{report:?}");
        let expected = view(&receiving);
        if declared == "parent" {
            assert!(expected.0.complete, "{:?}", expected.0);
            assert_eq!(expected.1, ["shared", "own"]);
        } else {
            assert!(
                matches!(expected.0.issues.as_slice(),[polylogue_model::read::ParentIssue::MissingCopiedMessage {parent,..}] if parent==&claim.parent)
            );
            assert_eq!(expected.1, ["own"]);
        }
        let id = polylogue_model::MessageId::parse("claude-code:shared").unwrap();
        assert_eq!(
            receiving
                .reads()
                .reader()
                .unwrap()
                .copied_for(&id)
                .unwrap()
                .as_slice(),
            std::slice::from_ref(&claim)
        );
        let portable = export_file(&receiving, &sessions);
        let final_home = ClaudeHome::new();
        let (mut final_engine, _final_lock) = final_home.engine(64 << 20);
        let report = final_engine.import(&[portable]).unwrap();
        assert!(report.refused.is_empty(), "{report:?}");
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert_eq!(view(&final_home), expected);
        assert_eq!(
            final_home
                .reads()
                .reader()
                .unwrap()
                .copied_for(&id)
                .unwrap(),
            [claim]
        );
    }
}
