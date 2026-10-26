//! An export imported into a fresh archive gives the same sessions and IDs,
//! and the same user state: notes with their revisions, withdrawals and
//! judgments, settings, kept results, deliveries, chat actions, backfill
//! jobs and excisions, which keep refusing what they removed.
use super::*;
use polylogue_model::{Author, DeliveryOutcome, NewNote, NoteKind, Target, Timestamp};
use polylogue_store::backfill_jobs::{BackfillCounts, BackfillReport, NewBackfillJob};
use polylogue_store::chat_actions::NewChatAction;
use polylogue_store::{Delivery, UserRequest, UserWrite};

/// The user-state tables, row by row, without the write counter, which
/// counts this archive's own transactions.
const USER_STATE: &[(&str, &str)] = &[
    ("client_request", "id, content_hash, created_at, request"),
    (
        "note",
        "id, first_note, kind, key, value, author, machine, created_at, valid_until",
    ),
    ("note_target", "note_id, ref"),
    ("note_relation", "note_id, kind, other"),
    ("kept_result", "*"),
    ("setting", "id, machine, key, value, author, set_at"),
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
    ("excision", "id, target, reason, created_at"),
    ("excised", "*"),
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

fn request(n: u32, write: UserWrite) -> UserRequest {
    UserRequest {
        id: format!("0199d8a0-0000-7000-8000-{n:012}"),
        machine: "synthetic-machine".into(),
        at: Timestamp::from_unix_ms(1_790_000_000_000 + i64::from(n) * 1000).unwrap(),
        write,
    }
}

fn note(kind: NoteKind, author: Author, target: &str, revises: Option<String>) -> UserWrite {
    UserWrite::AddNote {
        note: NewNote {
            kind,
            key: None,
            value: Some(format!("synthetic {}", kind.as_str())),
            author,
            targets: vec![Target::parse(target).unwrap()],
            relations: vec![],
            revises,
            valid_until: None,
        },
    }
}

fn export_file(w: &ClaudeHome, sessions: &[&str]) -> std::path::PathBuf {
    let pool = w.reads();
    let entries = pool
        .reader()
        .unwrap()
        .user_state()
        .unwrap()
        .iter()
        .map(|entry| serde_json::to_value(entry).unwrap())
        .collect();
    let doc =
        polylogue_model::export::Document::new(sessions.iter().map(|id| w.export(id)).collect())
            .with_user_state(entries);
    let file = w.dir.path().join("export.json");
    fs::write(&file, doc.to_json().unwrap()).unwrap();
    file
}

#[test]
fn an_export_imported_into_a_fresh_archive_gives_the_same_sessions_ids_and_user_state() {
    let w = ClaudeHome::new();
    w.append(
        "s1.jsonl",
        &(user("u1", "s1", "hello") + &assistant("a1", "s1", "hi") + &user("u2", "s1", "more")),
    );
    let dropped = user("d1", "s2", "drop pelican77");
    w.append("s2.jsonl", &dropped);
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.run_pass().unwrap().errors.is_empty());
    let writes = e.writer().unwrap();
    let agent = Author::Agent(Some(SessionId::parse("claude-code:s1").unwrap()));
    let mut n = 0;
    let mut write = |w: UserWrite| {
        n += 1;
        writes.user(request(n, w)).unwrap().id
    };
    let first = write(note(
        NoteKind::Note,
        Author::Operator,
        "session:claude-code:s1",
        None,
    ));
    write(note(
        NoteKind::Note,
        Author::Operator,
        "session:claude-code:s1",
        Some(first),
    ));
    let finding = write(note(
        NoteKind::Finding,
        agent.clone(),
        "session:claude-code:s1",
        None,
    ));
    write(note(
        NoteKind::Judgment,
        Author::Operator,
        &format!("note:{finding}"),
        None,
    ));
    let withdrawn = write(note(NoteKind::Lesson, agent, "repo:synthetic", None));
    write(UserWrite::Withdraw {
        note: withdrawn,
        author: Author::Operator,
        reason: Some("superseded".into()),
    });
    // Notes about the session excised below go with it.
    write(note(
        NoteKind::Note,
        Author::Operator,
        "session:claude-code:s2",
        None,
    ));
    write(UserWrite::SetSetting {
        key: "day.zone".into(),
        value: "UTC".into(),
        author: Author::Operator,
        machine: None,
    });
    write(UserWrite::KeepResult {
        name: Some("kept".into()),
        query: "sessions".into(),
        rows_json: "[]".into(),
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
    e.excise(ExciseRequest {
        id: "0199d8a0-0000-7000-8000-00000000e001".into(),
        target: ExciseTarget::parse("session:claude-code:s2").unwrap(),
        reason: Some("synthetic".into()),
        dry_run: false,
    })
    .unwrap();
    // A note after the excision replays after it.
    write(note(
        NoteKind::Note,
        Author::Operator,
        "session:claude-code:s1",
        None,
    ));
    let original = w.export("claude-code:s1");
    let source_state = user_state(&w);
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

    // Importing it again changes nothing.
    let again = e2.import(std::slice::from_ref(&file)).unwrap();
    assert_eq!(again.user_state.applied, 0, "{again:?}");
    assert_eq!(again.user_state.present, report.user_state.applied);
    assert_eq!(user_state(&fresh), source_state);

    // The imported excision refuses the excised session when its file
    // arrives again.
    fresh.append("s2.jsonl", &dropped);
    assert!(e2.run_pass().unwrap().errors.is_empty());
    assert_eq!(
        fresh.count("SELECT count(*) AS n FROM session WHERE id = 'claude-code:s2'"),
        0
    );
}

#[test]
fn an_imported_excision_refuses_when_the_archive_already_holds_what_it_removed() {
    let w = ClaudeHome::new();
    let dropped = user("d1", "s2", "drop pelican77");
    w.append("s2.jsonl", &dropped);
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.run_pass().unwrap().errors.is_empty());
    e.excise(ExciseRequest {
        id: "0199d8a0-0000-7000-8000-00000000e002".into(),
        target: ExciseTarget::parse("session:claude-code:s2").unwrap(),
        reason: None,
        dry_run: false,
    })
    .unwrap();
    let file = export_file(&w, &[]);

    let fresh = ClaudeHome::new();
    fresh.append("s2.jsonl", &dropped);
    let (mut e2, _lock2) = fresh.engine(64 << 20);
    assert!(e2.run_pass().unwrap().errors.is_empty());
    let report = e2.import(std::slice::from_ref(&file)).unwrap();
    assert_eq!(report.user_state.applied, 0, "{report:?}");
    assert!(
        report
            .pass
            .errors
            .iter()
            .any(|e| e.contains("excise it here, then import again")),
        "{report:?}"
    );
    assert_eq!(fresh.count("SELECT count(*) AS n FROM a.excision"), 0);
}

fn excise_s2(e: &mut Engine) -> Result<Excised, polylogue_engine::EngineError> {
    e.excise(ExciseRequest {
        id: "0199d8a0-0000-7000-8000-00000000e003".into(),
        target: ExciseTarget::parse("session:claude-code:s2").unwrap(),
        reason: None,
        dry_run: false,
    })
}

fn two_sessions() -> (ClaudeHome, Engine, polylogue_store::ArchiveLock) {
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &user("u1", "s1", "keep"));
    w.append("s2.jsonl", &user("d1", "s2", "drop"));
    let (mut e, lock) = w.engine(64 << 20);
    assert!(e.run_pass().unwrap().errors.is_empty());
    (w, e, lock)
}

/// A withdrawal is about the note it withdraws only, so it goes with that
/// note, request and all, and what is left replays.
#[test]
fn an_excision_takes_the_withdrawals_of_the_notes_it_removes() {
    let (w, mut e, _lock) = two_sessions();
    let writes = e.writer().unwrap();
    let withdrawn = writes
        .user(request(
            1,
            note(
                NoteKind::Note,
                Author::Operator,
                "session:claude-code:s2",
                None,
            ),
        ))
        .unwrap()
        .id;
    writes
        .user(request(
            2,
            UserWrite::Withdraw {
                note: withdrawn,
                author: Author::Operator,
                reason: None,
            },
        ))
        .unwrap();
    writes
        .user(request(
            3,
            note(
                NoteKind::Note,
                Author::Operator,
                "session:claude-code:s1",
                None,
            ),
        ))
        .unwrap();
    assert_eq!(excise_s2(&mut e).unwrap().notes_removed, 2);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.note"), 1);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.client_request"), 1);

    let file = export_file(&w, &[]);
    let fresh = ClaudeHome::new();
    let (mut e2, _lock2) = fresh.engine(64 << 20);
    let report = e2.import(std::slice::from_ref(&file)).unwrap();
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert_eq!(report.user_state.applied, 2, "{report:?}");
    assert_eq!(user_state(&fresh), user_state(&w));
}

/// A revision that would stay without the note it revises could not be
/// replayed, so the excision refuses and changes nothing.
#[test]
fn an_excision_refuses_to_leave_a_revision_without_its_note() {
    let (w, mut e, _lock) = two_sessions();
    let writes = e.writer().unwrap();
    let revised = writes
        .user(request(
            1,
            note(
                NoteKind::Note,
                Author::Operator,
                "session:claude-code:s2",
                None,
            ),
        ))
        .unwrap()
        .id;
    writes
        .user(request(
            2,
            note(
                NoteKind::Note,
                Author::Operator,
                "session:claude-code:s1",
                Some(revised),
            ),
        ))
        .unwrap();
    let before = user_state(&w);
    let refused = excise_s2(&mut e).unwrap_err().to_string();
    assert!(refused.contains("its revision"), "{refused}");
    assert_eq!(user_state(&w), before);
}
