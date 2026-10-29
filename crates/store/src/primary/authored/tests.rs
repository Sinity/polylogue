use super::*;
use polylogue_model::authored::ExecutionEligibility;
use polylogue_model::user::{
    BackfillCounts, BackfillReport, ChatAttachment, Delivery, NewBackfillJob, NewChatAction,
    UserWrite,
};
use polylogue_model::{Author, BlobHash, NewNote, NoteKind, Reference, Timestamp};

fn cache() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.execute_batch(&crate::source::schema().unwrap().ddl)
        .unwrap();
    conn
}
fn request(id: &str, at: i64, write: UserWrite) -> UserRequest {
    UserRequest {
        id: id.into(),
        machine: "machine-a".into(),
        at: Timestamp::from_unix_ms(at).unwrap(),
        write,
    }
}
fn note(value: &str) -> NewNote {
    NewNote {
        kind: NoteKind::Note,
        key: None,
        value: Some(value.into()),
        author: Author::Agent(Some(
            polylogue_model::SessionId::parse("codex:session-a").unwrap(),
        )),
        targets: vec![Reference::parse("repo:example").unwrap()],
        relations: vec![],
        revises: None,
        valid_until: None,
    }
}
fn saved(value: &str) -> NewNote {
    let mut n = note(value);
    n.kind = NoteKind::SavedQuery;
    n.key = Some("example".into());
    n.targets.clear();
    n
}
fn prepare_with(
    conn: &Connection,
    r: &UserRequest,
    pending: &[PreparedRequest],
    blobs: &[NewBlob],
) -> Result<PreparedRequest> {
    prepare(
        conn,
        r,
        |visit| {
            for p in pending {
                visit(&polylogue_model::authored::PendingAuthored::Live {
                    prepared: p.clone(),
                })?;
            }
            Ok(())
        },
        |visit| {
            for b in blobs {
                visit(b)?;
            }
            Ok(())
        },
    )
}
fn commit(conn: &mut Connection, p: &PreparedRequest, counter: i64) {
    let tx = conn.transaction().unwrap();
    apply(&tx, p, counter).unwrap();
    tx.commit().unwrap();
}
fn count(conn: &Connection, table: &str) -> i64 {
    conn.query_row(&format!("SELECT count(*) FROM source_{table}"), [], |r| {
        r.get(0)
    })
    .unwrap()
}
fn chat(id: &str) -> NewChatAction {
    NewChatAction {
        id: id.into(),
        request_key: format!("key-{id}"),
        request_hash: BlobHash::of(id.as_bytes()),
        request: serde_json::json!({"text":id}),
        stage_only: true,
        needs_approval: false,
        attachments: vec![],
    }
}
fn enqueue(conn: &mut Connection, id: &str, at: i64) {
    let p = prepare_with(
        conn,
        &request(
            &format!("enqueue-{id}"),
            at,
            UserWrite::ChatEnqueue(chat(id)),
        ),
        &[],
        &[],
    )
    .unwrap();
    commit(conn, &p, at);
}

#[test]
fn prepared_effect_is_private_and_preserves_full_note_provenance() {
    let mut conn = cache();
    let r = request(
        "note-a",
        10,
        UserWrite::AddNote {
            note: note("first"),
        },
    );
    let p = prepare_with(&conn, &r, &[], &[]).unwrap();
    assert_eq!(count(&conn, "note"), 0);
    assert_eq!(count(&conn, "client_request"), 0);
    assert_eq!(
        p.effect.notes[0].note.author,
        Author::Agent(Some(
            polylogue_model::SessionId::parse("codex:session-a").unwrap()
        ))
    );
    assert_eq!(p.effect.notes[0].machine, "machine-a");
    assert!(!p.outcome.pending);
    commit(&mut conn, &p, 1);
    assert_eq!(count(&conn, "note"), 1);
    assert_eq!(known_request(&conn, &r).unwrap(), Some(p));
}
#[test]
fn saved_family_and_revision_are_frozen_across_pending_overlay_and_replay() {
    let conn = cache();
    let first = prepare_with(
        &conn,
        &request(
            "first",
            10,
            UserWrite::SaveQuery {
                note: saved("sessions"),
            },
        ),
        &[],
        &[],
    )
    .unwrap();
    let second = prepare_with(
        &conn,
        &request(
            "second",
            20,
            UserWrite::SaveQuery {
                note: saved("messages"),
            },
        ),
        std::slice::from_ref(&first),
        &[],
    )
    .unwrap();
    let third = prepare_with(
        &conn,
        &request(
            "third",
            30,
            UserWrite::SaveQuery {
                note: saved("events"),
            },
        ),
        &[first.clone(), second.clone()],
        &[],
    )
    .unwrap();
    assert_eq!(third.effect.notes[0].note.first_note, "first");
    assert_eq!(third.effect.notes[0].revises, vec!["note:second"]);
    assert_eq!(count(&conn, "note"), 0);
    let mut rebuilt = cache();
    commit(&mut rebuilt, &third, 1);
    let got:(String,String)=rebuilt.query_row("SELECT n.first_note,r.other FROM source_note n JOIN source_note_relation r ON r.note_id=n.id WHERE n.id='third'",[],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
    assert_eq!(got, ("first".into(), "note:second".into()));
}
#[test]
fn changed_claim_state_never_reselects_another_chat_action() {
    let mut conn = cache();
    enqueue(&mut conn, "a", 10);
    enqueue(&mut conn, "b", 20);
    let original = prepare_with(
        &conn,
        &request("claim-a", 30, UserWrite::ChatClaim { by: "tab-a".into() }),
        &[],
        &[],
    )
    .unwrap();
    let other = prepare_with(
        &conn,
        &request(
            "claim-other",
            31,
            UserWrite::ChatClaim {
                by: "tab-other".into(),
            },
        ),
        &[],
        &[],
    )
    .unwrap();
    commit(&mut conn, &other, 3);
    commit(&mut conn, &original, 4);
    assert_eq!(original.outcome.id, "a");
    let events: Vec<(String, String)> = conn
        .prepare(
            "SELECT action,request FROM source_chat_action_event WHERE kind='claimed' ORDER BY n",
        )
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        events,
        vec![
            ("a".into(), "claim-other".into()),
            ("a".into(), "claim-a".into())
        ]
    );
}
#[test]
fn pending_claims_are_seen_after_restart_but_never_leak_into_queries() {
    let mut conn = cache();
    enqueue(&mut conn, "a", 10);
    let p = prepare_with(
        &conn,
        &request("claim", 20, UserWrite::ChatClaim { by: "tab-a".into() }),
        &[],
        &[],
    )
    .unwrap();
    let restored: Vec<PreparedRequest> =
        serde_json::from_str(&serde_json::to_string(&vec![p]).unwrap()).unwrap();
    assert!(matches!(
        prepare_with(
            &conn,
            &request("another", 21, UserWrite::ChatClaim { by: "tab-b".into() }),
            &restored,
            &[]
        ),
        Err(StoreError::NotFound(_))
    ));
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM source_chat_action_event WHERE kind='claimed'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
}
#[test]
fn duplicate_and_conflicting_retries_preserve_original_effect_and_outcome() {
    let mut conn = cache();
    enqueue(&mut conn, "a", 10);
    let r = request("claim", 20, UserWrite::ChatClaim { by: "tab-a".into() });
    let p = prepare_with(&conn, &r, &[], &[]).unwrap();
    let mut retry = r.clone();
    retry.at = Timestamp::from_unix_ms(90).unwrap();
    assert_eq!(
        prepare_with(&conn, &retry, std::slice::from_ref(&p), &[]).unwrap(),
        p
    );
    commit(&mut conn, &p, 2);
    commit(&mut conn, &p, 3);
    assert_eq!(prepare_with(&conn, &retry, &[], &[]).unwrap(), p);
    assert_eq!(count(&conn, "client_request"), 2);
    let tx = conn.transaction().unwrap();
    let repeated = crate::user_writes::apply(&tx, &retry, 4).unwrap();
    assert_eq!(repeated.id, "a");
    assert!(repeated.repeat);
    tx.rollback().unwrap();
    let mut changed = r;
    changed.write = UserWrite::ChatClaim { by: "other".into() };
    assert!(matches!(
        prepare_with(&conn, &changed, &[], &[]),
        Err(StoreError::Conflict(_))
    ));
    let mut tampered = p.clone();
    tampered.effect.chats[0].added_events[0].by = Some("other".into());
    let tx = conn.transaction().unwrap();
    assert!(matches!(
        apply(&tx, &tampered, 4),
        Err(StoreError::Conflict(_))
    ));
}
#[test]
fn canonical_refusal_creates_no_prepared_or_applied_state() {
    let conn = cache();
    let mut n = note("invalid");
    n.targets.clear();
    let r = request("bad", 10, UserWrite::AddNote { note: n });
    assert!(matches!(
        prepare_with(&conn, &r, &[], &[]),
        Err(StoreError::Invalid(_))
    ));
    assert_eq!(count(&conn, "client_request"), 0);
    assert_eq!(count(&conn, "note"), 0);
}
#[test]
fn pending_attachment_metadata_is_validated_once_and_reference_survives_missing_blob() {
    let mut conn = cache();
    let blob = NewBlob {
        hash: BlobHash::of(b"attachment"),
        size: 10,
    };
    let mut action = chat("attached");
    action.attachments.push(ChatAttachment {
        id: "file".into(),
        blob: blob.hash,
        name: "example.txt".into(),
        media_type: "text/plain".into(),
        size: 10,
    });
    let r = request("enqueue", 10, UserWrite::ChatEnqueue(action));
    assert!(matches!(
        prepare_with(&conn, &r, &[], &[]),
        Err(StoreError::ChatAction("unknown_attachment"))
    ));
    let p = prepare_with(&conn, &r, &[], std::slice::from_ref(&blob)).unwrap();
    commit(&mut conn, &p, 1);
    assert_eq!(count(&conn, "blob"), 0);
    let hash: Vec<u8> = conn
        .query_row("SELECT blob FROM source_chat_action_attachment", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(hash, blob.hash.0);
}
#[test]
fn backfill_claim_replays_its_original_job_and_preserves_reports() {
    let mut conn = cache();
    for (id, provider, at) in [("a", "service-a", 10), ("b", "service-b", 20)] {
        let p = prepare_with(
            &conn,
            &request(
                id,
                at,
                UserWrite::BackfillStart(NewBackfillJob {
                    id: id.into(),
                    provider: provider.into(),
                    since: None,
                }),
            ),
            &[],
            &[],
        )
        .unwrap();
        commit(&mut conn, &p, at);
    }
    let p = prepare_with(
        &conn,
        &request("claim", 30, UserWrite::BackfillClaim { by: "tab-a".into() }),
        &[],
        &[],
    )
    .unwrap();
    let other = prepare_with(
        &conn,
        &request("other", 31, UserWrite::BackfillClaim { by: "tab-b".into() }),
        &[],
        &[],
    )
    .unwrap();
    commit(&mut conn, &other, 3);
    commit(&mut conn, &p, 4);
    assert_eq!(p.outcome.id, "a");
    let report = prepare_with(
        &conn,
        &request(
            "progress",
            32,
            UserWrite::BackfillReport {
                job: "a".into(),
                by: "tab-a".into(),
                report: BackfillReport {
                    kind: "progress".into(),
                    position: Some("cursor".into()),
                    counts: BackfillCounts {
                        listed: 4,
                        captured: 2,
                        unchanged: 1,
                        failed: 1,
                    },
                    detail: Some("example".into()),
                },
            },
        ),
        &[],
        &[],
    )
    .unwrap();
    commit(&mut conn, &report, 5);
    let got: (String, i64, String) = conn
        .query_row(
            "SELECT position,captured,request FROM source_backfill_job_event WHERE kind='progress'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(got, ("cursor".into(), 2, "progress".into()));
}
#[test]
fn complete_authored_records_roundtrip_and_pending_is_never_canonical() {
    let mut conn = cache();
    let writes = [
        request("note", 1, UserWrite::AddNote { note: note("text") }),
        request(
            "setting",
            2,
            UserWrite::SetSetting {
                key: "example".into(),
                value: "value".into(),
                author: Author::Operator,
                machine: None,
            },
        ),
        request(
            "result",
            3,
            UserWrite::KeepResult {
                name: Some("kept".into()),
                query: "sessions".into(),
                rows_json: "[{\"id\":1}]".into(),
                read_at: "{\"now\":3}".into(),
            },
        ),
        request(
            "delivery",
            4,
            UserWrite::DeliveredAdvice(Delivery {
                recipient_kind: "operator".into(),
                recipient_id: None,
                sender: "hook".into(),
                event: "advice".into(),
                cwd: None,
                text: "example advice".into(),
                refs: vec!["note:note".into()],
                left_out: vec![],
            }),
        ),
    ];
    let mut prepared = Vec::new();
    for (n, r) in writes.iter().enumerate() {
        let p = prepare_with(&conn, r, &[], &[]).unwrap();
        commit(&mut conn, &p, n as i64 + 1);
        prepared.push(p);
    }
    let mut rebuilt = cache();
    for (n, p) in prepared.iter().enumerate() {
        let wire = serde_json::to_vec(p).unwrap();
        let restored: PreparedRequest = serde_json::from_slice(&wire).unwrap();
        commit(&mut rebuilt, &restored, n as i64 + 1);
        assert_eq!(
            known_request(&rebuilt, &p.request).unwrap(),
            Some(p.clone())
        );
    }
    assert_eq!(count(&rebuilt, "note"), 1);
    assert_eq!(count(&rebuilt, "setting"), 1);
    assert_eq!(count(&rebuilt, "kept_result"), 1);
    assert_eq!(count(&rebuilt, "delivery_event"), 1);
    let mut invalid = prepared[0].clone();
    invalid.request.id = "pending".into();
    invalid.outcome.pending = true;
    let tx = rebuilt.transaction().unwrap();
    assert!(matches!(
        apply(&tx, &invalid, 9),
        Err(StoreError::Invalid(_))
    ));
}
#[test]
fn historical_restore_preserves_history_but_cannot_authorize_execution() {
    let mut source = cache();
    enqueue(&mut source, "a", 10);
    let p = known_request(
        &source,
        &request("enqueue-a", 10, UserWrite::ChatEnqueue(chat("a"))),
    )
    .unwrap()
    .unwrap();
    let mut destination = cache();
    let attribution = ImportAttribution {
        machine: "machine-import".into(),
        input: "bundle-input".into(),
        chain: 1,
        actor: Author::Operator,
        at: Timestamp::from_unix_ms(100).unwrap(),
    };
    let tx = destination.transaction().unwrap();
    restore(&tx, &p, &[], &attribution, 1).unwrap();
    tx.commit().unwrap();
    let action = crate::chat_actions::read(&destination, "source_", "a")
        .unwrap()
        .unwrap();
    assert_eq!(action.execution, ExecutionEligibility::Historical);
    assert_eq!(action.events, p.effect.chats[0].action.events);
    assert!(matches!(
        prepare_with(
            &destination,
            &request("claim", 101, UserWrite::ChatClaim { by: "tab".into() }),
            &[],
            &[]
        ),
        Err(StoreError::NotFound(_))
    ));
    let tx = destination.transaction().unwrap();
    assert!(restore(&tx, &p, &[], &attribution, 2).unwrap().repeat);
    tx.commit().unwrap();
    assert_eq!(count(&destination, "chat_action_event"), 1);
}

#[test]
fn retry_reuses_frozen_annotation_targets_and_rejects_changed_machine() {
    let mut conn = cache();
    let r = request(
        "annotation",
        10,
        UserWrite::AnnotateQuery {
            query: "sessions".into(),
            note: note("example"),
        },
    );
    let p = prepare_with(&conn, &r, &[], &[]).unwrap();
    commit(&mut conn, &p, 1);
    let mut retry = r.clone();
    if let UserWrite::AnnotateQuery { note, .. } = &mut retry.write {
        note.targets = vec![Reference::parse("repo:changed").unwrap()];
    }
    assert_eq!(prepare_with(&conn, &retry, &[], &[]).unwrap(), p);
    retry.machine = "machine-b".into();
    assert!(matches!(
        prepare_with(&conn, &retry, &[], &[]),
        Err(StoreError::Conflict(_))
    ));
}
#[test]
fn subset_restore_keeps_full_chat_history_and_import_provenance() {
    let mut source = cache();
    enqueue(&mut source, "a", 10);
    let p = prepare_with(
        &source,
        &request("claim", 20, UserWrite::ChatClaim { by: "tab".into() }),
        &[],
        &[],
    )
    .unwrap();
    let current = ImportAttribution {
        machine: "machine-import".into(),
        input: "bundle".into(),
        chain: 4,
        actor: Author::Operator,
        at: Timestamp::from_unix_ms(30).unwrap(),
    };
    let mut dest = cache();
    let tx = dest.transaction().unwrap();
    restore(&tx, &p, &[], &current, 1).unwrap();
    tx.commit().unwrap();
    let got = crate::chat_actions::read(&dest, "source_", "a")
        .unwrap()
        .unwrap();
    assert_eq!(got.events, p.effect.chats[0].action.events);
    assert_eq!(got.execution, ExecutionEligibility::Historical);
    let mut n = 0;
    visit(&dest, &mut |record, claims| {
        n += 1;
        assert_eq!(record, &p);
        assert_eq!(claims, std::slice::from_ref(&current));
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn inconsistent_event_delta_is_refused_before_application() {
    let mut source = cache();
    enqueue(&mut source, "a", 10);
    let mut p = prepare_with(
        &source,
        &request("claim", 20, UserWrite::ChatClaim { by: "tab".into() }),
        &[],
        &[],
    )
    .unwrap();
    p.effect.chats[0].added_events[0].request = "another-request".into();
    let mut destination = cache();
    let tx = destination.transaction().unwrap();
    assert!(matches!(apply(&tx, &p, 1), Err(StoreError::Invalid(_))));
    tx.commit().unwrap();
    assert_eq!(count(&destination, "chat_action"), 0);
}

#[test]
fn repeated_import_unions_real_ingress_without_changing_frozen_state() {
    let source = cache();
    let p = prepare_with(
        &source,
        &request(
            "note",
            1,
            UserWrite::AddNote {
                note: note("example"),
            },
        ),
        &[],
        &[],
    )
    .unwrap();
    let mut conn = cache();
    let first = ImportAttribution {
        machine: "host-a".into(),
        input: "bundle".into(),
        chain: 1,
        actor: Author::Operator,
        at: Timestamp::from_unix_ms(20).unwrap(),
    };
    let tx = conn.transaction().unwrap();
    restore(&tx, &p, &[], &first, 1).unwrap();
    tx.commit().unwrap();
    let mut retry = first.clone();
    retry.at = Timestamp::from_unix_ms(30).unwrap();
    let tx = conn.transaction().unwrap();
    assert!(restore(&tx, &p, &[], &retry, 2).unwrap().repeat);
    tx.commit().unwrap();
    let mut second = retry;
    second.machine = "host-b".into();
    let tx = conn.transaction().unwrap();
    assert!(restore(&tx, &p, &[], &second, 3).unwrap().repeat);
    tx.commit().unwrap();
    assert_eq!(count(&conn, "note"), 1);
    visit(&conn, &mut |record, imports| {
        assert_eq!(record, &p);
        assert_eq!(imports, &[first.clone(), second.clone()]);
        Ok(())
    })
    .unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT commit_seq FROM source_client_request WHERE id='note'",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
}

#[test]
fn historical_claim_cannot_supply_current_import_observation() {
    let source = cache();
    let p = prepare_with(
        &source,
        &request(
            "note",
            1,
            UserWrite::AddNote {
                note: note("example"),
            },
        ),
        &[],
        &[],
    )
    .unwrap();
    let current = ImportAttribution {
        machine: "host".into(),
        input: "bundle".into(),
        chain: 1,
        actor: Author::Operator,
        at: Timestamp::from_unix_ms(20).unwrap(),
    };
    let mut claimed = current.clone();
    claimed.at = Timestamp::from_unix_ms(2).unwrap();
    let mut conn = cache();
    let tx = conn.transaction().unwrap();
    restore(&tx, &p, std::slice::from_ref(&claimed), &current, 1).unwrap();
    tx.commit().unwrap();
    let wire: String = conn
        .query_row(
            "SELECT imports FROM source_client_request WHERE id='note'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let history: ImportHistory = serde_json::from_str(&wire).unwrap();
    assert_eq!(history.observed, vec![current]);
    assert_eq!(history.declared, vec![claimed]);
}

#[test]
fn pending_historical_restore_keeps_execution_disabled_after_restart() {
    let mut source = cache();
    enqueue(&mut source, "historical", 10);
    let prepared = known_request(
        &source,
        &request(
            "enqueue-historical",
            10,
            UserWrite::ChatEnqueue(chat("historical")),
        ),
    )
    .unwrap()
    .unwrap();
    let pending = polylogue_model::authored::PendingAuthored::Restore {
        prepared: prepared.clone(),
        claims: vec![],
        current: ImportAttribution {
            machine: "import-host".into(),
            input: "bundle".into(),
            chain: 1,
            actor: Author::Operator,
            at: Timestamp::from_unix_ms(20).unwrap(),
        },
    };
    let restarted: polylogue_model::authored::PendingAuthored =
        serde_json::from_slice(&serde_json::to_vec(&pending).unwrap()).unwrap();
    let destination = cache();
    let claim = request(
        "claim",
        30,
        UserWrite::ChatClaim {
            by: "extension".into(),
        },
    );
    assert!(matches!(
        prepare(&destination, &claim, |visit| visit(&restarted), |_| Ok(())),
        Err(StoreError::NotFound(_))
    ));
    assert_eq!(count(&destination, "chat_action"), 0);
    assert_eq!(
        prepare(
            &destination,
            &prepared.request,
            |visit| visit(&restarted),
            |_| Ok(())
        )
        .unwrap(),
        prepared
    );
    let mut conflicting = prepared.request.clone();
    conflicting.machine = "another-machine".into();
    assert!(matches!(
        prepare(
            &destination,
            &conflicting,
            |visit| visit(&restarted),
            |_| Ok(())
        ),
        Err(StoreError::Conflict(_))
    ));
}

#[test]
fn reverse_full_snapshot_restore_keeps_event_order_and_historical_execution() {
    let mut source = cache();
    let writes = [
        request("enqueue", 10, UserWrite::ChatEnqueue(chat("chat"))),
        request("chat-claim", 20, UserWrite::ChatClaim { by: "tab".into() }),
        request(
            "start",
            30,
            UserWrite::BackfillStart(NewBackfillJob {
                id: "job".into(),
                provider: "service".into(),
                since: None,
            }),
        ),
        request(
            "job-claim",
            40,
            UserWrite::BackfillClaim { by: "tab".into() },
        ),
        request(
            "delivery",
            50,
            UserWrite::DeliveredAdvice(Delivery {
                recipient_kind: "operator".into(),
                recipient_id: None,
                sender: "hook".into(),
                event: "advice".into(),
                cwd: None,
                text: "example".into(),
                refs: vec![],
                left_out: vec![],
            }),
        ),
        request(
            "delivery-used",
            60,
            UserWrite::DeliveryEvent {
                delivery: "delivery".into(),
                what: polylogue_model::DeliveryOutcome::Used,
                detail: Some("evidence".into()),
            },
        ),
    ];
    let mut snapshots = Vec::new();
    for (index, r) in writes.iter().enumerate() {
        let p = prepare_with(&source, r, &[], &[]).unwrap();
        commit(&mut source, &p, index as i64 + 1);
        snapshots.push(p);
    }
    let mut forward = cache();
    let mut reverse = cache();
    for (destination, order) in [
        (&mut forward, snapshots.iter().collect::<Vec<_>>()),
        (&mut reverse, snapshots.iter().rev().collect::<Vec<_>>()),
    ] {
        for (index, p) in order.into_iter().enumerate() {
            let current = ImportAttribution {
                machine: "import-host".into(),
                input: format!("bundle-{}", p.request.id),
                chain: 1,
                actor: Author::Operator,
                at: Timestamp::from_unix_ms(100).unwrap(),
            };
            let tx = destination.transaction().unwrap();
            restore(&tx, p, &[], &current, index as i64 + 1).unwrap();
            tx.commit().unwrap();
        }
    }
    assert_eq!(
        crate::chat_actions::read(&forward, "source_", "chat").unwrap(),
        crate::chat_actions::read(&reverse, "source_", "chat").unwrap()
    );
    assert_eq!(
        crate::backfill_jobs::read(&forward, "source_", "job").unwrap(),
        crate::backfill_jobs::read(&reverse, "source_", "job").unwrap()
    );
    for table in ["chat_action_event", "backfill_job_event"] {
        let rows = |conn: &Connection| {
            conn.prepare(&format!("SELECT n,request FROM source_{table} ORDER BY n"))
                .unwrap()
                .query_map([], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
        };
        assert_eq!(rows(&forward), rows(&reverse));
        assert_eq!(
            rows(&reverse).iter().map(|row| row.0).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }
    let deliveries = |conn: &Connection| {
        conn.prepare("SELECT what,at,detail,request FROM source_delivery_event ORDER BY at,what")
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    };
    assert_eq!(deliveries(&forward), deliveries(&reverse));
    assert_eq!(deliveries(&reverse).len(), 2);
    assert!(matches!(
        prepare_with(
            &reverse,
            &request(
                "new-chat-claim",
                1_000_000,
                UserWrite::ChatClaim {
                    by: "new-tab".into()
                }
            ),
            &[],
            &[]
        ),
        Err(StoreError::NotFound(_))
    ));
    assert!(matches!(
        prepare_with(
            &reverse,
            &request(
                "new-job-claim",
                1_000_000,
                UserWrite::BackfillClaim {
                    by: "new-tab".into()
                }
            ),
            &[],
            &[]
        ),
        Err(StoreError::NotFound(_))
    ));
}
