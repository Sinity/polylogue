use super::*;
use polylogue_model::authored::ExecutionEligibility;
use polylogue_model::user::{
    BackfillCounts, BackfillReport, ChatAttachment, Delivery, NewBackfillJob, NewChatAction,
    UserWrite,
};
use polylogue_model::{
    Author, BlobHash, Reference, Timestamp,
    assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink},
};

fn cache() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.execute_batch(&crate::source::schema().unwrap().ddl)
        .unwrap();
    conn
}
fn request(id: &str, at: i64, write: UserWrite) -> UserRequest {
    UserRequest {
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        },
        id: id.into(),
        machine: "machine-a".into(),
        at: Timestamp::from_unix_ms(at).unwrap(),
        write,
    }
}
fn statement(value: &str) -> NewAssertion {
    NewAssertion {
        kind: AssertionKind::Note,
        key: None,
        value: Some(serde_json::json!(value)),
        links: vec![NewAssertionLink {
            role: LinkRole::About,
            reference: Reference::parse("repo:example").unwrap(),
        }],
        valid_until: None,
    }
}

fn prepare_with(
    conn: &Connection,
    r: &UserRequest,
    pending: &[PreparedRequest],
    blobs: &[NewBlob],
) -> Result<PreparedRequest> {
    let context = crate::user_writes::PreparationContext::new(vec!["codex".into()]);
    prepare(
        conn,
        r,
        Some(&context),
        None,
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
fn prepared_effect_is_private_and_preserves_full_assertion_provenance() {
    let mut conn = cache();
    let mut r = request(
        "note-a",
        10,
        UserWrite::Assert {
            assertions: vec![statement("first")],
        },
    );
    r.authorship.author = Author::Agent(Some(
        polylogue_model::SessionId::parse("codex:session-a").unwrap(),
    ));
    let p = prepare_with(&conn, &r, &[], &[]).unwrap();
    assert_eq!(count(&conn, "assertion"), 0);
    assert_eq!(count(&conn, "client_request"), 0);
    assert_eq!(
        p.effect.assertions[0].author,
        Author::Agent(Some(
            polylogue_model::SessionId::parse("codex:session-a").unwrap()
        ))
    );
    assert_eq!(p.effect.assertions[0].machine, "machine-a");
    assert_eq!(p.effect.assertions[0].via, r.authorship.via);
    assert_eq!(p.effect.assertions[0].created_at, Some(r.at));
    assert!(!p.outcome.pending);
    commit(&mut conn, &p, 1);
    assert_eq!(count(&conn, "assertion"), 1);
    assert_eq!(known_request(&conn, &r).unwrap(), Some(p));
}
#[test]
fn saved_family_and_revision_are_frozen_across_pending_overlay_and_replay() {
    let conn = cache();
    // This focused cache fixture retains real binder-produced question bodies.
    // The Writer tests separately exercise their native ACK boundary.
    for text in ["sessions", "messages", "events"] {
        let plan = polylogue_query::plan::prepare(
            text,
            &polylogue_query::Bindings {
                origins: vec!["codex".into()],
                ..Default::default()
            },
            |_| Ok(()),
        )
        .unwrap();
        crate::query_objects::apply_object(&conn, &plan.query_object().unwrap(), 1).unwrap();
    }

    let first = prepare_with(
        &conn,
        &request(
            "first",
            10,
            UserWrite::SaveQuery {
                name: "example".into(),
                text: "sessions".into(),
                denominator: None,
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
                name: "example".into(),
                text: "messages".into(),
                denominator: None,
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
                name: "example".into(),
                text: "events".into(),
                denominator: None,
            },
        ),
        &[first.clone(), second.clone()],
        &[],
    )
    .unwrap();
    assert_eq!(third.effect.assertions[0].root, "first");
    assert!(third.effect.assertions[0].links.iter().any(|link| link.role
        == polylogue_model::assertion::LinkRole::Revises
        && link.reference == Reference::parse("assertion:first@second").unwrap()));
    assert_eq!(count(&conn, "assertion"), 0);
    let mut rebuilt = cache();
    commit(&mut rebuilt, &third, 1);
    let got:(String,String)=rebuilt.query_row("SELECT a.root,l.ref FROM source_assertion a JOIN source_link l ON l.record=a.id WHERE a.id='third' AND l.role='revises'",[],|row|Ok((row.get(0)?,row.get(1)?))).unwrap();
    assert_eq!(got, ("first".into(), "assertion:first@second".into()));
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
    for authorship in [
        polylogue_model::authorship::Authorship {
            author: Author::Unattributed,
            via: r.authorship.via,
        },
        polylogue_model::authorship::Authorship {
            author: r.authorship.author.clone(),
            via: polylogue_model::authorship::Channel::Mcp,
        },
    ] {
        let mut changed = retry.clone();
        changed.authorship = authorship;
        assert!(matches!(
            prepare_with(&conn, &changed, std::slice::from_ref(&p), &[]),
            Err(StoreError::Conflict(_))
        ));
    }
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
    let mut assertion = statement("invalid");
    assertion.links.clear();
    let r = request(
        "bad",
        10,
        UserWrite::Assert {
            assertions: vec![assertion],
        },
    );
    assert!(matches!(
        prepare_with(&conn, &r, &[], &[]),
        Err(StoreError::Invalid(_))
    ));
    assert_eq!(count(&conn, "client_request"), 0);
    assert_eq!(count(&conn, "assertion"), 0);
}
#[test]
fn pending_attachment_metadata_is_validated_once_and_reference_survives_missing_blob() {
    let mut conn = cache();
    let blob = NewBlob::whole(BlobHash::of(b"attachment"), 10);
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
        request(
            "note",
            1,
            UserWrite::Assert {
                assertions: vec![statement("text")],
            },
        ),
        request(
            "setting",
            2,
            UserWrite::SetSetting {
                key: "browser.actions".into(),
                value: "on".into(),
                machine: None,
            },
        ),
        request(
            "result",
            3,
            UserWrite::KeepResult {
                name: Some("kept".into()),
                query: "sessions".into(),
                rows_json: "{\"unit\":\"sessions\",\"columns\":[{\"name\":\"id\",\"kind\":\"number\"}],\"rows\":[{\"id\":1}]}".into(),
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
                refs: vec!["assertion:note".into()],
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
    assert_eq!(count(&rebuilt, "assertion"), 2);
    let kinds: Vec<(String, i64)> = rebuilt
        .prepare("SELECT kind,count(*) FROM source_assertion GROUP BY kind ORDER BY kind")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        kinds,
        vec![("note".to_owned(), 1), ("setting".to_owned(), 1)]
    );
    for request in &prepared {
        for fact in &request.effect.assertions {
            assert_eq!(
                crate::assertions::get(&rebuilt, &fact.id)
                    .unwrap()
                    .unwrap()
                    .fact,
                *fact
            );
        }
    }
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
fn a_kept_body_later_preparations_cannot_decode_is_refused_through_primary_apply() {
    let conn = cache();
    let kept = request(
        "kept",
        1,
        UserWrite::KeepResult {
            name: None,
            query: "sessions".into(),
            rows_json: "{\"unit\":\"sessions\",\"columns\":[],\"rows\":[]}".into(),
            read_at: "{\"now\":1}".into(),
        },
    );
    let mut malformed = prepare_with(&conn, &kept, &[], &[]).unwrap();
    if let UserWrite::KeepResult { rows_json, .. } = &mut malformed.request.write {
        *rows_json = "[]".into();
    }
    malformed.effect.kept_results[0].rows_json = "[]".into();
    let mut fresh = cache();
    let tx = fresh.transaction().unwrap();
    assert!(matches!(
        apply(&tx, &malformed, 1),
        Err(StoreError::Invalid(message)) if message.starts_with("kept result \"kept\" field rows_json: ")
    ));
    drop(tx);
    assert_eq!(count(&fresh, "kept_result"), 0);
}

#[test]
fn restored_actions_keep_their_history_but_never_run() {
    let mut source = cache();
    enqueue(&mut source, "a", 10);
    let p = known_request(
        &source,
        &request("enqueue-a", 10, UserWrite::ChatEnqueue(chat("a"))),
    )
    .unwrap()
    .unwrap();
    let mut destination = cache();
    let tx = destination.transaction().unwrap();
    restore(&tx, &p, 1).unwrap();
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
    assert!(restore(&tx, &p, 2).unwrap().repeat);
    tx.commit().unwrap();
    assert_eq!(count(&destination, "chat_action_event"), 1);
}

#[test]
fn retry_reuses_frozen_annotation_targets_and_rejects_changed_machine() {
    let mut conn = cache();
    let mut assertion = statement("example");
    assertion.links.clear();
    let r = request(
        "annotation",
        10,
        UserWrite::AssertQuery {
            query: "sessions".into(),
            assertion,
            resolved: vec![Reference::parse("repo:example").unwrap()],
        },
    );
    let p = prepare_with(&conn, &r, &[], &[]).unwrap();
    commit(&mut conn, &p, 1);
    let mut retry = r.clone();
    if let UserWrite::AssertQuery { resolved, .. } = &mut retry.write {
        *resolved = vec![Reference::parse("repo:changed").unwrap()];
    }
    assert_eq!(prepare_with(&conn, &retry, &[], &[]).unwrap(), p);
    retry.machine = "machine-b".into();
    assert!(matches!(
        prepare_with(&conn, &retry, &[], &[]),
        Err(StoreError::Conflict(_))
    ));
}
#[test]
fn subset_restore_keeps_full_chat_history() {
    let mut source = cache();
    enqueue(&mut source, "a", 10);
    let p = prepare_with(
        &source,
        &request("claim", 20, UserWrite::ChatClaim { by: "tab".into() }),
        &[],
        &[],
    )
    .unwrap();
    let mut dest = cache();
    let tx = dest.transaction().unwrap();
    restore(&tx, &p, 1).unwrap();
    tx.commit().unwrap();
    let got = crate::chat_actions::read(&dest, "source_", "a")
        .unwrap()
        .unwrap();
    assert_eq!(got.events, p.effect.chats[0].action.events);
    assert_eq!(got.execution, ExecutionEligibility::Historical);
    let mut n = 0;
    visit(&dest, &mut |record| {
        n += 1;
        assert_eq!(record, &p);
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
fn repeated_import_changes_no_frozen_state() {
    let source = cache();
    let p = prepare_with(
        &source,
        &request(
            "note",
            1,
            UserWrite::Assert {
                assertions: vec![statement("example")],
            },
        ),
        &[],
        &[],
    )
    .unwrap();
    let mut conn = cache();
    let tx = conn.transaction().unwrap();
    restore(&tx, &p, 1).unwrap();
    tx.commit().unwrap();
    for counter in [2, 3] {
        let tx = conn.transaction().unwrap();
        assert!(restore(&tx, &p, counter).unwrap().repeat);
        tx.commit().unwrap();
    }
    assert_eq!(count(&conn, "assertion"), 1);
    visit(&conn, &mut |record| {
        assert_eq!(record, &p);
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
fn pending_restore_keeps_execution_disabled_after_restart() {
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
        prepare(
            &destination,
            &claim,
            None,
            None,
            |visit| visit(&restarted),
            |_| Ok(())
        ),
        Err(StoreError::NotFound(_))
    ));
    assert_eq!(count(&destination, "chat_action"), 0);
    assert_eq!(
        prepare(
            &destination,
            &prepared.request,
            None,
            None,
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
            None,
            None,
            |visit| visit(&restarted),
            |_| Ok(())
        ),
        Err(StoreError::Conflict(_))
    ));
}

#[test]
fn reverse_full_snapshot_restore_keeps_event_order_and_disabled_execution() {
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
            let tx = destination.transaction().unwrap();
            restore(&tx, p, index as i64 + 1).unwrap();
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

#[test]
fn two_hosts_first_setting_show_conflict_in_primary() {
    let mut conn = cache();
    let set = |id: &str, at: i64, value: &str| {
        request(
            id,
            at,
            UserWrite::SetSetting {
                key: "browser.actions".into(),
                value: value.into(),
                machine: None,
            },
        )
    };
    let in_conflict = |conn: &Connection| -> i64 {
        conn.query_row(
            &format!(
                "SELECT count(*) FROM source_assertion a WHERE a.kind = 'setting' AND {}",
                crate::assertions::sql::in_conflict("a")
            ),
            [],
            |row| row.get(0),
        )
        .unwrap()
    };
    // Each host prepares its first setting before the stream shows it the
    // other's; the stream then applies both.
    let laptop = prepare_with(&conn, &set("laptop", 1, "on"), &[], &[]).unwrap();
    let desktop = prepare_with(&conn, &set("desktop", 2, "off"), &[], &[]).unwrap();
    commit(&mut conn, &laptop, 1);
    commit(&mut conn, &desktop, 2);
    assert_eq!(in_conflict(&conn), 2);
    // The next setting from either host revises both heads.
    let settled = prepare_with(&conn, &set("settled", 3, "on"), &[], &[]).unwrap();
    commit(&mut conn, &settled, 3);
    assert_eq!(in_conflict(&conn), 0);
    let current: i64 = conn
        .query_row(
            "SELECT count(*) FROM source_revision_state s JOIN source_assertion a ON a.id = s.id \
             WHERE a.kind = 'setting' AND s.current = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(current, 1);
}

fn literal_delivery(advice: bool) -> UserRequest {
    let body = Delivery {
        recipient_kind: "session".into(),
        recipient_id: Some("session:later:arrival".into()),
        sender: "hook".into(),
        event: "advice".into(),
        cwd: Some("/home/op/project".into()),
        text: "First line\nSecond line: café\u{0000}tail".into(),
        refs: vec![
            "message:codex:later".into(),
            "opaque:future:reference".into(),
        ],
        left_out: vec![polylogue_model::user::LeftOut {
            what: "unavailable section".into(),
            reason: "not read".into(),
        }],
    };
    request(
        if advice {
            "printed-advice"
        } else {
            "prepared-delivery"
        },
        42,
        if advice {
            UserWrite::DeliveredAdvice(body)
        } else {
            UserWrite::Deliver(body)
        },
    )
}

// The original transactional SQL is an independent oracle for this literal
// family. Other mutation families continue to use the overlay in production.
fn delivery_from_transaction(index: &Connection, request: &UserRequest) -> PreparedRequest {
    with_overlay(index, |_| Ok(()), |tx| {
        let delivery = match &request.write {
            UserWrite::Deliver(delivery) | UserWrite::DeliveredAdvice(delivery) => delivery,
            _ => unreachable!(),
        };
        tx.execute("INSERT INTO source_delivery(id,recipient_kind,recipient_id,sender,event,cwd,at,content_hash,bytes,body,refs_json,left_out_json,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,1)", params![request.id,delivery.recipient_kind,delivery.recipient_id,delivery.sender,delivery.event,delivery.cwd,request.at.unix_ms(),blake3::hash(delivery.text.as_bytes()).as_bytes().as_slice(),delivery.text.len() as i64,crate::encoding::pack_text(&delivery.text)?,serde_json::to_string(&delivery.refs)?,serde_json::to_string(&delivery.left_out)?])?;
        if matches!(request.write, UserWrite::DeliveredAdvice(_)) {
            tx.execute("INSERT INTO source_delivery_event(delivery,what,at,detail,request,commit_seq) VALUES(?1,'delivered',?2,'written to the harness hook output',?1,1)",params![request.id,request.at.unix_ms()])?;
        }
        Ok(PreparedRequest {
            request: request.clone(),
            outcome: UserOutcome { pending: false, id: request.id.clone(), repeat: false },
            effect: capture(tx, request, 1)?,
        })
    }).unwrap()
}

fn counted_delivery_preparation(
    conn: &Connection,
    request: &UserRequest,
) -> (PreparedRequest, usize) {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let instructions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&instructions);
    conn.progress_handler(
        1,
        Some(move || {
            observed.fetch_add(1, Ordering::Relaxed);
            false
        }),
    );
    let prepared = prepare(
        conn,
        request,
        None,
        None,
        |_| Ok(()),
        |_| panic!("a literal delivery has no attachment acquisition"),
    );
    conn.progress_handler(0, None::<fn() -> bool>);
    (prepared.unwrap(), instructions.load(Ordering::Relaxed))
}

#[test]
fn literal_deliveries_match_transactional_records_without_touching_history() {
    let mut conn = cache();
    for n in 0..128 {
        let mut old = literal_delivery(false);
        old.id = format!("old-{n}");
        if let UserWrite::Deliver(body) = &mut old.write {
            body.text = "history ".repeat(512);
        }
        let p = prepare_with(&conn, &old, &[], &[]).unwrap();
        commit(&mut conn, &p, n + 1);
    }
    for advice in [false, true] {
        let request = literal_delivery(advice);
        let expected = delivery_from_transaction(&conn, &request);
        let (empty, empty_steps) = counted_delivery_preparation(&cache(), &request);
        let (prepared, steps) = counted_delivery_preparation(&conn, &request);
        assert_eq!(prepared, expected);
        assert_eq!(prepared, empty);
        // A keyed request lookup stays proportional to this delivery, even when
        // the archive holds large unrelated delivery bodies and request records.
        assert!(
            steps <= empty_steps * 2,
            "empty={empty_steps}, history={steps}"
        );
        println!("delivery preparation: empty={empty_steps} VM steps, history={steps} VM steps");
        assert_eq!(count(&conn, "delivery"), 128);
        assert!(known_request(&conn, &request).unwrap().is_none());
    }
}

#[test]
fn literal_delivery_retry_keeps_pending_body_and_refuses_changed_intent() {
    for advice in [false, true] {
        let request = literal_delivery(advice);
        let conn = cache();
        let original = prepare_with(&conn, &request, &[], &[]).unwrap();
        let pending = polylogue_model::authored::PendingAuthored::Live {
            prepared: original.clone(),
        };
        let reopened: polylogue_model::authored::PendingAuthored =
            serde_json::from_slice(&serde_json::to_vec(&pending).unwrap()).unwrap();
        let mut retry = request.clone();
        retry.at = Timestamp::from_unix_ms(999).unwrap();
        assert_eq!(
            prepare(
                &conn,
                &retry,
                None,
                None,
                |visit| visit(&reopened),
                |_| Ok(())
            )
            .unwrap(),
            original
        );
        match &mut retry.write {
            UserWrite::Deliver(body) | UserWrite::DeliveredAdvice(body) => {
                body.refs.push("message:codex:changed".into())
            }
            _ => unreachable!(),
        }
        assert!(matches!(
            prepare(
                &conn,
                &retry,
                None,
                None,
                |visit| visit(&reopened),
                |_| Ok(())
            ),
            Err(StoreError::Conflict(_))
        ));
        assert_eq!(count(&conn, "delivery"), 0);
    }
}

#[test]
fn literal_delivery_refuses_effects_that_change_content_references_or_receipts() {
    for advice in [false, true] {
        let request = literal_delivery(advice);
        let conn = cache();
        let prepared = prepare_with(&conn, &request, &[], &[]).unwrap();
        for mutation in 0..4 {
            let mut changed = prepared.clone();
            let delivery = &mut changed.effect.deliveries[0];
            match mutation {
                0 => delivery.delivery.prepared.text.push_str(" changed"),
                1 => delivery
                    .delivery
                    .prepared
                    .refs
                    .push("message:codex:other".into()),
                2 => delivery.delivery.id = "different-delivery".into(),
                _ => delivery
                    .added_events
                    .push(polylogue_model::authored::DeliveryEvent {
                        request: "different-request".into(),
                        what: polylogue_model::DeliveryOutcome::Delivered,
                        at: request.at,
                        detail: None,
                    }),
            }
            let mut target = cache();
            let tx = target.transaction().unwrap();
            assert!(matches!(
                apply(&tx, &changed, 1),
                Err(StoreError::Invalid(_))
            ));
            tx.rollback().unwrap();
            assert_eq!(count(&target, "delivery"), 0);
            assert_eq!(count(&target, "client_request"), 0);
        }
    }
}
