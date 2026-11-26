use super::*;
use serde_json::{Value, json};

#[test]
fn a_pending_write_is_degraded_with_its_note_and_http_success() {
    let write_result = json!({"id":"record:one", "pending":true});
    let answer = written(write_result.clone(), true);
    assert_eq!(answer.exit_code(), 5);
    assert_eq!(answer.http_status(), Some(200));
    assert_eq!(answer.value(), Some(&write_result));
    assert_eq!(
        crate::protocol::notes(&answer).collect::<Vec<_>>(),
        [AWAITING_CONFIRMATION]
    );
    assert!(matches!(
        written(write_result, false),
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(_),
            ..
        }
    ));
}

#[test]
fn failures_follow_the_shared_table() {
    use FailureKind as K;
    use polylogue_model::Code::*;
    // Expected transport behavior is written from the Exit codes table.
    let rows: &[(K, u8, Option<u16>, bool, &[polylogue_model::Code])] = &[
        (
            K::InvalidQuery,
            2,
            Some(400),
            false,
            &[Query, Sql, InvalidQuery],
        ),
        (K::UnknownField, 2, Some(400), false, &[UnknownField]),
        (K::BadValue, 2, Some(400), false, &[BadValue]),
        (K::NotAQuery, 2, Some(400), false, &[NotAQuery]),
        (
            K::BadRequest,
            2,
            Some(400),
            false,
            &[
                BadRequest,
                Invalid,
                IncompleteParents,
                UnsupportedProvider,
                EmptyText,
                ModelRequired,
                AttachmentNameNotABasename,
                InvalidAction,
                SubmissionReceiptRequired,
                SubmissionConversationMismatch,
                SinceIsADateOrTime,
                UnknownOutcome,
                UnknownAttachment,
                UnknownReport,
                NotAVerb,
                NeedsAll,
                RowNumbersOffTerminal,
            ],
        ),
        (K::Ambiguous, 2, Some(400), false, &[Ambiguous]),
        (K::StaleCursor, 2, Some(400), false, &[StaleCursor]),
        (K::NotBuilt, 2, Some(501), false, &[NotBuilt]),
        (
            K::NotYetAvailable,
            2,
            Some(501),
            true,
            &[
                NotYetAvailable,
                IndexBuilding,
                PrimaryContextMissing,
                PrimaryContextPending,
                PrimaryContextStale,
            ],
        ),
        (
            K::Refused,
            3,
            Some(403),
            false,
            &[
                ModelCallsOff,
                Refused,
                OperatorRequired,
                EmbeddingsOff,
                Confirm,
                Rejected,
                PrimaryContextRefused,
            ],
        ),
        (
            K::NotFound,
            4,
            Some(404),
            false,
            &[
                NotFound,
                UnknownChatAction,
                UnknownChatActionAttachment,
                AttachmentMissing,
                AttachmentUnavailable,
                UnknownImportJob,
            ],
        ),
        (
            K::Conflict,
            4,
            Some(409),
            false,
            &[
                Conflict,
                Stale,
                ScopeChanged,
                ResyncRequired,
                IdempotencyKeyReused,
                ActionFinished,
                NotTheClaimant,
                StageOnlyActionCannotSubmit,
                NotAwaitingApproval,
                JobEnded,
                BrowserActionsOff,
            ],
        ),
        (K::StoppedEarly, 5, Some(500), false, &[StoppedEarly]),
        (
            K::NotRunning,
            6,
            None,
            true,
            &[
                NotRunning,
                DaemonRequired,
                ShuttingDown,
                DaemonProtocolMismatch,
            ],
        ),
        (K::Timeout, 7, Some(504), true, &[DeadlineExceeded, Busy]),
        (K::Internal, 4, Some(500), false, &[Internal]),
        (K::Cancelled, 7, None, true, &[Cancelled]),
        (
            K::Io,
            4,
            Some(500),
            false,
            &[
                Demo,
                AttachmentOutput,
                ReaderUnavailable,
                Schema,
                ReadOutput,
                Io,
                BlobCorrupt,
                Store,
                Import,
                Embed,
                SinexUnavailable,
                PrimaryContextUnavailable,
                PrimaryContextInvalid,
            ],
        ),
    ];
    let mut seen = std::collections::BTreeSet::new();
    for &(kind, exit, http, retryable, codes) in rows {
        for &code in codes {
            assert!(seen.insert(code), "duplicate {code:?}");
            let answer = Answer::<Value>::error(failure(code, "failed"));
            let error = answer.failure().unwrap();
            assert_eq!(
                (
                    error.kind,
                    answer.exit_code(),
                    answer.http_status(),
                    error.retryable
                ),
                (kind, exit, http, retryable),
                "{code:?}"
            );
        }
    }
    assert_eq!(seen, polylogue_model::Code::ALL.iter().copied().collect());
    for (answer, exit) in [
        (Answer::empty(Value::Null), 1),
        (
            Answer::degraded(Value::Null, vec!["missing source".into()]),
            5,
        ),
    ] {
        assert_eq!(
            (answer.exit_code(), answer.http_status(), answer.failure()),
            (exit, Some(200), None)
        );
    }
}

#[test]
fn a_degraded_answer_keeps_its_outcome_and_notes_on_the_socket_wire() {
    let answer = written(json!({"id":"record:pending", "pending":true}), true);
    let bytes = serde_json::to_vec(&answer).unwrap();
    let read: Answer<Value> = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(read, answer);
    assert_eq!(
        crate::protocol::notes(&read).collect::<Vec<_>>(),
        [AWAITING_CONFIRMATION]
    );
    assert_eq!(read.exit_code(), 5);
}

#[test]
fn bundle_failures_distinguish_conflicts_from_invalid_input() {
    use polylogue_model::bundle::BundleError;
    for (issue, kind) in [
        (BundleError::BlockSelectionConflict, FailureKind::Conflict),
        (BundleError::SourceVersionMismatch, FailureKind::Conflict),
        (BundleError::InvalidBlockSelection, FailureKind::BadRequest),
    ] {
        let answer =
            crate::daemon::service::store_error::<()>(&polylogue_store::StoreError::Bundle(issue));
        let command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } = answer
        else {
            panic!("block selection failure is not success")
        };
        assert_eq!(error.kind, kind);
    }
}

#[test]
fn an_ambiguous_reference_names_every_candidate() {
    let error = polylogue_store::StoreError::Ambiguous {
        reference: "7f3a".into(),
        candidates: vec!["session:test:7f3a-1".into(), "session:test:7f3a-2".into()],
    };
    let failure = store_failure(&error);
    assert_eq!(failure.kind, FailureKind::Ambiguous);
    assert_eq!(
        failure.candidates(),
        ["session:test:7f3a-1", "session:test:7f3a-2"]
    );
    let selection = polylogue_store::StoreError::Selection {
        code: polylogue_model::Code::StaleCursor,
        message: "the selected row position is unavailable".into(),
    };
    assert_eq!(store_failure(&selection).kind, FailureKind::StaleCursor);
    assert_eq!(
        store_failure(&selection).code.as_deref(),
        Some("stale_cursor")
    );
}

#[test]
fn noncurrent_primary_reads_keep_their_data_and_say_why() {
    use polylogue_model::freshness::IndexFreshness;
    for (freshness, reason) in [
        (
            IndexFreshness::primary(Some(5), Some(8), Some(0.5), Some(true), None),
            "the index is behind the observed Sinex position",
        ),
        (
            IndexFreshness::primary(Some(5), Some(8), Some(9.0), Some(true), None),
            "the Sinex position is unknown",
        ),
        (
            IndexFreshness::primary(Some(5), None, None, Some(false), Some("unreachable".into())),
            "Sinex is unavailable",
        ),
    ] {
        let value = json!({"rows":[{"ref":"session:synthetic"}]});
        let freshness = polylogue_model::freshness::ReadFreshness {
            local: polylogue_model::freshness::ReadCompleteness::empty(),
            sinex: freshness.clone(),
        };
        let result = read(value.clone(), &freshness);
        assert_eq!(result.exit_code(), 5);
        assert_eq!(result.http_status(), Some(200));
        assert_eq!(result.value(), Some(&value));
        assert_eq!(
            crate::protocol::notes(&result).collect::<Vec<_>>(),
            [reason],
            "{freshness:?}"
        );
    }
    let current = IndexFreshness::primary(Some(8), Some(8), Some(0.5), Some(true), None);
    assert!(matches!(
        read(
            json!({}),
            &polylogue_model::freshness::ReadFreshness {
                local: polylogue_model::freshness::ReadCompleteness::empty(),
                sinex: current
            }
        ),
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(_),
            ..
        }
    ));
}

#[test]
fn typed_failure_codes_keep_their_wire_names_and_classes() {
    use polylogue_model::Code;
    for (code, wire, expected) in [
        (Code::Io, "io", FailureKind::Io),
        (Code::ModelCallsOff, "model_calls_off", FailureKind::Refused),
        (Code::Store, "store", FailureKind::Io),
        (Code::Internal, "internal", FailureKind::Internal),
        (
            Code::InvalidQuery,
            "invalid_query",
            FailureKind::InvalidQuery,
        ),
        (Code::StaleCursor, "stale_cursor", FailureKind::StaleCursor),
        (
            Code::NotYetAvailable,
            "not_yet_available",
            FailureKind::NotYetAvailable,
        ),
    ] {
        let result = failure(code, "synthetic failure");
        assert_eq!(result.kind, expected);
        assert_eq!(result.code.as_deref(), Some(wire));
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["code"], wire);
        let decoded: command_surface::Failure = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.kind, expected);
        assert_eq!(decoded.code.as_deref(), Some(wire));
    }
}

#[test]
fn every_store_error_variant_has_a_code_and_kind() {
    use polylogue_model::Code;
    use polylogue_store::StoreError as E;
    let busy =
        rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY), None);
    let full = rusqlite::Error::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CORRUPT),
        None,
    );
    let hash = polylogue_model::BlobHash([7; 32]);
    let table: Vec<(E, Code, FailureKind, bool)> = vec![
        (E::Sqlite(busy), Code::Busy, FailureKind::Timeout, true),
        (E::Sqlite(full), Code::Store, FailureKind::Io, false),
        (
            E::Io(std::io::ErrorKind::PermissionDenied.into()),
            Code::Io,
            FailureKind::Io,
            false,
        ),
        (
            E::Blobs(hashed_blobs::Error::Corrupt {
                segment: "s".into(),
                offset: 0,
                len: 1,
                what: "hash".into(),
            }),
            Code::BlobCorrupt,
            FailureKind::Io,
            false,
        ),
        (
            E::Blobs(hashed_blobs::Error::Unwrap("key".into())),
            Code::Io,
            FailureKind::Io,
            false,
        ),
        (
            E::Json(serde_json::from_str::<u8>("x").unwrap_err()),
            Code::Store,
            FailureKind::Io,
            false,
        ),
        (
            E::Bundle(polylogue_model::bundle::BundleError::InvalidPath),
            Code::BadRequest,
            FailureKind::BadRequest,
            false,
        ),
        (
            E::Locked {
                path: "/home/op/archive".into(),
                holder: Some(7),
            },
            Code::Conflict,
            FailureKind::Conflict,
            true,
        ),
        (
            E::NewerSchema {
                database: "index",
                found: 9,
                current: 8,
            },
            Code::Refused,
            FailureKind::Refused,
            false,
        ),
        (
            E::OtherFormat {
                database: "archive",
                found: 1,
            },
            Code::Refused,
            FailureKind::Refused,
            false,
        ),
        (
            E::EmbeddingsSchemaDiffers {
                difference: "a column".into(),
            },
            Code::Refused,
            FailureKind::Refused,
            false,
        ),
        (
            E::OlderFormat {
                database: "archive",
            },
            Code::Refused,
            FailureKind::Refused,
            false,
        ),
        (
            E::IndexVersion {
                found: 1,
                wanted: 2,
            },
            Code::IndexBuilding,
            FailureKind::NotYetAvailable,
            true,
        ),
        (
            E::BlobCorrupt(hash),
            Code::BlobCorrupt,
            FailureKind::Io,
            false,
        ),
        (
            E::Counts(polylogue_model::CountMismatch {
                records: 2,
                counted: 1,
            }),
            Code::Internal,
            FailureKind::Internal,
            false,
        ),
        (
            E::WriterGone,
            Code::NotRunning,
            FailureKind::NotRunning,
            true,
        ),
        (
            E::Stopped,
            Code::ShuttingDown,
            FailureKind::NotRunning,
            true,
        ),
        (
            E::StorageFailed(polylogue_store::storage_failure::StorageFailure {
                path: "/home/op/archive".into(),
                message: "sync failed".into(),
                errno: Some(5),
            }),
            Code::Io,
            FailureKind::Io,
            false,
        ),
        (
            E::ReadDeadlineExceeded,
            Code::DeadlineExceeded,
            FailureKind::Timeout,
            true,
        ),
        (
            E::ReadCancelled,
            Code::Cancelled,
            FailureKind::Cancelled,
            true,
        ),
        (
            E::ReadPoolClosed,
            Code::ShuttingDown,
            FailureKind::NotRunning,
            true,
        ),
        (
            E::ReadPoolNested,
            Code::Internal,
            FailureKind::Internal,
            false,
        ),
        (
            E::ScopeChanged,
            Code::ScopeChanged,
            FailureKind::Conflict,
            true,
        ),
        (
            E::ReadIndexReplaced,
            Code::Conflict,
            FailureKind::Conflict,
            true,
        ),
        (
            E::IndexBuilding(Box::new(
                polylogue_store::archive::index_build::IndexBuild {
                    serving_previous: false,
                    chains: 2,
                    chains_read: 1,
                    bytes: 10,
                    bytes_read: 5,
                },
            )),
            Code::IndexBuilding,
            FailureKind::NotYetAvailable,
            true,
        ),
        (
            E::Conflict("request-1".into()),
            Code::Conflict,
            FailureKind::Conflict,
            false,
        ),
        (
            E::Stale {
                reference: "annotation:a@1".into(),
                current: vec!["annotation:a@2".into()],
            },
            Code::Stale,
            FailureKind::Conflict,
            false,
        ),
        (
            E::Ambiguous {
                reference: "query:ab".into(),
                candidates: vec!["query:ab1".into(), "query:ab2".into()],
            },
            Code::Ambiguous,
            FailureKind::Ambiguous,
            false,
        ),
        (
            E::IncompleteParents(Box::new(polylogue_model::read::ParentIssue::Cycle {
                session: polylogue_model::SessionId::parse("codex:s").unwrap(),
            })),
            Code::IncompleteParents,
            FailureKind::BadRequest,
            false,
        ),
        (
            E::NotFound("session:codex:s".into()),
            Code::NotFound,
            FailureKind::NotFound,
            false,
        ),
        (
            E::ChatAction("action_finished"),
            Code::Conflict,
            FailureKind::Conflict,
            false,
        ),
        (
            E::ImportJob("job_ended"),
            Code::Conflict,
            FailureKind::Conflict,
            false,
        ),
        (
            E::Invalid("refused".into()),
            Code::BadRequest,
            FailureKind::BadRequest,
            false,
        ),
        (
            E::Refused(polylogue_model::Reason::NoStatements {}),
            Code::BadRequest,
            FailureKind::BadRequest,
            false,
        ),
        (
            E::internal("an invariant"),
            Code::Internal,
            FailureKind::Internal,
            false,
        ),
        (
            E::RowsRefused {
                stored: polylogue_store::staging::StoredInput {
                    input: "input".into(),
                    chain: 1,
                    length: 3,
                    already_stored: false,
                    bytes_written: 3,
                },
                reason: "a row".into(),
                cause: polylogue_model::Cause::Content,
            },
            Code::Import,
            FailureKind::Io,
            false,
        ),
        (
            E::stored_data("an extent gap"),
            Code::Store,
            FailureKind::Io,
            false,
        ),
        (
            E::Query(polylogue_store::query::QueryError::NotBuilt {
                what: "a feature".into(),
            }),
            Code::NotBuilt,
            FailureKind::NotBuilt,
            false,
        ),
        (
            E::ReferenceUnavailable {
                reference: "message:codex:m".into(),
                reason: "pending".into(),
            },
            Code::NotYetAvailable,
            FailureKind::NotYetAvailable,
            true,
        ),
        (
            E::Selection {
                code: Code::StaleCursor,
                message: "moved".into(),
            },
            Code::StaleCursor,
            FailureKind::StaleCursor,
            false,
        ),
        (
            E::Failed(polylogue_model::Failure::of(&E::Sqlite(
                rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
                    None,
                ),
            ))),
            Code::Busy,
            FailureKind::Timeout,
            true,
        ),
        // An engine invariant breach is a bug, never retryable I/O.
        (
            E::Failed(polylogue_model::Failure::of(
                &polylogue_engine::EngineError::from(E::internal(
                    "a replacement needs a stored chain",
                )),
            )),
            Code::Internal,
            FailureKind::Internal,
            false,
        ),
    ];
    for (error, code, kind, retryable) in table {
        let failure = store_failure(&error);
        assert_eq!(error.code(), code, "{error:?}");
        assert_eq!(failure.code.as_deref(), Some(code.as_str()), "{error:?}");
        assert_eq!(failure.kind, kind, "{error:?}");
        assert_eq!(failure.retryable, retryable, "{error:?}");
    }
}

#[test]
fn user_impossible_invariants_are_internal() {
    let error = polylogue_store::StoreError::internal("a selected row has no session");
    let failure = store_failure(&error);
    assert_eq!(failure.kind, FailureKind::Internal);
    assert_eq!(failure.code.as_deref(), Some("internal"));
    assert_eq!(
        failure.message,
        "a selected row has no session; this is a Polylogue bug; report it with the request ID"
    );
    let daemon = crate::daemon::Failure::internal("an accepted line has no request");
    assert_eq!(daemon.kind, FailureKind::Internal);
    assert_eq!(
        daemon.message,
        "an accepted line has no request; this is a Polylogue bug; report it with the request ID"
    );
}

/// The operation table registers every code with its kind once; a code
/// a failure carries is always one of them, so a reason's code is too.
#[test]
fn every_failure_code_is_registered() {
    let _ = crate::commands::declarations();
    for code in polylogue_model::Code::ALL {
        let failure = failure(*code, "failed");
        assert_eq!(failure.kind, kind(*code), "{code:?}");
        assert_eq!(
            failure
                .code
                .as_deref()
                .and_then(polylogue_model::Code::from_name),
            Some(*code)
        );
    }
}
