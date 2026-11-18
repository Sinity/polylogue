use super::*;
use polylogue_model::{
    Reference,
    authorship::{Authorship, Channel},
    user::{Save, UserWrite},
};
use polylogue_store::{user_writes::PreparationContext, writer::UserSubmission};
use serde_json::json;

fn save(seed: &polylogue_store::fixture::writer::WriterArchive, id: &str) {
    seed.writer
        .user(UserSubmission {
            request: polylogue_store::fixture::authored::operator_request(
                id.into(),
                UserWrite::Save(Box::new(Save {
                    name: Some("pending-focus".into()),
                    query: "sessions".into(),
                    denominator: None,
                    rows_json: "{\"unit\":\"sessions\",\"columns\":[],\"rows\":[]}".into(),
                    read_at: "{}".into(),
                    watch: None,
                    outcomes: None,
                })),
            ),
            scope: None,
            preparation: Some(PreparationContext::new(vec!["codex".into()])),
            parent: None,
        })
        .unwrap();
}
fn authored(
    seed: &polylogue_store::fixture::writer::WriterArchive,
    id: &str,
) -> polylogue_model::authored::PreparedRequest {
    let pool = polylogue_store::ReadPool::open(&seed.archive).unwrap();
    let mut found = None;
    pool.reader()
        .unwrap()
        .visit_authored(&mut |record| {
            if record.request.id == id {
                found = Some(record.clone());
            }
            Ok(())
        })
        .unwrap();
    found.unwrap()
}

#[test]
fn an_offline_withdrawal_retry_keeps_its_original_saved_family() {
    use crate::daemon::verbs::{Given, Selection, StepCall, Verb, annotate::Withdraw};
    let home = ClaudeHome::new();
    let runtime = launch(primary(&home)).unwrap();
    let denied = primary_table_call(
        &runtime,
        "rows",
        json!({"selection":"query:pending-focus",
        "steps":[{"verb":"withdraw"}]}),
        "no-context",
    );
    assert!(
        matches!(denied.outcome, command_surface::Outcome::Error(error)
        if matches!(error.code.as_deref(), Some("primary_context_missing" | "primary_context_unavailable")))
    );
    assert!(pending_requests(&runtime).is_empty());
    let ModeState::Primary { connection } = &runtime.daemon.mode else {
        panic!("primary required")
    };
    let connection = connection.clone();
    drop(runtime);
    let seed = polylogue_store::fixture::writer::WriterArchive::new();
    save(&seed, "first-definition");
    let original_definition = authored(&seed, "first-definition");
    let old_root = Reference::Assertion(polylogue_model::RevisionReference {
        root: original_definition
            .effect
            .assertions
            .iter()
            .find(|fact| fact.kind == polylogue_model::assertion::AssertionKind::SavedQuery)
            .unwrap()
            .root
            .clone(),
        revision: None,
    });
    confirm_authored(
        &home,
        connection,
        vec![(1, "synthetic-withdrawal-1", original_definition)],
    );
    let runtime = launch(primary(&home)).unwrap();
    // The line's already-resolved family crosses the actual verb and writer,
    // without inventing public context for an offline primary connection.
    let withdraw = |runtime: &Runtime| {
        let invocation = runtime.daemon.invocation(&Authorship {
            author: polylogue_model::Author::Operator,
            via: Channel::Cli,
        });
        let caller = crate::commands::Caller {
            surface: crate::commands::Surface::Cli,
            cwd: None,
        };
        let acquired = std::sync::atomic::AtomicBool::new(false);
        let call = crate::daemon::handler::Call {
            daemon: &runtime.daemon,
            row: crate::commands::command("rows").unwrap(),
            invocation: &invocation,
            caller: &caller,
            request_id: "offline-withdrawal",
            cursor: None,
            budget: Default::default(),
            reader_acquired: &acquired,
        };
        let selection = Selection {
            given: Given::Query("query:pending-focus".into()),
            resolved: Some(vec![old_root.clone()]),
            question: Some("query:pending-focus".into()),
            retained: None,
        };
        let done = Withdraw::run(
            Default::default(),
            &selection,
            &StepCall::new(&call, "offline-withdrawal-1"),
        )
        .unwrap();
        let crate::protocol::line::StepAnswer::Wrote(report) = done.reply else {
            panic!("expected withdrawal")
        };
        report
    };
    let original = withdraw(&runtime);
    assert!(original.pending && !original.repeat);
    let pending = pending_requests(&runtime);
    assert!(
        pending
            .iter()
            .any(|record| record.request.id == "offline-withdrawal-1")
    );
    let ModeState::Primary { connection } = &runtime.daemon.mode else {
        panic!("primary required")
    };
    let connection = connection.clone();
    drop(runtime);
    seed.writer
        .user(polylogue_store::fixture::authored::operator_request(
            "other-withdrawal".into(),
            UserWrite::WithdrawAssertions {
                targets: vec![old_root.clone()],
                reason: None,
            },
        ))
        .unwrap();
    save(&seed, "replacement-definition");
    let replacement = authored(&seed, "replacement-definition");
    let new_root = replacement
        .effect
        .assertions
        .iter()
        .find(|fact| fact.kind == polylogue_model::assertion::AssertionKind::SavedQuery)
        .unwrap()
        .root
        .clone();
    confirm_authored(
        &home,
        connection,
        vec![
            (
                2,
                "synthetic-withdrawal-2",
                authored(&seed, "other-withdrawal"),
            ),
            (3, "synthetic-withdrawal-3", replacement),
        ],
    );
    let runtime = launch(primary(&home)).unwrap();
    let expected = Some(Reference::Assertion(polylogue_model::RevisionReference {
        root: new_root,
        revision: None,
    }));
    assert_eq!(
        runtime
            .daemon
            .reads
            .reader()
            .unwrap()
            .saved_query_target("pending-focus")
            .unwrap(),
        expected
    );
    let repeat = withdraw(&runtime);
    assert!(repeat.pending && repeat.repeat);
    assert_eq!(repeat.id, original.id);
    assert_eq!(pending_requests(&runtime), pending);
    assert_eq!(
        runtime
            .daemon
            .reads
            .reader()
            .unwrap()
            .saved_query_target("pending-focus")
            .unwrap(),
        expected
    );
}
