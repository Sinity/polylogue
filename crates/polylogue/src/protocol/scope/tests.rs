use super::*;
use crate::{
    commands::{Caller, Surface},
    daemon::service::Daemon,
};
use command_surface::{Answer, Operation, OperationFuture, RequestContext};
use serde_json::json;
use std::{
    future::{Future, poll_fn},
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::Poll,
};
use tokio::sync::Semaphore;

/// Holds the setting producer after it selects its data, reproducing the
/// scheduling interval between a worker answer and handing out the shared result.
struct HeldCatalog {
    daemon: Arc<Daemon>,
    release: Arc<Semaphore>,
    calls: Arc<AtomicUsize>,
}
impl Operation for HeldCatalog {
    type Request = crate::protocol::setting::Arguments;
    type Response = crate::protocol::setting::Answer;

    fn execute(
        &self,
        arguments: Self::Request,
        context: RequestContext,
    ) -> OperationFuture<'_, Answer<Self::Response>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let accepted = context
                .caller()
                .identity::<crate::operation::Accepted>()
                .unwrap();
            let limit =
                polylogue_store::read::limit::ReadLimit::without_deadline(Default::default());
            let response = crate::daemon::handler::local(
                "set",
                serde_json::to_value(arguments).unwrap(),
                &crate::daemon::handler::Local {
                    daemon: &self.daemon,
                    invocation: accepted.invocation.as_ref().unwrap(),
                    caller: &Caller {
                        surface: Surface::Mcp,
                        cwd: None,
                    },
                    limit: &limit,
                    request_id: &context.request_id,
                    cursor: None,
                },
            );
            let Some(data) = response.value().cloned() else {
                panic!("setting producer failed: {response:?}");
            };
            let value = serde_json::from_value(data).unwrap();
            self.release.acquire().await.unwrap().forget();
            Answer::ok(value)
        })
    }
}
fn accept_request(
    daemon: &Daemon,
    envelope: &command_surface::RunnableRequest,
) -> (command_surface::Caller, Option<ReadScopeToken>) {
    let command = crate::commands::command("set").unwrap();
    let caller = Caller {
        surface: Surface::Mcp,
        cwd: None,
    };
    let policy = command
        .registration()
        .unwrap()
        .admit(&envelope.arguments, command, &caller)
        .unwrap();
    let scope = policy
        .read_only
        .then(|| daemon.reads.scope_token().unwrap());
    let mut accepted = crate::operation::Accepted::new(
        &daemon.callers,
        &Default::default(),
        command_surface::Via::Mcp,
        Default::default(),
    );
    accepted.invocation = Some(
        daemon
            .invocation(&accepted.authorship)
            .with_accepted_request(Some(Arc::new(envelope.clone()))),
    );
    accepted.read_scope = scope.clone();
    (
        command_surface::Caller::authenticated(accepted.scope(), accepted)
            .with_access(command_surface::Access::Write),
        scope,
    )
}
async fn pending<T>(mut future: Pin<&mut impl Future<Output = T>>) {
    poll_fn(|cx| {
        assert!(
            future.as_mut().poll(cx).is_pending(),
            "held flight unexpectedly completed"
        );
        Poll::Ready(())
    })
    .await;
}

#[tokio::test]
async fn each_request_answers_only_in_its_own_scope() {
    let daemon = crate::test_daemon::start("per-request answer scope");
    let release = Arc::new(Semaphore::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut declarations =
        command_surface::OperationDeclarations::builder("polylogue", Default::default());
    declarations
        .declare_tool(command_surface::ToolFamily {
            name: "write".into(),
            description: "Settings".into(),
            open_world: false,
            idempotent: true,
        })
        .unwrap();
    declarations
        .declare::<crate::protocol::setting::Arguments, crate::protocol::setting::Answer>(
            crate::commands::specification::<crate::protocol::setting::Arguments>(
                crate::commands::command("set").unwrap(),
            ),
            |answer, _| serde_json::to_string(answer).unwrap(),
        )
        .unwrap();
    let mut table = command_surface::OperationTable::new(declarations.finish());
    table
        .bind(
            "set",
            HeldCatalog {
                daemon: Arc::clone(&daemon.daemon),
                release: Arc::clone(&release),
                calls: Arc::clone(&calls),
            },
        )
        .unwrap();
    let envelope = command_surface::Request {
        operation: "set".into(),
        arguments: json!({}),
        request_id: "same-id-across-generation".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let (first_caller, first_scope) = accept_request(&daemon.daemon, &envelope);
    let mut first = Box::pin(table.call_as(
        envelope.clone(),
        command_surface::Via::Mcp,
        first_caller,
        Default::default(),
    ));
    pending(first.as_mut()).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    daemon.daemon.archive.advance_generation_for_test().unwrap();
    assert!(first_scope.as_ref().unwrap().check().is_err());
    let (waiter_caller, waiter_scope) = accept_request(&daemon.daemon, &envelope);
    assert!(waiter_scope.as_ref().unwrap().check().is_ok());
    let mut waiter = Box::pin(table.call_as(
        envelope,
        command_surface::Via::Mcp,
        waiter_caller,
        Default::default(),
    ));
    pending(waiter.as_mut()).await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "each accepted request selects its own source data"
    );
    release.add_permits(2);
    let first = first.await;
    let waiter = waiter.await;
    let fresh = ReadScopes::single(waiter_scope);
    assert!(super::answer(waiter, &fresh).value().is_some());
    let first_scopes = ReadScopes::single(first_scope);
    let answer = super::answer(first, &first_scopes);
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = answer
    else {
        panic!("old selected data escaped to a fresh scope");
    };
    assert_eq!(error.code.as_deref(), Some("scope_changed"));
    let ran = AtomicBool::new(false);
    assert!(
        first_scopes
            .with_current(|| {
                ran.store(true, Ordering::SeqCst);
                Ok(())
            })
            .is_err()
    );
    assert!(!ran.load(Ordering::SeqCst));
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    // The same group's write form is accepted by the writer, not by a read-output token.
    let write = command_surface::Request {
        operation: "set".into(),
        arguments: json!({"key":"browser.actions","value":"on"}),
        request_id: "write-only-request".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let (caller, scope) = accept_request(&daemon.daemon, &write);
    assert!(scope.is_none());
    let accepted = caller.identity::<crate::operation::Accepted>().unwrap();
    assert!(accepted.read_scope.is_none());
}
