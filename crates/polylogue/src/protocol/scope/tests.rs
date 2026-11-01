use super::*;
use crate::{
    commands::{Caller, Surface},
    daemon::{Request, service::Daemon},
};
use command_surface::{Answer, Operation, OperationFuture, OperationSpec, RequestContext};
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

/// Holds the actual setting producer after selecting its data, reproducing the
/// scheduling interval between a worker answer and shared result publication.
struct HeldCatalog {
    daemon: Arc<Daemon>,
    release: Arc<Semaphore>,
    calls: Arc<AtomicUsize>,
}
impl Operation for HeldCatalog {
    type Request = crate::protocol::setting::Arguments;
    type Response = crate::protocol::setting::Answer;
    fn spec(&self) -> OperationSpec {
        let command = crate::commands::command("setting").unwrap();
        OperationSpec {
            name: command.name.into(),
            command: command.cli_words(),
            summary: command.help.into(),
            reason: command.reason.into(),
            http_route: format!("/api/v1{}", command.http_path()),
            latency: command.latency,
            failures: command.failures.to_vec(),
            write: command.writes,
        }
    }
    fn execute(
        &self,
        _: Self::Request,
        context: RequestContext,
    ) -> OperationFuture<'_, Answer<Self::Response>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let accepted = context
                .caller()
                .identity::<crate::protocol::caller::Accepted>()
                .unwrap();
            let request = accepted.admitted_request.clone().unwrap();
            let limit =
                polylogue_store::read::limit::ReadLimit::without_deadline(Default::default());
            let response = crate::daemon::service::handle_with_invocation(
                request,
                &self.daemon,
                &limit,
                accepted.invocation.as_ref().unwrap(),
            );
            let Some(data) = response.value().cloned() else {
                panic!("setting producer failed: {response:?}");
            };
            let value = serde_json::from_value(data).unwrap();
            self.release.acquire().await.unwrap().forget();
            Answer::ok(value)
        })
    }
    fn render_text(&self, answer: &Self::Response) -> String {
        serde_json::to_string(answer).unwrap()
    }
}
fn admitted(
    daemon: &Daemon,
    envelope: &command_surface::RunnableRequest,
) -> (command_surface::Caller, Option<ReadScopeToken>) {
    let command = crate::commands::command("setting").unwrap();
    let caller = Caller {
        surface: Surface::Mcp,
        cwd: None,
        json: true,
    };
    let internal =
        (command.build_typed.unwrap())(envelope.arguments.clone(), &caller, &envelope.request_id)
            .unwrap();
    let scope = crate::protocol::operations::read_scope_required(Some(command), Some(&internal))
        .then(|| daemon.reads.scope_token().unwrap());
    let mut accepted = crate::protocol::caller::accept(
        &daemon.callers,
        &Default::default(),
        polylogue_model::authorship::Channel::Mcp,
        Default::default(),
    );
    accepted.invocation = Some(
        daemon
            .invocation(&accepted.authorship)
            .with_accepted_request(Some(Arc::new(envelope.clone()))),
    );
    accepted.admitted_request = Some(internal);
    accepted.read_scope = scope.clone();
    (
        command_surface::Caller::authenticated(accepted.scope(), accepted),
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
async fn a_new_scope_waiter_cannot_publish_the_old_scope_shared_catalog() {
    let daemon = crate::test_daemon::start("native coalesced publication scope");
    let release = Arc::new(Semaphore::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut table = command_surface::OperationTable::new("polylogue", Default::default());
    table
        .register(HeldCatalog {
            daemon: Arc::clone(&daemon.daemon),
            release: Arc::clone(&release),
            calls: Arc::clone(&calls),
        })
        .unwrap();
    let envelope = command_surface::Request {
        operation: "setting".into(),
        arguments: json!({}),
        request_id: "same-id-across-generation".into(),
        deadline: command_surface::Deadline::None,
    };
    let (owner_caller, owner_scope) = admitted(&daemon.daemon, &envelope);
    let mut owner = Box::pin(table.call_as(
        envelope.clone(),
        command_surface::Origin::Mcp,
        owner_caller,
        Default::default(),
    ));
    pending(owner.as_mut()).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    daemon.daemon.archive.advance_generation_for_test().unwrap();
    assert!(owner_scope.as_ref().unwrap().check().is_err());
    let (waiter_caller, waiter_scope) = admitted(&daemon.daemon, &envelope);
    assert!(waiter_scope.as_ref().unwrap().check().is_ok());
    let mut waiter = Box::pin(table.call_as(
        envelope,
        command_surface::Origin::Mcp,
        waiter_caller,
        Default::default(),
    ));
    pending(waiter.as_mut()).await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "coalescing identity must not change with the epoch"
    );
    release.add_permits(1);
    let owner = owner.await;
    let waiter = waiter.await;
    assert_eq!(owner.answer, waiter.answer);
    assert!(
        waiter.answer.value().is_some(),
        "the shared producer returned its selected catalog"
    );
    let publication = ReadScopes::dispatched(&waiter, waiter_scope);
    let answer = super::answer(waiter.answer, &publication);
    let Answer::Error { error, .. } = answer else {
        panic!("old selected data escaped to a fresh scope");
    };
    assert_eq!(error.code.as_deref(), Some("scope_changed"));
    let published = AtomicBool::new(false);
    assert!(
        publication
            .with_current(|| {
                published.store(true, Ordering::SeqCst);
                Ok(())
            })
            .is_err()
    );
    assert!(!published.load(Ordering::SeqCst));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // The same group's actual write form uses Writer admission, not a read-output token.
    let write = command_surface::Request {
        operation: "setting".into(),
        arguments: json!({"key":"browser.actions","value":"on"}),
        request_id: "pure-write-admission".into(),
        deadline: command_surface::Deadline::None,
    };
    let (caller, scope) = admitted(&daemon.daemon, &write);
    assert!(scope.is_none());
    let accepted = caller
        .identity::<crate::protocol::caller::Accepted>()
        .unwrap();
    assert!(accepted.read_scope.is_none());
    assert!(matches!(
        accepted.admitted_request.as_ref(),
        Some(Request::Setting {
            request_id: Some(_),
            ..
        })
    ));
}
