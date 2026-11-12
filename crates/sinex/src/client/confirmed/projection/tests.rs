use super::*;
use polylogue_model::batch::StepProgress;
use sinex_protocol::{ConfirmedConsumer, ConfirmedConsumerReply, FetchEvent, FetchEventReply};

async fn reply(
    client: &async_nats::Client,
    request: async_nats::Message,
    body: impl serde::Serialize + Send + 'static,
) {
    sinex_transport::replies::publish_json(
        client,
        request.reply.unwrap().to_string(),
        body,
        Duration::from_secs(5),
    )
    .await
    .unwrap();
}

fn roundtrip(wrong_source: bool) {
    let server = crate::client::tests::server();
    let root = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(10), async {
            let daemon = async_nats::ConnectOptions::with_user_and_password("daemon".into(), "synthetic-daemon-password".into()).connect(server.url()).await.unwrap();
            let subjects = sinex_protocol::Subjects::new("auth");
            let mut consumers = daemon.subscribe(subjects.confirmed_consumer("polylogue")).await.unwrap();
            let mut replay = daemon.subscribe(subjects.replay_scope("polylogue")).await.unwrap();
            let mut fetches = daemon.subscribe(subjects.fetch_event("polylogue")).await.unwrap();
            daemon.flush().await.unwrap();
            let mut client = Client {
                client: async_nats::ConnectOptions::with_user_and_password("polylogue".into(), "synthetic-password".into()).connect(server.url()).await.unwrap(),
                subjects,
                producer: "polylogue".into(),
                family: "polylogue".into(),
                max_message: 1 << 20,
                store_root: root.path().into(),
                consumer_cache: None,
                consumer_instance: None,
                health: Default::default(),
                consumer: None,
                pending: Vec::new(),
                spool: std::sync::Arc::new(std::sync::Mutex::new(crate::spool::Spool::open(root.path(), "polylogue").unwrap())),
                publication: Default::default(),
                _backup: None,
            };
            let product = crate::client::confirmed::tests::document(StepProgress::AtEnd);
            let event = crate::client::confirmed::tests::current_event(&product);
            let expected_basis = sinex_protocol::FullProjectionComplete { stamp: "owner-signature".into(), position: 21, scope: "scope".into(), catalog: "catalog".into() };
            let basis = expected_basis.clone();
            let owner = tokio::spawn(async move {
                for step in 0..5 {
                    let request = consumers.next().await.unwrap();
                    let query: ConfirmedConsumer = serde_json::from_slice(&request.payload).unwrap();
                    assert_eq!(query.cache, "cache");
                    assert!(!query.status_only);
                    if step == 4 {
                        assert_eq!(query.basis, (!wrong_source).then_some(basis));
                        assert_eq!(query.after, if wrong_source { 0 } else { 21 });
                        reply(&daemon, request, ConfirmedConsumerReply::Refused { reason: "fixture finished".into() }).await;
                        break;
                    }
                    assert_eq!(query.after, 0);
                    assert!(query.basis.is_none());
                    reply(&daemon, request, ConfirmedConsumerReply::ProjectionRequired { origin: 10, position: 21, scope: "scope".into(), stream_instance: "stream".into() }).await;
                    if step == 0 { continue; }
                    let request = replay.next().await.unwrap();
                    let ReplayQuery::Inputs { cache, cursor } = serde_json::from_slice(&request.payload).unwrap() else { panic!("indexed snapshot expected"); };
                    assert_eq!(cache, "cache");
                    assert_eq!(cursor.as_deref(), (step == 2).then_some("next-page"));
                    if step == 2 {
                        reply(&daemon, request, ReplayReply::Inputs { result: InputReplayReply::Incomplete { reason: "owner context changed".into(), basis: sinex_protocol::ReplayBasisStatus::Invalidated { scope: Some("scope".into()) } } }).await;
                        continue;
                    }
                    let reference = sinex_protocol::InputReference {
                        event: event.id, commit_seq: event.commit_seq, archived: true, lifecycle_position: Some(19),
                        projection: event.projection.clone(), kind: sinex_model::KindRef { name: event.event_type.clone(), version: event.kind_version },
                        kind_definition_hash: event.kind_definition_hash.clone(), source: event.source_ref.clone(), source_fingerprint: event.source_fingerprint.clone(),
                    };
                    reply(&daemon, request, ReplayReply::Inputs { result: InputReplayReply::Page {
                        position: if step == 1 { 20 } else { 21 }, scope: "scope".into(), inputs: vec![reference],
                        continuation: if step == 1 { ReplayContinuation::Next { cursor: "next-page".into() } } else { ReplayContinuation::Complete { projection: basis.clone(), raw_inputs: sinex_protocol::CoverageStatus::NotRequested } },
                    } }).await;
                    let request = fetches.next().await.unwrap();
                    let query: FetchEvent = serde_json::from_slice(&request.payload).unwrap();
                    assert_eq!(query.event, event.id);
                    let mut fetched = event.clone();
                    if step == 3 && wrong_source { fetched.source_fingerprint = Some("different-source".into()); }
                    reply(&daemon, request, FetchEventReply::Event(Box::new(fetched))).await;
                }
            });
            let initial = client.page(&ReplayState::default(), "cache").await.unwrap();
            assert!(matches!(initial[0].change, ConfirmedChange::Control(ref c) if matches!(c.as_ref(), ReplayControl::ProjectionStart { origin: 10, .. })));
            let mut state = ReplayState { scope: Some("scope".into()), projection: Some(local::ProjectionReplay { origin: 10, stream_instance: "stream".into(), position: None, cursor: None }), ..Default::default() };
            let page = client.page(&state, "cache").await.unwrap();
            assert_eq!(page[0].position, 17);
            assert_eq!(page[1].position, 19);
            assert!(matches!(page[1].change, ConfirmedChange::Archived));
            let progress = state.projection.as_mut().unwrap();
            progress.position = Some(20);
            progress.cursor = Some("next-page".into());
            let reset = client.page(&state, "cache").await.unwrap();
            assert!(matches!(reset[0].change, ConfirmedChange::Control(ref c) if matches!(c.as_ref(), ReplayControl::ProjectionStart { reason: Some(reason), .. } if reason == "owner context changed")));
            state.projection.as_mut().unwrap().position = None;
            state.projection.as_mut().unwrap().cursor = None;
            let final_page = client.page(&state, "cache").await;
            if wrong_source {
                assert!(final_page.unwrap_err().to_string().contains("certified reference"));
            } else {
                let final_page = final_page.unwrap();
                let ConfirmedChange::Control(c) = &final_page.last().unwrap().change else { panic!("certificate expected"); };
                let ReplayControl::ProjectionComplete { basis } = c.as_ref() else { panic!("certificate expected"); };
                assert_eq!(basis.stamp, expected_basis.stamp);
                state.basis = Some(basis.clone());
                state.applied_position = Some(21);
                state.projection = None;
            }
            assert!(client.page(&state, "cache").await.unwrap_err().to_string().contains("fixture finished"));
            owner.await.unwrap();
        }).await.expect("bounded projection transport fixture");
    });
}

#[test]
fn projection_transport_resumes_exact_cursor_restarts_invalid_context_and_supplies_completed_basis()
{
    roundtrip(false);
}

#[test]
fn projection_transport_refuses_changed_native_source_before_supplying_a_basis() {
    roundtrip(true);
}
