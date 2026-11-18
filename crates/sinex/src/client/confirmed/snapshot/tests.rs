use super::*;
use polylogue_model::batch::StepProgress;
use sinex_protocol::{ConfirmedConsumer, ConfirmedConsumerReply, FetchEvent, FetchEventReply};
use sinex_protocol::{
    ReplayQuery, ReplayReply, SnapshotCursor, SnapshotNext, SnapshotReading, SnapshotReply,
};

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

fn roundtrip(wrong_position: bool) {
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
                frame_root: root.path().into(),
                remote_upload: false,
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
            let last = sinex_protocol::LastStreamMessage { sequence: 4, fingerprint: "a".repeat(64) };
            let cursor = SnapshotCursor { position: 21, scope: "scope".into(), catalog: "catalog".into(), stream_instance: "stream".into(), last: Some(last.clone()), after: Some((event.commit_seq, event.id)) };
            let withheld = SnapshotReading::Withheld(sinex_protocol::WithheldEvent { commit_seq: event.commit_seq, event: event.id, event_type: event.event_type.clone(), class: sinex_model::Class::Guarded, reason: sinex_protocol::WithheldReason::Class });
            let entry = |reading: SnapshotReading| sinex_protocol::SnapshotEvent { event: reading, archived: true, lifecycle_position: Some(19) };
            let (expected_cursor, expected_last) = (cursor.clone(), last.clone());
            let served = event.clone();
            let fake_sinex = tokio::spawn(async move {
                for step in 0..6 {
                    let request = consumers.next().await.unwrap();
                    let query: ConfirmedConsumer = serde_json::from_slice(&request.payload).unwrap();
                    assert_eq!(query.cache, "cache");
                    assert!(!query.status_only);
                    if step == 5 {
                        let start = sinex_protocol::SnapshotStart { stream_instance: "stream".into(), last: Some(expected_last) };
                        assert_eq!(query.start, (!wrong_position).then_some(start));
                        assert_eq!(query.after, if wrong_position { 0 } else { 21 });
                        reply(&daemon, request, ConfirmedConsumerReply::Refused(sinex_protocol::FetchRefusal { kind: sinex_protocol::FetchRefusalKind::Unavailable, reason: "fixture finished".into() })).await;
                        break;
                    }
                    assert_eq!(query.after, 0);
                    assert!(query.start.is_none());
                    reply(&daemon, request, ConfirmedConsumerReply::SnapshotRequired { origin: 10, position: 21, scope: "scope".into(), stream_instance: "stream".into() }).await;
                    if step == 0 { continue; }
                    let request = replay.next().await.unwrap();
                    let ReplayQuery::Snapshot { cursor: asked } = serde_json::from_slice(&request.payload).unwrap() else { panic!("snapshot expected"); };
                    assert_eq!(asked, matches!(step, 2 | 3).then(|| expected_cursor.clone()));
                    let answer = match step {
                        1 => SnapshotReply::Page { events: vec![entry(withheld.clone())], next: SnapshotNext::More { cursor: expected_cursor.clone() } },
                        2 => SnapshotReply::Unavailable { reason: "sinex unavailable".into() },
                        3 => SnapshotReply::Restart { reason: "snapshot_scope_changed".into() },
                        _ => SnapshotReply::Page {
                            events: vec![entry(if wrong_position { withheld.clone() } else { SnapshotReading::Inline(Box::new(served.clone())) })],
                            next: SnapshotNext::Complete { complete: sinex_protocol::SnapshotComplete { position: 21, scope: "scope".into(), stream_instance: "stream".into(), last: Some(expected_last.clone()) } },
                        },
                    };
                    reply(&daemon, request, ReplayReply::Snapshot { reply: answer }).await;
                    if step == 1 || (step == 4 && wrong_position) {
                        let request = fetches.next().await.unwrap();
                        let query: FetchEvent = serde_json::from_slice(&request.payload).unwrap();
                        assert_eq!(query.event, served.id);
                        let mut fetched = served.clone();
                        if step == 4 { fetched.commit_seq += 1; }
                        reply(&daemon, request, FetchEventReply::Event(Box::new(fetched))).await;
                    }
                }
            });
            let initial = client.page(&ReplayState::default(), "cache").await.unwrap();
            assert!(matches!(initial[0].change, ConfirmedChange::Control(ref c) if matches!(c.as_ref(), ReplayControl::SnapshotStart { origin: 10, .. })));
            let mut state = ReplayState { scope: Some("scope".into()), snapshot_progress: Some(local::SnapshotProgress { origin: 10, stream_instance: "stream".into(), position: None, cursor: None }), ..Default::default() };
            let page = client.page(&state, "cache").await.unwrap();
            assert_eq!(page[0].position, 17);
            assert_eq!(page[1].position, 19);
            assert!(matches!(page[1].change, ConfirmedChange::Archived));
            let ConfirmedChange::Control(control) = &page[2].change else { panic!("a page control") };
            let ReplayControl::SnapshotPage { position: 21, cursor: saved } = control.as_ref() else { panic!("the next cursor") };
            assert_eq!(serde_json::from_str::<SnapshotCursor>(saved).unwrap(), cursor);
            let progress = state.snapshot_progress.as_mut().unwrap();
            progress.position = Some(21);
            progress.cursor = Some(saved.clone());
            let staged = state.clone();
            assert!(client.page(&state, "cache").await.unwrap_err().to_string().contains("sinex unavailable"));
            assert_eq!(state, staged);
            let reset = client.page(&state, "cache").await.unwrap();
            assert!(matches!(reset[0].change, ConfirmedChange::Control(ref c) if matches!(c.as_ref(), ReplayControl::SnapshotStart { reason: Some(reason), .. } if reason == "snapshot_scope_changed")));
            state.snapshot_progress.as_mut().unwrap().position = None;
            state.snapshot_progress.as_mut().unwrap().cursor = None;
            let final_page = client.page(&state, "cache").await;
            if wrong_position {
                assert!(final_page.unwrap_err().to_string().contains("differs from its position"));
            } else {
                let final_page = final_page.unwrap();
                let ConfirmedChange::Control(c) = &final_page.last().unwrap().change else { panic!("completion expected"); };
                let ReplayControl::SnapshotComplete { snapshot } = c.as_ref() else { panic!("completion expected"); };
                assert_eq!(snapshot.position, 21);
                assert_eq!(snapshot.last.as_ref().unwrap().sequence, 4);
                state.snapshot = Some(snapshot.clone());
                state.applied_position = Some(21);
                state.snapshot_progress = None;
            }
            assert!(client.page(&state, "cache").await.unwrap_err().to_string().contains("fixture finished"));
            fake_sinex.await.unwrap();
        }).await.expect("bounded snapshot transport fixture");
    });
}

#[test]
fn snapshot_transport_resumes_its_cursor_restarts_when_asked_and_names_its_start() {
    roundtrip(false);
}

#[test]
fn snapshot_transport_refuses_a_withheld_reading_fetched_at_another_position() {
    roundtrip(true);
}
