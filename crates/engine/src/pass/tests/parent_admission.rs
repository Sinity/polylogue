use super::*;
use polylogue_model::authorship::{Author, Authorship, Channel};
use polylogue_model::{BlobHash, SessionId};
use polylogue_store::{StoreError, writer::ParentLifetime};
use std::sync::mpsc::sync_channel;

#[test]
fn queued_import_checks_its_captured_parent_before_source_admission() {
    for expired in [false, true] {
        let home = ClaudeHome::new();
        let path = home.dir.path().join("queued-conversation.json");
        let bytes = serde_json::to_vec(&serde_json::json!({
            "uuid":"queued-session", "chat_messages":[{
                "uuid":"queued-message", "sender":"human", "text":"queued native body"
            }]
        }))
        .unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let (mut engine, lock) = open_engine(&home);
        let cancellation = polylogue_store::read::limit::ReadCancellation::default();
        let parent = ParentLifetime::capture(expired.then(Instant::now), cancellation.clone());
        if !expired {
            parent.check().unwrap();
        }
        let (tx, rx) = channel();
        let worker_tx = tx.clone();
        let (ready, started) = sync_channel(1);
        let (release, held) = sync_channel(1);
        let status = Arc::new(Mutex::new(LoopStatus::default()));
        let worker = std::thread::spawn(move || {
            let _lock = lock;
            engine.run(
                worker_tx,
                rx,
                &status,
                crate::Lifetime::UntilStopped,
                || {
                    ready.send(()).unwrap();
                    // The existing ready boundary holds the real loop before its
                    // first pass and request dequeue; no provider work has begun.
                    let _ = held.recv();
                },
            )
        });
        started.recv_timeout(Duration::from_secs(5)).unwrap();
        let authorship = Authorship {
            author: Author::Operator,
            via: Channel::Web,
        };
        let (reply, wait) = sync_channel(1);
        tx.send(LoopMsg::Import {
            paths: vec![path.clone()],
            format: None,
            from_archive: None,
            dry_run: false,
            authorship: authorship.clone(),
            parent: Some(parent),
            reply,
        })
        .unwrap();
        // Cancellation occurs after send, while the owner cannot dequeue it.
        if !expired {
            cancellation.cancel();
        }
        release.send(()).unwrap();
        let rejected = wait.recv_timeout(Duration::from_secs(30)).unwrap();
        let before = [
            home.count("SELECT count(*) AS n FROM a.input"),
            home.count("SELECT count(*) AS n FROM a.revision"),
            home.count("SELECT count(*) AS n FROM a.observation"),
            home.count("SELECT count(*) AS n FROM a.blob"),
        ];
        // A refusal must not stop queue servicing or poison later admission.
        let (reply, wait) = sync_channel(1);
        tx.send(LoopMsg::Import {
            paths: vec![path.clone()],
            format: None,
            from_archive: None,
            dry_run: false,
            authorship,
            parent: None,
            reply,
        })
        .unwrap();
        let admitted = wait.recv_timeout(Duration::from_secs(30)).unwrap();
        tx.send(LoopMsg::Stop).unwrap();
        worker.join().unwrap().unwrap();
        match rejected {
            Err(ImportFailure::Parent(StoreError::ReadDeadlineExceeded)) if expired => {}
            Err(ImportFailure::Parent(StoreError::ReadCancelled)) if !expired => {}
            other => panic!("unexpected queued import reply: {other:?}"),
        }
        assert_eq!(before, [0; 4]);
        let report = admitted.unwrap();
        assert!(report.refused.is_empty(), "{report:?}");
        assert!(report.pass.errors.is_empty(), "{report:?}");
        let pool = ReadPool::open(&Archive::at(&home.root)).unwrap();
        let reader = pool.reader().unwrap();
        let messages = reader
            .messages(
                &SessionId::parse("claude-ai:queued-session").unwrap(),
                false,
            )
            .unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text(), "queued native body");
        let input = BlobHash::of(&bytes).to_string();
        let progress = reader.input_progress(&input).unwrap().unwrap();
        assert_eq!(progress.length, bytes.len() as u64);
        let mut actual = Vec::new();
        reader
            .stored_chain(&input, progress.chain, progress.length)
            .unwrap()
            .reader(Arc::new(reader.blob_reader().unwrap()))
            .read_to_end(&mut actual)
            .unwrap();
        assert_eq!(actual, bytes);
        assert_eq!(std::fs::read(path).unwrap(), bytes);
    }
}
