use super::*;
use polylogue_model::{BlobHash, SessionId};
use polylogue_store::input_rows::{Extent, InputRows, ObservationRow, Revision};
use polylogue_store::primary::{Confirmed, ConfirmedChange, Material, Product, ReplayControl};
use std::time::{Duration, Instant};

fn until_ready(runtime: &Runtime, reason: &str, mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while !ready() {
        assert!(
            Instant::now() < deadline,
            "{reason}: {:?}",
            runtime.daemon.status.lock().unwrap()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn an_idle_primary_parses_later_confirmed_originals_without_provider_sources() {
    let home = ClaudeHome::new();
    let mut startup = primary(&home);
    startup.engine.reads_sources = false;
    let runtime = launch(startup).unwrap();
    let writer = runtime.daemon.writes.as_ref().unwrap();
    let control = |control| Confirmed {
        position: 0,
        event: String::new(),
        output_index: 0,
        sinex_context: None,
        change: ConfirmedChange::Control(Box::new(control)),
    };
    writer
        .apply_confirmed(vec![control(ReplayControl::Reset {
            scope: "late-input".into(),
        })])
        .unwrap();
    writer
        .apply_confirmed(vec![control(ReplayControl::Completion {
            position: 0,
            delivery: None,
        })])
        .unwrap();
    until_ready(&runtime, "initial primary did not become idle", || {
        let status = runtime.daemon.status.lock().unwrap();
        status.passes > 0 && !status.busy
    });

    // Repeat after the first parse's own commits: consuming their inbox wake
    // while parsing must not hide a later source commit.
    for number in 1..=2 {
        let session_name = format!("late-session-{number}");
        let bytes = user(
            &format!("late-user-{number}"),
            &session_name,
            "Read only from confirmed originals",
        )
        .into_bytes();
        let hash = BlobHash::of(&bytes);
        let length = i64::try_from(bytes.len()).unwrap();
        let store = home.dir.path().join("sinex-store");
        let prepared = hashed_blobs::Preparation::new(store.join("fixture-staging"))
            .unwrap()
            .plain(bytes.as_slice(), 3)
            .unwrap();
        let mut segment = hashed_blobs::SegmentWriter::open(
            &store,
            format!("segments/confirmed-{number}.hb"),
            0,
            &[],
            0o600,
        )
        .unwrap();
        let committed = segment
            .append_group([Ok(prepared)], |_| Ok(None))
            .unwrap()
            .wait()
            .unwrap();
        segment.acknowledge(&committed).unwrap();
        let path = home
            .dir
            .path()
            .join(format!("absent-provider/session-{number}.jsonl"));
        assert!(!path.exists());
        writer
            .apply_confirmed(vec![
                Confirmed {
                    position: number,
                    event: format!("019b0000-0000-7000-8000-{number:012x}"),
                    output_index: 0,
                    sinex_context: None,
                    change: ConfirmedChange::Live(Box::new(Product::Input {
                        rows: Box::new(InputRows {
                            acquisition: None,
                            input: format!("late-input-{number}"),
                            format: "claude-code".into(),
                            revision: Some(Revision {
                                number: 1,
                                produced: "first".into(),
                                chain: 1,
                                length,
                            }),
                            extents: vec![Extent {
                                chain: 1,
                                offset: 0,
                                length,
                                blob: hash,
                                blob_offset: 0,
                            }],
                            observation: Some(ObservationRow {
                                id: format!("late-observation-{number}"),
                                machine: "synthetic-machine".into(),
                                host: "synthetic-host".into(),
                                path: path.to_str().unwrap().into(),
                                path_bytes: None,
                                class: "first".into(),
                                revision: Some(1),
                                chain: None,
                                length: Some(length),
                                device: None,
                                inode: None,
                                size: None,
                                mtime_ns: None,
                                ctime_ns: None,
                                polylogue_version: "synthetic".into(),
                                seen_at: 1_770_000_000_000,
                                details_json: None,
                                container: None,
                            }),
                        }),
                        material: Some(Material {
                            id: format!("019b0000-0000-7000-8000-{:012x}", number + 100),
                            hash,
                            length: bytes.len() as u64,
                            extents: committed
                                .extents
                                .iter()
                                .map(hashed_blobs::Extent::encode)
                                .collect(),
                        }),
                    })),
                },
                control(ReplayControl::Completion {
                    position: number,
                    delivery: None,
                }),
            ])
            .unwrap();
        let session = SessionId::parse(&format!("claude-code:{session_name}")).unwrap();
        until_ready(&runtime, "confirmed original was not parsed", || {
            runtime
                .daemon
                .reads
                .reader()
                .unwrap()
                .messages(&session, false)
                .unwrap()
                .len()
                == 1
        });
        let messages = runtime
            .daemon
            .reads
            .reader()
            .unwrap()
            .messages(&session, false)
            .unwrap();
        assert_eq!(messages[0].text(), "Read only from confirmed originals");
        assert!(
            !path.exists(),
            "reading confirmed originals must not recreate a provider file"
        );
        until_ready(&runtime, "parsed primary did not become idle", || {
            !runtime.daemon.status.lock().unwrap().busy
        });
        let passes = runtime.daemon.status.lock().unwrap().passes;
        std::thread::sleep(Duration::from_millis(200));
        let later = runtime.daemon.status.lock().unwrap();
        assert!(
            later.passes <= passes + 4,
            "own parse commits kept the idle loop running: before={passes}, after={}",
            later.passes
        );
        assert_eq!(later.full_scans, 0);
    }
}
