use super::*;

fn stored_home(behind: bool) -> ClaudeHome {
    let home = ClaudeHome::new();
    let path = home.file("idle.jsonl");
    let first = claude_user("one", "idle", "first body");
    let second = claude_user("one", "idle", "other body");
    {
        let (mut engine, _lock) = open_engine(&home);
        for bytes in [&first, &second, &first] {
            std::fs::write(&path, bytes).unwrap();
            let pass = engine.run_pass().unwrap();
            assert!(pass.errors.is_empty(), "{pass:?}");
        }
    }
    if behind {
        let (archive, _lock) = Archive::open_or_create(&home.root).unwrap();
        assert_eq!(
            polylogue_store::fixture::older_reading(&archive, "claude-code")
                .unwrap()
                .0,
            3
        );
    }
    home
}

fn idle_bytes(engine: &mut Engine) -> [u64; 2] {
    std::array::from_fn(|_| {
        let before = engine.blobs.bytes_read();
        let (pass, _) = engine.one_pass(false, &Default::default()).unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert!(!pass.capture_pending, "{pass:?}");
        engine.blobs.bytes_read() - before
    })
}

#[test]
fn idle_passes_read_no_segment_bytes_after_current_or_older_readings() {
    for sources in [true, false] {
        for behind in [false, true] {
            let home = stored_home(behind);
            {
                let (mut engine, _lock) = open_engine(&home);
                engine.config.reads_sources = sources;
                let before = engine.blobs.bytes_read();
                let pass = engine.run_pass().unwrap();
                assert!(pass.errors.is_empty(), "{pass:?}");
                let replay = engine.blobs.bytes_read() - before;
                if behind {
                    assert!(replay > 0, "older readings must read their stored bytes");
                } else {
                    assert_eq!(replay, 0);
                }
                let bytes = idle_bytes(&mut engine);
                eprintln!("sources={sources} behind={behind}: replay={replay}, idle={bytes:?}");
                assert_eq!(bytes, [0, 0]);
                assert_eq!(home.texts("idle"), ["first body"]);
                assert!(
                    engine
                        .reads
                        .reader()
                        .unwrap()
                        .chains_behind_index(
                            &[(
                                "claude-code".into(),
                                polylogue_sources::origin::by_name("claude-code")
                                    .unwrap()
                                    .reading_version()
                            )],
                            polylogue_store::StoppedChains::Finished,
                            RevisionChains::All,
                            None,
                        )
                        .unwrap()
                        .is_empty()
                );
            }
            // Saved parse progress also survives opening another engine.
            let (mut engine, _lock) = open_engine(&home);
            engine.config.reads_sources = sources;
            let pass = engine.run_pass().unwrap();
            assert!(pass.errors.is_empty(), "{pass:?}");
            assert_eq!(engine.blobs.bytes_read(), 0);
            assert_eq!(idle_bytes(&mut engine), [0, 0]);
        }
    }
}

#[cfg(feature = "faults")]
#[test]
fn idle_passes_need_no_chain_planning_query() {
    use polylogue_store::fault::{self, Action};

    for sources in [true, false] {
        for behind in [false, true] {
            let home = stored_home(behind);
            let (mut engine, _lock) = open_engine(&home);
            engine.config.reads_sources = sources;
            assert!(engine.run_pass().unwrap().errors.is_empty());
            fault::arm("read.chains_behind_index", "", Action::Fail, 0, 1);
            let first = engine.one_pass(false, &Default::default());
            let second = engine.one_pass(false, &Default::default());
            fault::clear();
            assert!(
                first.is_ok(),
                "sources={sources} behind={behind}: {first:?}"
            );
            assert!(
                second.is_ok(),
                "sources={sources} behind={behind}: {second:?}"
            );
        }
    }
}
