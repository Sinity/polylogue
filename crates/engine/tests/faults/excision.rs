use super::*;
use polylogue_engine::{ExciseRequest, ExciseScope};
use polylogue_model::SessionId;
use polylogue_store::StoreError;
use polylogue_testkit::{ClaudeHome, claude_user};

#[test]
fn a_partially_committed_excision_stops_writes_until_restart() {
    for point in ["excision.between_commits", "excision.after_index_commit"] {
        fault::clear();
        let w = ClaudeHome::new();
        w.append(
            "victim.jsonl",
            &claude_user("v1", "victim", "synthetic removed message"),
        );
        w.append(
            "kept.jsonl",
            &claude_user("k1", "kept", "untouched exact message"),
        );
        let expected;
        {
            let (mut e, _lock) = w.engine(64 << 20);
            assert!(e.converge().unwrap().errors.is_empty());
            expected = w.export("claude-code:kept");
            fault::arm(point, "interrupted", Action::Fail, 0, 1);
            let error = e
                .excise(ExciseRequest {
                    id: "interrupted".into(),
                    scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
                    reason: None,
                    dry_run: false,
                })
                .unwrap_err();
            assert!(error.to_string().contains("injected fault"), "{error}");
            assert!(matches!(
                error,
                polylogue_engine::EngineError::Store(StoreError::ExcisionInterrupted(_))
            ));
            let continued = e
                .writer()
                .and_then(|sender| sender.derive().map_err(Into::into));
            assert!(
                matches!(
                    continued,
                    Err(polylogue_engine::EngineError::Store(StoreError::WriterGone))
                ),
                "{point}: a failed derived commit must not let later writes acknowledge incomplete removal"
            );
            let later = e.excise(ExciseRequest {
                id: "must-not-run".into(),
                scope: ExciseScope::Session(SessionId::parse("claude-code:kept").unwrap()),
                reason: None,
                dry_run: false,
            });
            assert!(later.is_err());
            let pool = w.reads();
            if point == "excision.between_commits" {
                assert!(matches!(
                    pool.reader().unwrap().status(),
                    Err(StoreError::ExcisionPending)
                ));
            } else {
                let r = pool.reader().unwrap();
                assert!(
                    r.messages(&SessionId::parse("claude-code:victim").unwrap(), false)
                        .unwrap()
                        .is_empty()
                );
            }
        }
        fault::clear();
        let (mut e, _lock) = w.engine(64 << 20);
        assert!(e.converge().unwrap().errors.is_empty());
        assert_eq!(w.export("claude-code:kept"), expected);
        assert!(
            w.reads()
                .reader()
                .unwrap()
                .messages(&SessionId::parse("claude-code:victim").unwrap(), false)
                .unwrap()
                .is_empty()
        );
        assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 1);
    }
}
