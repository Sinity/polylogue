use super::*;
use polylogue_engine::{Config, Engine, ExciseRequest, ExciseTarget};
use polylogue_model::SessionId;
use polylogue_store::Archive;
use polylogue_testkit::{ClaudeHome, claude_user};

#[test]
fn reopening_deletes_only_the_local_files_a_committed_excision_marked() {
    for point in ["excision.after_index_commit", "blob.before_delete"] {
        fault::clear();
        let w = ClaudeHome::new();
        let victim = claude_user(
            "v1",
            "victim",
            &format!("removed synthetic text {}", "x".repeat(256 << 10)),
        );
        let kept = claude_user(
            "k1",
            "kept",
            &format!("kept synthetic text {}", "y".repeat(256 << 10)),
        );
        w.append("victim.jsonl", &victim);
        w.append("kept.jsonl", &kept);
        let victim_hash = hashed_blobs::Hash(*blake3::hash(victim.as_bytes()).as_bytes());
        let kept_hash = hashed_blobs::Hash(*blake3::hash(kept.as_bytes()).as_bytes());
        let local = hashed_blobs::Store::open(w.root.clone(), vec![], 0o600).unwrap();
        {
            let (archive, _lock) = Archive::open_or_create(&w.root).unwrap();
            let config: Config = w.config(64 << 20);
            let mut e = Engine::new(config, archive).unwrap();
            assert!(e.run_pass().unwrap().errors.is_empty());
            assert!(local.contains(&victim_hash));
            assert!(local.contains(&kept_hash));
            let subject = if point == "excision.after_index_commit" {
                "interrupted-files".to_owned()
            } else {
                victim_hash.hex()
            };
            fault::arm(point, &subject, Action::Fail, 0, 1);
            assert!(
                e.excise(ExciseRequest {
                    id: "interrupted-files".into(),
                    target: ExciseTarget::Session(SessionId::parse("claude-code:victim").unwrap()),
                    reason: None,
                    dry_run: false,
                })
                .is_err()
            );
            assert!(local.contains(&victim_hash));
            assert!(matches!(
                e.writer().unwrap().derive(),
                Err(polylogue_store::StoreError::WriterGone)
            ));
        }
        fault::clear();
        drop(Archive::open_or_create(&w.root).unwrap());
        assert!(!local.contains(&victim_hash));
        for _ in 0..2 {
            let (mut e, _lock) = w.engine(64 << 20);
            assert!(e.run_pass().unwrap().errors.is_empty());
            assert!(
                !local.contains(&victim_hash),
                "restart left an uncatalogued removed file in the primary store"
            );
            assert!(local.contains(&kept_hash));
            assert_eq!(
                local.read_range(&kept_hash, 0, kept.len() as u64).unwrap(),
                kept.as_bytes()
            );
        }
    }
}
