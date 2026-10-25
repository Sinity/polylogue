use super::*;
use polylogue_engine::{Config, Engine, ExciseRequest, ExciseScope};
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
        let external_root = w.dir.path().join("external");
        let external = hashed_blobs::Store::open(external_root.clone(), vec![], 0o600).unwrap();
        let local = hashed_blobs::Store::open(w.root.clone(), vec![], 0o600).unwrap();
        {
            let (archive, _lock) = Archive::open_or_create(&w.root).unwrap();
            let mut config: Config = w.config(64 << 20);
            config.blob_roots = vec![external_root];
            let mut e = Engine::new(config, archive).unwrap();
            assert!(e.converge().unwrap().errors.is_empty());
            assert!(local.contains(&victim_hash));
            assert!(local.contains(&kept_hash));
            assert_eq!(external.put(victim.as_bytes(), 1).unwrap(), victim_hash);
            let subject = if point == "excision.after_index_commit" {
                "interrupted-files".to_owned()
            } else {
                victim_hash.hex()
            };
            fault::arm(point, &subject, Action::Fail, 0, 1);
            assert!(
                e.excise(ExciseRequest {
                    id: "interrupted-files".into(),
                    scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
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
            assert!(e.converge().unwrap().errors.is_empty());
            assert!(
                !local.contains(&victim_hash),
                "restart left an uncatalogued removed file in the primary store"
            );
            assert!(
                external.contains(&victim_hash),
                "cleanup must preserve external read roots"
            );
            assert!(local.contains(&kept_hash));
            assert_eq!(
                local.read_range(&kept_hash, 0, kept.len() as u64).unwrap(),
                kept.as_bytes()
            );
        }
    }
}
