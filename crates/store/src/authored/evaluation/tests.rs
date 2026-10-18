use super::*;
use polylogue_model::authored::GraphNode;
use std::collections::BTreeMap;
fn id(n: u32) -> String {
    format!("01900000-0000-7000-8000-{n:012}")
}
#[test]
fn disk_evaluation_matches_in_memory_model() {
    for seed in 0..32u32 {
        let dir = tempfile::tempdir().unwrap();
        let writer = Writer::open(dir.path()).unwrap();
        let path = dir.path().join("archive.db");
        let mut graph: BTreeMap<String, GraphNode> = BTreeMap::new();
        let mut random = seed + 1;
        writer.archive.execute_batch("BEGIN IMMEDIATE").unwrap();
        let mut family = vec![];
        for n in 1..=12 {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            let withdrawal = n > 1 && random % 3 == 0;
            let root = if withdrawal { id(n) } else { id(1) };
            let parent = if withdrawal {
                Some(("withdraws", 1 + random % (n - 1)))
            } else if n > 1 {
                Some(("revises", family[random as usize % family.len()]))
            } else {
                None
            };
            writer.archive.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,local_seq) VALUES(?1,?2,?3,'payload is irrelevant to topology','agent','machine',?4,?5)",params![id(n),root,if withdrawal {"withdrawal"}else{"note"},1000-n,n]).unwrap();
            let mut relations = vec![];
            if let Some((kind, parent)) = parent {
                let other_root = &graph[&id(parent)].root;
                writer
                    .archive
                    .execute(
                        "INSERT INTO note_relation(note_id,kind,other) VALUES(?1,?2,?3)",
                        params![id(n), kind, format!("note:{other_root}@{}", id(parent))],
                    )
                    .unwrap();
                relations.push(Relation {
                    kind: kind.into(),
                    other: id(parent),
                });
            }
            graph.insert(
                id(n),
                GraphNode {
                    root: root.clone(),
                    relations,
                },
            );
            if !withdrawal {
                family.push(n);
            }
        }
        writer.archive.execute_batch("COMMIT").unwrap();
        for root in graph
            .values()
            .map(|n| n.root.clone())
            .collect::<std::collections::BTreeSet<_>>()
        {
            let mut cancel = || false;
            let mut actual = Evaluation::new(
                &writer.archive,
                &path,
                &root,
                12,
                default_until(),
                &mut cancel,
            )
            .unwrap();
            assert_eq!(
                actual.heads(&root).unwrap(),
                authored::current(&root, &graph).unwrap(),
                "seed {seed} root {root}"
            );
            for (node, n) in &graph {
                // Unrelated families need not be in the scratch graph.
                if actual
                    .scratch
                    .query_row(
                        "SELECT EXISTS(SELECT 1 FROM node WHERE id=?1)",
                        [node],
                        |r| r.get::<_, bool>(0),
                    )
                    .unwrap()
                {
                    assert_eq!(
                        actual.withdrawn(node).unwrap(),
                        authored::withdrawn(node, &graph).unwrap(),
                        "seed {seed} node {node}"
                    );
                    assert_eq!(
                        actual.is_current(node, &n.root).unwrap(),
                        authored::current(&n.root, &graph).unwrap().contains(node)
                    );
                }
            }
        }
    }
}
#[test]
fn large_graph_spills_to_disk_and_scratch_is_removed() {
    let dir = tempfile::tempdir().unwrap();
    let writer = Writer::open(dir.path()).unwrap();
    let path = dir.path().join("archive.db");
    let root = id(1);
    writer.archive.execute_batch("BEGIN IMMEDIATE").unwrap();
    for n in 1..=4000 {
        writer.archive.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,local_seq) VALUES(?1,?2,'note','body','agent','machine',0,?3)",params![id(n),root,n]).unwrap();
        if n > 1 {
            writer
                .archive
                .execute(
                    "INSERT INTO note_relation(note_id,kind,other) VALUES(?1,'revises',?2)",
                    params![id(n), format!("note:{root}@{}", id(n - 1))],
                )
                .unwrap();
        }
    }
    writer.archive.execute_batch("COMMIT").unwrap();
    for count in [1000u64, 4000] {
        let mut cancel = || false;
        let mut eval = Evaluation::new(
            &writer.archive,
            &path,
            &root,
            count,
            default_until(),
            &mut cancel,
        )
        .unwrap();
        assert_eq!(eval.heads(&root).unwrap(), [id(count as u32)]);
        assert_eq!(
            eval.scratch
                .query_row("PRAGMA cache_size", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            -64
        );
        let scratch = eval._file.path().to_owned();
        let bytes = std::fs::metadata(&scratch).unwrap().len();
        let cache = crate::native::cache_bytes(&eval.scratch).unwrap();
        let peak = eval.cache_peak.max(cache);
        assert!(bytes > 65536 && bytes <= SCRATCH_PAGES * 4096, "{bytes}");
        assert!(
            cache > 0 && peak <= 128 * 1024 && (peak as u64) < bytes,
            "nodes={count} peak={peak} cache={cache} file={bytes}"
        );
        eprintln!(
            "authored scratch: nodes={count}, file_bytes={bytes}, cache_peak_bytes={peak}, cache_bytes={cache}, configured_cache_kib=64, disk_limit_bytes={}",
            SCRATCH_PAGES * 4096
        );
        assert_eq!(scratch.parent(), Some(std::env::temp_dir().as_path()));
        eval.steps = STEPS;
        assert!(matches!(
            eval.checkpoint(),
            Err(Error::Authored(Failure {
                kind: ErrorKind::NotYetAvailable,
                ..
            }))
        ));
        eval.steps = 0;
        eval.until = Instant::now();
        assert!(matches!(eval.checkpoint(), Err(Error::Cancelled)));
        drop(eval);
        assert!(!scratch.exists());
    }
    // Source queries that walk large populations must be index seeks with no
    // full scan or temporary sort.
    for sql in [
        "SELECT rowid,id FROM note INDEXED BY note_root WHERE root='r' AND rowid>0 ORDER BY rowid LIMIT 1",
        "SELECT r.rowid,r.note_id,n.local_seq FROM note_relation r INDEXED BY note_relation_other JOIN note n ON n.id=r.note_id WHERE r.other='r' AND r.kind='withdraws' AND r.rowid>0 ORDER BY r.rowid LIMIT 1",
        "SELECT t.rowid,n.root,n.local_seq FROM note_target t INDEXED BY note_target_ref JOIN note n ON n.id=t.note_id WHERE t.ref='t' AND t.rowid>0 ORDER BY t.rowid LIMIT 1",
    ] {
        let plan = writer
            .archive
            .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
            .unwrap()
            .query_map([], |r| r.get::<_, String>(3))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert!(
            plan.iter()
                .all(|p| !p.contains("SCAN") && !p.contains("TEMP B-TREE")),
            "{sql}: {plan:?}"
        );
    }
}
