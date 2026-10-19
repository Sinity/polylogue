//! Property laws over generated Claude Code archives (architecture section 9;
//! the "Laws for" Beads: L1 idempotent ingest, L2 append equivalence, L4 order
//! independence, L5 rebuild equivalence, L11 lineage without duplication).
//!
//! The generator writes main session files, some of which resume an earlier
//! session and so begin with a byte-identical copy of its first records.

use crate::tests::{World, assistant, delete_index, export_of, user};
use polylogue_model::export::SessionExport;
use polylogue_store::{Archive, ReadPool};
use proptest::prelude::*;
use std::collections::BTreeSet;
use std::fs;

/// One generated archive: files in creation order, each a list of lines.
#[derive(Debug, Clone)]
struct Corpus {
    files: Vec<(String, Vec<String>)>,
    /// Every distinct record uuid written, copies counted once.
    uuids: BTreeSet<String>,
    /// The resumed sessions.
    resumes: Vec<String>,
}

#[derive(Debug, Clone)]
struct Shape {
    parents: Vec<usize>,
    children: Vec<(usize, usize, usize)>,
}

fn shape() -> impl Strategy<Value = Shape> {
    (
        prop::collection::vec(1usize..6, 1..4),
        prop::collection::vec((0usize..8, 1usize..6, 1usize..4), 0..3),
    )
        .prop_map(|(parents, children)| Shape { parents, children })
}

fn corpus(shape: &Shape) -> Corpus {
    let mut files = Vec::new();
    let mut uuids = BTreeSet::new();
    let mut resumes = Vec::new();
    let line = |i: usize, sid: &str, uuid: &str| {
        if i % 2 == 0 {
            user(uuid, sid, &format!("{sid} says {i}"))
        } else {
            assistant(uuid, sid, &format!("{sid} answers {i}"))
        }
    };
    for (p, len) in shape.parents.iter().enumerate() {
        let sid = format!("p{p}");
        let lines: Vec<String> = (0..*len)
            .map(|i| {
                let uuid = format!("{sid}-{i}");
                uuids.insert(uuid.clone());
                line(i, &sid, &uuid)
            })
            .collect();
        files.push((format!("{sid}.jsonl"), lines));
    }
    for (c, (source, take, own)) in shape.children.iter().enumerate() {
        // A resume of any earlier session, an earlier resume included (a
        // chained resume copies records from several sessions).
        let source_idx = source % files.len();
        let source_lines = files[source_idx].1.clone();
        let k = (*take).min(source_lines.len());
        let sid = format!("c{c}");
        let mut lines: Vec<String> = source_lines[..k].to_vec();
        for i in 0..*own {
            let uuid = format!("{sid}-{i}");
            uuids.insert(uuid.clone());
            lines.push(line(k + i, &sid, &uuid));
        }
        resumes.push(sid.clone());
        files.push((format!("{sid}.jsonl"), lines));
    }
    Corpus {
        files,
        uuids,
        resumes,
    }
}

/// Everything a reader sees for every session, in session order.
fn view(w: &World) -> Vec<SessionExport> {
    let pool = ReadPool::open(&Archive::at(&w.root)).unwrap();
    let ids = pool
        .reader()
        .unwrap()
        .query("SELECT id FROM session ORDER BY id", &[])
        .unwrap();
    ids.iter()
        .map(|row| export_of(&w.root, row["id"].as_str().unwrap()))
        .collect()
}

/// Row count of every archive.db table.
fn archive_row_counts(w: &World) -> Vec<(String, i64)> {
    let pool = ReadPool::open(&Archive::at(&w.root)).unwrap();
    let r = pool.reader().unwrap();
    r.query(
        "SELECT name FROM a.sqlite_master WHERE type = 'table' ORDER BY name",
        &[],
    )
    .unwrap()
    .iter()
    .map(|row| {
        let name = row["name"].as_str().unwrap().to_owned();
        let n = r
            .query(&format!("SELECT COUNT(*) AS n FROM a.\"{name}\""), &[])
            .unwrap()[0]["n"]
            .as_i64()
            .unwrap();
        (name, n)
    })
    .collect()
}

fn write_whole(w: &World, c: &Corpus) {
    for (name, lines) in &c.files {
        w.append(name, &lines.concat());
    }
}

fn converge_all_at_once(c: &Corpus) -> (World, Vec<SessionExport>) {
    let w = World::new();
    write_whole(&w, c);
    let (mut e, lock) = w.engine(64 << 20);
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    drop((e, lock));
    let v = view(&w);
    (w, v)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// L1: converging again changes nothing and reads nothing; the same bytes
    /// at a second location give the same view (L40).
    #[test]
    fn law_1_idempotent_ingest(s in shape()) {
        let c = corpus(&s);
        let w = World::new();
        write_whole(&w, &c);
        let (mut e, _lock) = w.engine(64 << 20);
        prop_assert!(e.converge().unwrap().errors.is_empty());
        let first = view(&w);
        let again = e.converge().unwrap();
        prop_assert_eq!(again.inputs_changed, 0);
        prop_assert_eq!(again.bytes_read, 0);
        prop_assert_eq!(&view(&w), &first);
        let second = w.home.join(".claude/projects/-proj-copy");
        fs::create_dir_all(&second).unwrap();
        for (name, lines) in &c.files {
            fs::write(second.join(name), lines.concat()).unwrap();
        }
        prop_assert!(e.converge().unwrap().errors.is_empty());
        prop_assert_eq!(&view(&w), &first);
    }

    /// L2 (split equals whole): any file cut after any line, converged, then
    /// completed by an append, gives the view of storing it whole.
    #[test]
    fn law_2_append_equivalence(s in shape(), pick in 0usize..8, cut in 0usize..8) {
        let c = corpus(&s);
        let (_, whole) = converge_all_at_once(&c);
        let f = pick % c.files.len();
        let (name, lines) = &c.files[f];
        let k = cut % (lines.len() + 1);
        let w = World::new();
        for (i, (n, ls)) in c.files.iter().enumerate() {
            if i != f {
                w.append(n, &ls.concat());
            }
        }
        w.append(name, &lines[..k].concat());
        let (mut e, _lock) = w.engine(64 << 20);
        prop_assert!(e.converge().unwrap().errors.is_empty());
        w.append(name, &lines[k..].concat());
        let pass = e.converge().unwrap();
        prop_assert!(pass.errors.is_empty());
        prop_assert_eq!(pass.bytes_read, lines[k..].concat().len() as u64);
        prop_assert_eq!(&view(&w), &whole);
    }

    /// L4: files arriving one by one in any order, a convergence after each,
    /// give the view of converging them all at once (children before parents
    /// included).
    #[test]
    fn law_4_order_independence(s in shape(), order in prop::collection::vec(any::<u32>(), 8)) {
        let c = corpus(&s);
        let (_, whole) = converge_all_at_once(&c);
        let mut idx: Vec<usize> = (0..c.files.len()).collect();
        idx.sort_by_key(|i| (order[i % order.len()], *i));
        let w = World::new();
        let (mut e, _lock) = w.engine(64 << 20);
        for i in idx {
            let (name, lines) = &c.files[i];
            w.append(name, &lines.concat());
            prop_assert!(e.converge().unwrap().errors.is_empty());
        }
        prop_assert_eq!(&view(&w), &whole);
    }

    /// L5: deleting index.db and converging again gives the same view, even
    /// with the provider files gone. L81: the rebuild adds no archive.db row.
    #[test]
    fn law_5_rebuild_equivalence(s in shape(), files_gone in any::<bool>()) {
        let c = corpus(&s);
        let (w, before) = converge_all_at_once(&c);
        let archive_rows = archive_row_counts(&w);
        if files_gone {
            for (name, _) in &c.files {
                fs::remove_file(w.file(name)).unwrap();
            }
        }
        delete_index(&w.root);
        let (mut e, lock) = w.engine(64 << 20);
        prop_assert!(e.converge().unwrap().errors.is_empty());
        drop((e, lock));
        prop_assert_eq!(&view(&w), &before);
        // A deleted provider file is observed gone once; the rebuild itself
        // adds nothing.
        let expected: Vec<(String, i64)> = archive_rows
            .into_iter()
            .map(|(table, n)| {
                let gone = if files_gone && table == "observation" { c.files.len() as i64 } else { 0 };
                (table, n + gone)
            })
            .collect();
        prop_assert_eq!(archive_row_counts(&w), expected);
    }

    /// L11: every record is stored once however many files copy it; a resumed
    /// session reads its parent's prefix logically and owns only its tail.
    #[test]
    fn law_11_lineage_without_duplication(s in shape()) {
        let c = corpus(&s);
        let (w, _) = converge_all_at_once(&c);
        prop_assert_eq!(
            w.count("SELECT COUNT(*) AS n FROM message"),
            c.uuids.len() as i64
        );
        let pool = ReadPool::open(&Archive::at(&w.root)).unwrap();
        let r = pool.reader().unwrap();
        for child in &c.resumes {
            let cid = polylogue_model::SessionId::parse(&format!("claude-code:{child}")).unwrap();
            let own = r.messages(&cid, false).unwrap();
            prop_assert!(own.iter().all(|m| m.session == cid), "a copy is owned by its origin");
            // Read logically, a resumed session shows its own file's records
            // in file order, each once, wherever it is stored.
            let (_, lines) = c.files.iter().find(|(n, _)| n == &format!("{child}.jsonl")).unwrap();
            let expected: Vec<String> = lines
                .iter()
                .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["uuid"].as_str().unwrap().to_owned())
                .collect();
            let logical: Vec<String> = r
                .messages(&cid, true)
                .unwrap()
                .iter()
                .map(|m| m.id.native().as_str().to_owned())
                .collect();
            prop_assert_eq!(logical, expected);
        }
    }
}
