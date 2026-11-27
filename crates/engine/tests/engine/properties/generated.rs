//! Properties of generated Claude Code archives: ingesting again changes
//! nothing, appends equal whole files, arrival order does not matter, a
//! rebuild gives the same view, and a copied record is stored once.
//!
//! The generator writes main session files, some of which resume an earlier
//! session and so begin with a byte-identical copy of its first records.

use polylogue_model::{SessionId, bundle::Record};
use polylogue_store::{Archive, Reader};
use polylogue_testkit::{
    ClaudeHome, claude_assistant as assistant, claude_user as user, delete_index,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;
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
        if i.is_multiple_of(2) {
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

/// Built or kept current, the index names only rows it holds.
fn assert_references_hold(reader: &Reader<'_>) {
    assert_eq!(
        reader
            .query(
                "SELECT \"table\", rowid, parent FROM pragma_foreign_key_check",
                &[]
            )
            .unwrap(),
        Vec::<serde_json::Value>::new()
    );
}

/// Everything a reader sees for every session, in session order.
fn view(w: &ClaudeHome) -> Vec<Vec<Record>> {
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    assert_references_hold(&reader);
    let ids = reader
        .query("SELECT id FROM session ORDER BY id", &[])
        .unwrap();
    ids.iter()
        .map(|row| {
            let id = SessionId::parse(row["id"].as_str().unwrap()).unwrap();
            polylogue_testkit::session_bundle(&reader, &id).unwrap()
        })
        .collect()
}

/// What the search indexes hold, by content rather than row id: for each
/// index the hashes of the texts in it, and each term's document and
/// occurrence counts over every full-text file. A first build fills the
/// files while it composes and a live pass through the search queue; both
/// must give the same index.
fn search_index(w: &ClaudeHome) -> Vec<serde_json::Value> {
    let archive = Archive::at(&w.root);
    let conn = polylogue_store::archive::open_read(&archive.index_db()).unwrap();
    let files = conn
        .prepare("SELECT name FROM pragma_database_list WHERE name LIKE 'search_%' ORDER BY name")
        .unwrap()
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    assert!(!files.is_empty());
    let mut out = Vec::new();
    for fts in ["text_fts", "output_fts"] {
        let mut keys = Vec::new();
        let mut vocabulary = Vec::new();
        for file in &files {
            keys.push(format!("SELECT rowid FROM {file}.{fts} WHERE rowid > 0"));
            let vocab = format!("vocab_{file}_{fts}");
            conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE temp.{vocab} USING fts5vocab({file}, {fts}, 'row')"
            ))
            .unwrap();
            vocabulary.push(format!("SELECT term, doc, cnt FROM temp.{vocab}"));
        }
        let mut rows = |sql: String| {
            let mut statement = conn.prepare(&sql).unwrap();
            let names: Vec<String> = statement
                .column_names()
                .iter()
                .map(|name| (*name).to_owned())
                .collect();
            let found = statement
                .query_map([], |row| {
                    let mut value = serde_json::Map::new();
                    for (n, name) in names.iter().enumerate() {
                        let cell = match row.get_ref(n)? {
                            rusqlite::types::ValueRef::Integer(i) => serde_json::json!(i),
                            rusqlite::types::ValueRef::Text(t) => {
                                serde_json::json!(String::from_utf8_lossy(t))
                            }
                            other => serde_json::json!(format!("{other:?}")),
                        };
                        value.insert(name.clone(), cell);
                    }
                    Ok(serde_json::Value::Object(value))
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            out.extend(found);
        };
        rows(format!(
            "SELECT '{fts}' AS idx, hex(t.hash) AS h FROM text t
             WHERE t.id IN ({}) ORDER BY h",
            keys.join(" UNION ALL ")
        ));
        rows(format!(
            "SELECT '{fts} terms' AS idx, term, sum(doc) AS doc, sum(cnt) AS cnt
             FROM ({}) GROUP BY term ORDER BY term",
            vocabulary.join(" UNION ALL ")
        ));
    }
    out
}

/// Row count of every archive.db table.
fn archive_row_counts(w: &ClaudeHome) -> Vec<(String, i64)> {
    let pool = w.reads();
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

fn write_whole(w: &ClaudeHome, c: &Corpus) {
    for (name, lines) in &c.files {
        w.append(name, &lines.concat());
    }
}

fn run_all_at_once(c: &Corpus) -> ClaudeHome {
    let w = ClaudeHome::new();
    write_whole(&w, c);
    let (mut e, lock) = w.engine(64 << 20);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    drop((e, lock));
    w
}

#[test]
fn two_resumes_of_one_parent_store_each_record_once() {
    let c = corpus(&Shape {
        parents: vec![5],
        children: vec![(7, 4, 3), (2, 3, 3)],
    });
    let w = run_all_at_once(&c);
    w.read(|reader| {
        assert_references_hold(reader);
        assert_eq!(
            reader.query("SELECT count(*) AS n FROM message", &[])?[0]["n"],
            11
        );
        Ok(())
    });
}

proptest! {
    // Seeds of past failures are kept beside this file; cargo runs
    // integration tests from the package directory.
    #![proptest_config(ProptestConfig {
        cases: 16,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(
            "tests/engine/properties/regressions.txt",
        ))),
        ..ProptestConfig::default()
    })]

    /// Another pass changes nothing and reads nothing; the same bytes
    /// at a second location give the same view.
    #[test]
    fn idempotent_ingest(s in shape()) {
        let c = corpus(&s);
        let w = ClaudeHome::new();
        write_whole(&w, &c);
        let (mut e, _lock) = w.engine(64 << 20);
        prop_assert!(e.run_pass().unwrap().errors.is_empty());
        let first = view(&w);
        let again = e.run_pass().unwrap();
        prop_assert_eq!(again.inputs_changed, 0);
        prop_assert_eq!(again.bytes_read, 0);
        prop_assert_eq!(&view(&w), &first);
        let second = w.home.join(".claude/projects/-proj-copy");
        fs::create_dir_all(&second).unwrap();
        for (name, lines) in &c.files {
            fs::write(second.join(name), lines.concat()).unwrap();
        }
        prop_assert!(e.run_pass().unwrap().errors.is_empty());
        prop_assert_eq!(&view(&w), &first);
    }

    /// Split equals whole: any file cut after any line, run, then
    /// completed by an append, gives the view of storing it whole.
    #[test]
    fn append_equivalence(s in shape(), pick in 0usize..8, cut in 0usize..8) {
        let c = corpus(&s);
        let whole_archive = run_all_at_once(&c);
        let whole = view(&whole_archive);
        let whole_search = search_index(&whole_archive);
        prop_assert!(whole_search.iter().any(|v| v["idx"] == "text_fts"));
        let f = pick % c.files.len();
        let (name, lines) = &c.files[f];
        let k = cut % (lines.len() + 1);
        let w = ClaudeHome::new();
        for (i, (n, ls)) in c.files.iter().enumerate() {
            if i != f {
                w.append(n, &ls.concat());
            }
        }
        w.append(name, &lines[..k].concat());
        let (mut e, _lock) = w.engine(64 << 20);
        prop_assert!(e.run_pass().unwrap().errors.is_empty());
        w.append(name, &lines[k..].concat());
        let pass = e.run_pass().unwrap();
        prop_assert!(pass.errors.is_empty());
        prop_assert_eq!(pass.bytes_read, lines[k..].concat().len() as u64);
        prop_assert_eq!(&view(&w), &whole);
        drop((e, _lock));
        prop_assert_eq!(search_index(&w), whole_search);
    }

    /// Files arriving one by one in any order, a pass after each,
    /// give the view of running a pass them all at once (children before parents
    /// included).
    #[test]
    fn order_independence(s in shape(), order in prop::collection::vec(any::<u32>(), 8)) {
        let c = corpus(&s);
        let whole = view(&run_all_at_once(&c));
        let mut idx: Vec<usize> = (0..c.files.len()).collect();
        idx.sort_by_key(|i| (order[i % order.len()], *i));
        let w = ClaudeHome::new();
        let (mut e, _lock) = w.engine(64 << 20);
        for i in idx {
            let (name, lines) = &c.files[i];
            w.append(name, &lines.concat());
            prop_assert!(e.run_pass().unwrap().errors.is_empty());
        }
        prop_assert_eq!(&view(&w), &whole);
    }

    /// Deleting index.db and running a pass again gives the same view, even
    /// with the provider files gone, and the rebuild adds no archive.db row.
    #[test]
    fn rebuild_equivalence(s in shape(), files_gone in any::<bool>()) {
        let c = corpus(&s);
        let w = run_all_at_once(&c);
        let before = view(&w);
        let archive_rows = archive_row_counts(&w);
        if files_gone {
            for (name, _) in &c.files {
                fs::remove_file(w.file(name)).unwrap();
            }
        }
        delete_index(&w.root);
        let (mut e, lock) = w.engine(64 << 20);
        prop_assert!(e.run_pass().unwrap().errors.is_empty());
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

    /// Every record is stored once however many files copy it; a resumed
    /// session reads its parent's prefix logically and owns only its tail.
    #[test]
    fn parents_without_duplication(s in shape()) {
        let c = corpus(&s);
        let w = run_all_at_once(&c);
        let pool = w.reads();
        let r = pool.reader().unwrap();
        assert_references_hold(&r);
        prop_assert_eq!(
            r.query("SELECT count(*) AS n FROM message", &[]).unwrap()[0]["n"].as_i64().unwrap(),
            c.uuids.len() as i64
        );
        for child in &c.resumes {
            let cid = SessionId::parse(&format!("claude-code:{child}")).unwrap();
            let own = r.messages(&cid, false).unwrap();
            prop_assert!(own.iter().all(|m| m.session == cid), "a copy is owned by its origin");
            let (_, lines) = c.files.iter().find(|(n, _)| n == &format!("{child}.jsonl")).unwrap();
            // The generator names a child's own records `<child>-<i>`; the
            // copied prefix keeps its parent's names.
            let tail: Vec<String> = lines
                .iter()
                .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["uuid"].as_str().unwrap().to_owned())
                .filter(|uuid| uuid.starts_with(&format!("{child}-")))
                .collect();
            let own_ids: Vec<String> = own.iter().map(|m| m.id.native().as_str().to_owned()).collect();
            prop_assert_eq!(own_ids, tail);
            // Read logically, a resumed session shows its own file's records
            // in file order, each once, wherever it is stored.
            let expected: Vec<String> = lines
                .iter()
                .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["uuid"].as_str().unwrap().to_owned())
                .collect();
            let messages: Vec<String> = r
                .messages(&cid, true)
                .unwrap()
                .iter()
                .map(|m| m.id.native().as_str().to_owned())
                .collect();
            prop_assert_eq!(messages, expected);
        }
    }
}
