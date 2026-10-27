//! Properties of the Morrow example archive, built through the real loop and
//! import (`polylogue-testkit`).

use polylogue_engine::Config;
use polylogue_store::{Archive, SqlValue};
use polylogue_testkit::{Morrow, Set, session};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

fn jsonl_files(w: &Morrow) -> Vec<PathBuf> {
    w.provider_files()
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl" || e == "ndjson"))
        .collect()
}

fn count(w: &Morrow, sql: &str) -> i64 {
    let pool = w.reads();
    pool.reader().unwrap().query(sql, &[]).unwrap()[0]["n"]
        .as_i64()
        .unwrap()
}

/// raw bytes survive the source. Deleting any provider file and
/// running a pass leaves the view unchanged, and the file's bytes still read
/// back exactly from the store.
#[test]
fn raw_bytes_survive_the_source() {
    let probe = Morrow::build(Set::Main);
    let home = probe.home.clone();
    // Every provider file stored as an input of its own; the rest are side
    // files read with another (a subagent's .meta.json) or databases kept
    // as row dumps.
    let (inputs, side): (Vec<PathBuf>, Vec<PathBuf>) = probe
        .provider_files()
        .into_iter()
        .partition(|f| probe.stored_bytes(f).is_ok());
    assert!(
        side.iter().all(|f| {
            let n = f.to_string_lossy();
            n.ends_with(".meta.json") || n.ends_with(".sqlite")
        }),
        "provider files that were never stored: {side:?}"
    );
    for rel in inputs.iter().map(|f| f.strip_prefix(&home).unwrap()) {
        let w = Morrow::build(Set::Main);
        let file = w.home.join(rel);
        let bytes = w.stored_bytes(&file).unwrap();
        if !polylogue_sources::ORIGINS.iter().any(|origin| {
            origin.capture == polylogue_sources::Capture::RowDump
                && (origin.open)(&file).is_some()
        }) {
            assert_eq!(bytes, fs::read(&file).unwrap(), "{}", file.display());
        }
        let before = w.view();
        fs::remove_file(&file).unwrap();
        w.run_pass();
        assert_eq!(
            w.view(),
            before,
            "deleting {} changed the view",
            rel.display()
        );
        assert_eq!(
            w.stored_bytes(&file).unwrap(),
            bytes,
            "{} does not read back from the store",
            rel.display()
        );
    }
}

/// identity stability. An unrelated new session, an append to
/// another session's file, and a renamed session file leave every recorded
/// message with the same ID and document (a session may gain messages).
#[test]
fn ids_and_documents_survive_unrelated_changes() {
    let w = Morrow::build(Set::Main);
    let before = w.view();
    let project = w.file(".claude/projects/-home-op-morrow");
    // An unrelated session.
    fs::write(
        project.join("7f3a2c10-0901-4000-8000-000000000901.jsonl"),
        r#"{"type":"user","uuid":"u901","sessionId":"7f3a2c10-0901-4000-8000-000000000901","timestamp":"2026-10-08T08:00:00Z","promptSource":"typed","message":{"role":"user","content":"unrelated"}}
"#,
    )
    .unwrap();
    // An append to #1.
    let one = project.join("7f3a2c10-0001-4000-8000-000000000001.jsonl");
    let mut text = fs::read_to_string(&one).unwrap();
    text.push_str(r#"{"type":"user","uuid":"u1-late","parentUuid":null,"sessionId":"7f3a2c10-0001-4000-8000-000000000001","timestamp":"2026-10-08T09:00:00Z","promptSource":"typed","message":{"role":"user","content":"one more thing"}}
"#);
    fs::write(&one, text).unwrap();
    w.run_pass();
    // #2's file moved to another name in the same project directory.
    let two = project.join("7f3a2c10-0002-4000-8000-000000000002.jsonl");
    fs::rename(&two, project.join("renamed.jsonl")).unwrap();
    w.run_pass();

    let after = w.view();
    let mut wrong = Vec::new();
    for (id, doc) in &before {
        let Some(now) = after.get(id) else {
            wrong.push(format!("{id} is gone"));
            continue;
        };
        let msgs = |d: &serde_json::Value| -> BTreeMap<String, serde_json::Value> {
            d["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| (m["id"].as_str().unwrap().to_owned(), m.clone()))
                .collect()
        };
        let (was, is) = (msgs(doc), msgs(now));
        for (mid, m) in &was {
            match is.get(mid) {
                None => wrong.push(format!("{id}: message {mid} is gone")),
                Some(n) if n != m => wrong.push(format!("{id}: message {mid} changed")),
                _ => {}
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// corruption is loud. A stored blob with one byte flipped fails
/// the read, naming the blob; nothing is returned silently.
#[test]
fn a_corrupt_blob_fails_its_read() {
    let w = Morrow::build(Set::Main);
    let file =
        w.file(".claude/projects/-home-op-morrow/7f3a2c10-0002-4000-8000-000000000002.jsonl");
    let bytes = fs::read(&file).unwrap();
    assert_eq!(w.stored_bytes(&file).unwrap(), bytes);
    let archive = rusqlite::Connection::open(Archive::at(&w.root).archive_db()).unwrap();
    let blobs: Vec<(Vec<u8>, String)> = archive
        .prepare(
            "SELECT DISTINCT b.hash, b.place FROM extent e JOIN blob b ON b.hash = e.blob
             JOIN observation o ON o.input = e.input WHERE o.path = ?1",
        )
        .unwrap()
        .query_map([file.to_string_lossy()], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(!blobs.is_empty());
    let (hash, place) = &blobs[0];
    if place == "inline" {
        // Flip one byte of the stored (compressed) data.
        let mut data: Vec<u8> = archive
            .query_row("SELECT data FROM blob WHERE hash = ?1", [hash], |r| {
                r.get(0)
            })
            .unwrap();
        let last = data.len() - 1;
        data[last] ^= 0x01;
        archive
            .execute(
                "UPDATE blob SET data = ?1 WHERE hash = ?2",
                rusqlite::params![data, hash],
            )
            .unwrap();
    } else {
        let path = find_blob_file(&w.root, hash).expect("the blob's file");
        let mut data = fs::read(&path).unwrap();
        let mid = data.len() / 2;
        data[mid] ^= 0x01;
        fs::write(&path, data).unwrap();
    }
    drop(archive);
    let err = w
        .stored_bytes(&file)
        .expect_err("a corrupt blob must not read back");
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    assert!(
        err.contains(&hex[..12]) || err.to_lowercase().contains("corrupt") || err.contains("blob"),
        "the error names neither the blob nor corruption: {err}"
    );
}

fn find_blob_file(root: &Path, hash: &[u8]) -> Option<PathBuf> {
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in fs::read_dir(&dir).ok()? {
            let p = e.ok()?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name()?.to_string_lossy().contains(&hex[..16]) {
                return Some(p);
            }
        }
    }
    None
}

/// one subagent parent. No session has two incoming subagent
/// links, also when a subagent's file is symlinked into a second session's
/// `subagents/` folder.
#[test]
fn a_subagent_has_one_parent() {
    let w = Morrow::lay_out(Set::Main);
    let project = w.file(".claude/projects/-home-op-morrow");
    let sub = project
        .join("7f3a2c10-0002-4000-8000-000000000002/subagents/agent-a7f3a2c1000000003.jsonl");
    let other = project.join("7f3a2c10-0001-4000-8000-000000000001/subagents");
    fs::create_dir_all(&other).unwrap();
    std::os::unix::fs::symlink(&sub, other.join("agent-a7f3a2c1000000003.jsonl")).unwrap();
    w.run_pass();
    let twice = count(
        &w,
        "SELECT count(*) AS n FROM (SELECT child_id FROM link WHERE kind = 'subagent'
             GROUP BY child_id HAVING count(DISTINCT parent_id) > 1)",
    );
    assert_eq!(twice, 0, "a session has two subagent parents");
    let parent = w
        .reads()
        .reader()
        .unwrap()
        .query(
            "SELECT parent_id FROM link WHERE kind = 'subagent' AND child_id = ?1",
            &[SqlValue::Text(session(3).to_string())],
        )
        .unwrap();
    assert_eq!(parent.len(), 1, "{parent:?}");
    assert_eq!(parent[0]["parent_id"], session(2).to_string());
}

/// compaction keeps the tree connected. Every message of #4 (its
/// parent messages and its own, across the compaction) has a parent chain
/// that reaches a root inside the session's read with its parents.
#[test]
fn compaction_keeps_the_tree_connected() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let msgs = pool.reader().unwrap().messages(&session(4), true).unwrap();
    let ids: BTreeSet<String> = msgs.iter().map(|m| m.id.to_string()).collect();
    let parent: BTreeMap<String, Option<String>> = msgs
        .iter()
        .map(|m| (m.id.to_string(), m.parent.as_ref().map(ToString::to_string)))
        .collect();
    let roots = msgs.iter().filter(|m| m.parent.is_none()).count();
    let mut broken = Vec::new();
    for m in &msgs {
        let mut at = m.id.to_string();
        let mut steps = 0;
        while let Some(Some(p)) = parent.get(&at) {
            if !ids.contains(p) {
                broken.push(format!("{} → {p}: parent not in the session", m.id));
                break;
            }
            at = p.clone();
            steps += 1;
            assert!(steps <= msgs.len(), "a parent cycle through {}", m.id);
        }
    }
    assert_eq!(roots, 1, "#4 should have one root, has {roots}");
    assert!(broken.is_empty(), "{}", broken.join("\n"));
}

/// every Morrow JSONL file stored in many appended steps reads
/// back identical to the file, and the view equals storing each whole.
#[test]
fn appended_steps_reassemble_exactly() {
    let whole = Morrow::build(Set::Main);
    let new_blobs = Morrow::lay_out(Set::Main);
    let files = jsonl_files(&new_blobs);
    let full: Vec<Vec<u8>> = files.iter().map(|f| fs::read(f).unwrap()).collect();
    // Grow every file in three steps, cut at line ends, running a pass after each.
    for f in &files {
        fs::write(f, b"").unwrap();
    }
    for step in 1..=3 {
        for (f, bytes) in files.iter().zip(&full) {
            let lines: Vec<&[u8]> = bytes.split_inclusive(|b| *b == b'\n').collect();
            let upto = lines.len() * step / 3;
            fs::write(f, lines[..upto].concat()).unwrap();
        }
        new_blobs.run_pass();
    }
    for (f, bytes) in files.iter().zip(&full) {
        assert_eq!(
            &new_blobs.stored_bytes(f).unwrap(),
            bytes,
            "{}",
            f.display()
        );
    }
    assert_eq!(new_blobs.view(), whole.view());
}

/// deleting the index and running a pass builds one index
/// again, and idle passes after it change nothing.
#[test]
fn a_rebuild_leaves_one_index() {
    let w = Morrow::build(Set::Main);
    let before = w.view();
    let index = Archive::at(&w.root).index_db();
    fs::remove_file(&index).unwrap();
    w.run_pass();
    let counts = |w: &Morrow| -> Vec<i64> {
        ["session", "message", "block", "tool_call", "link", "event"]
            .iter()
            .map(|t| count(w, &format!("SELECT count(*) AS n FROM {t}")))
            .collect()
    };
    let after_rebuild = counts(&w);
    w.run_pass();
    w.run_pass();
    assert_eq!(counts(&w), after_rebuild, "idle passes changed the index");
    assert_eq!(w.view(), before);
    let dbs: Vec<String> = fs::read_dir(&w.root)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with("index") && n.ends_with(".db"))
        .collect();
    assert_eq!(dbs, ["index.db"], "{dbs:?}");
}

/// the same home under a renamed host re-stores
/// nothing and classes every file unchanged.
#[test]
fn a_host_rename_restores_nothing() {
    let w = Morrow::build(Set::Main);
    let revisions = count(&w, "SELECT count(*) AS n FROM a.revision");
    let blobs = count(&w, "SELECT count(*) AS n FROM a.blob");
    let config = Config {
        host: "morrow-renamed".into(),
        ..w.config()
    };
    let (mut engine, _lock) = w.engine_with(config);
    let pass = engine.run_pass().unwrap();
    assert_eq!(pass.inputs_changed, 0, "{pass:?}");
    assert_eq!(pass.bytes_stored, 0, "{pass:?}");
    drop(engine);
    assert_eq!(count(&w, "SELECT count(*) AS n FROM a.revision"), revisions);
    assert_eq!(count(&w, "SELECT count(*) AS n FROM a.blob"), blobs);
}
