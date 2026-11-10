//! A document its source saves whole is stored as content-defined chunks: a
//! revision that changes one byte stores only the chunk(s) around it, and the
//! new chain reads back as the file.

use super::*;
use std::collections::BTreeSet;
use std::io::Read;

/// About 1 MiB of varied text, so cut points fall throughout.
fn varied_text(length: usize) -> String {
    let words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut text = String::with_capacity(length + 16);
    while text.len() < length {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        text.push_str(words[(state % words.len() as u64) as usize]);
        text.push_str(&(state >> 40).to_string());
        text.push(if state.is_multiple_of(11) { '\n' } else { ' ' });
    }
    text
}

fn antigravity_export(markdown: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "source": "antigravity_language_server",
        "cascadeId": "rewritten",
        "markdown": format!("### User Input\n{markdown}\n### Planner Response\nNoted"),
    }))
    .unwrap()
}

/// The chunks the shared chunker cuts, as hex names and lengths.
fn chunks(bytes: &[u8]) -> Vec<(String, u64)> {
    hashed_blobs::Chunker::plain(hashed_blobs::ChunkSizes::default())
        .chunks(bytes)
        .map(|chunk| {
            let chunk = chunk.unwrap();
            (
                blake3::hash(&chunk).to_hex().to_string(),
                chunk.len() as u64,
            )
        })
        .collect()
}

fn stored_blobs(w: &ClaudeHome) -> BTreeSet<String> {
    w.reads()
        .reader()
        .unwrap()
        .query("SELECT lower(hex(hash)) AS hash FROM a.blob", &[])
        .unwrap()
        .into_iter()
        .map(|row| row["hash"].as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn a_rewritten_document_stores_only_its_changed_chunks_and_reads_back_whole() {
    let w = ClaudeHome::new();
    let dir = w.config(64 << 20).spool_dir("antigravity-exports");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("export.json");
    let first = antigravity_export(&varied_text(1 << 20));
    let mut second = first.clone();
    let middle = second.len() / 2;
    second[middle] = if second[middle] == b'a' { b'b' } else { b'a' };
    let (old, new) = (chunks(&first), chunks(&second));
    let held: BTreeSet<_> = old.iter().map(|(hash, _)| hash.clone()).collect();
    let changed: BTreeSet<_> = new
        .iter()
        .filter(|(hash, _)| !held.contains(hash))
        .map(|(hash, _)| hash.clone())
        .collect();
    assert!(old.len() > 4 && (1..=2).contains(&changed.len()));

    let (mut engine, _lock) = w.engine(64 << 20);
    fs::write(&path, &first).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let before = stored_blobs(&w);
    // A new file renamed over the old one, as editors save.
    fs::write(dir.join("export.json.new"), &second).unwrap();
    fs::rename(dir.join("export.json.new"), &path).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(pass.blobs_appended, changed.len() as u64);
    let after = stored_blobs(&w);
    assert_eq!(
        after.difference(&before).cloned().collect::<BTreeSet<_>>(),
        changed
    );

    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let latest = &reader
        .query(
            "SELECT r.input, r.chain, r.length FROM a.revision r
             JOIN a.observation o ON o.input = r.input AND o.path LIKE '%export.json'
             ORDER BY r.number DESC LIMIT 1",
            &[],
        )
        .unwrap()[0];
    let input = latest["input"].as_str().unwrap();
    let chain = latest["chain"].as_i64().unwrap();
    assert_eq!(latest["length"].as_u64(), Some(second.len() as u64));
    let extents: Vec<(String, u64, i64)> = reader
        .query(
            &format!(
                "SELECT lower(hex(blob)) AS blob, length, blob_offset FROM a.extent
                 WHERE input = '{input}' AND chain = {chain} ORDER BY offset"
            ),
            &[],
        )
        .unwrap()
        .into_iter()
        .map(|row| {
            (
                row["blob"].as_str().unwrap().to_owned(),
                row["length"].as_u64().unwrap(),
                row["blob_offset"].as_i64().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        extents,
        new.iter()
            .map(|(hash, length)| (hash.clone(), *length, 0))
            .collect::<Vec<_>>()
    );

    let stored = polylogue_store::blobs::StoredChain::load(
        &polylogue_store::blobs::ChainSource::of(&engine.archive),
        input,
        chain,
        second.len() as u64,
    )
    .unwrap();
    let mut bytes = Vec::new();
    stored
        .reader(std::sync::Arc::new(engine.archive.blobs().unwrap()))
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes == second, "the stored chain reads back as the file");
}
