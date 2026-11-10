//! In primary mode a document its source saves whole goes to Sinex as
//! content-defined chunks: each capture's material frame names its chunks in
//! order, the spool holds every chunk's encoding for the frame directory, and
//! a capture after a one-byte change names only the changed chunk(s) Sinex
//! does not already hold.
use super::client;
use polylogue_engine::Engine;
use polylogue_store::Archive;
use polylogue_testkit::ClaudeHome;
use sinex_protocol::{Ingress, MaterialChunk, MaterialFrame};
use std::collections::BTreeSet;
use std::io::Read;
use std::sync::Arc;

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

/// The chunks the shared chunker cuts, named as Sinex names blobs.
fn chunks(bytes: &[u8]) -> Vec<MaterialChunk> {
    hashed_blobs::Chunker::plain(hashed_blobs::ChunkSizes::default())
        .chunks(bytes)
        .map(|chunk| {
            let chunk = chunk.unwrap();
            MaterialChunk {
                blob: format!("blake3:{}", blake3::hash(&chunk).to_hex()),
                length: chunk.len() as u64,
            }
        })
        .collect()
}

/// The material frames that captured the export, in queue order.
fn captures(sdk: &crate::Sinex) -> Vec<(String, MaterialFrame)> {
    let queue = sdk.queue().unwrap();
    let mut found = Vec::new();
    let mut after = None;
    loop {
        let page = queue.pending_after(after, 64).unwrap();
        if page.is_empty() {
            return found;
        }
        for entry in page {
            after = Some(entry.order);
            if let Ingress::Material { material, .. } = queue.decode_frame(&entry.identity).unwrap()
                && material.input.locator.ends_with("export.json")
                && material.parser != "polylogue-observation"
            {
                found.push((entry.record.identity.clone(), *material));
            }
        }
    }
}

#[test]
fn a_rewritten_document_is_sent_as_chunks_naming_only_its_changed_chunks_as_new() {
    let home = ClaudeHome::new();
    let config = home.config(64 << 20);
    let dir = config.spool_dir("antigravity-exports");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("export.json");
    let first = antigravity_export(&varied_text(1 << 20));
    let mut second = first.clone();
    let middle = second.len() / 2;
    second[middle] = if second[middle] == b'a' { b'b' } else { b'a' };

    let store = home.dir.path().join("sdk");
    let sdk = Arc::new(client(&store));
    let (archive, _lock) = Archive::open_primary(&home.root, &store).unwrap();
    let mut engine = Engine::on_sinex(
        config,
        archive,
        sdk.clone(),
        polylogue_testkit::stopped_attempt(),
    )
    .unwrap();
    std::fs::write(&path, &first).unwrap();
    engine.run_pass().unwrap();
    std::fs::write(dir.join("export.json.new"), &second).unwrap();
    std::fs::rename(dir.join("export.json.new"), &path).unwrap();
    engine.run_pass().unwrap();

    let captured = captures(&sdk);
    let [(_, old), (identity, new)] = captured.as_slice() else {
        panic!("two captures of the export expected: {captured:?}");
    };
    assert_eq!(old.chunks, chunks(&first));
    assert_eq!(new.chunks, chunks(&second));
    assert_eq!(
        new.blob,
        format!("blake3:{}", blake3::hash(&second).to_hex())
    );
    assert_eq!(new.length, second.len() as u64);
    let held: BTreeSet<_> = old.chunks.iter().map(|chunk| &chunk.blob).collect();
    let fresh = new
        .chunks
        .iter()
        .filter(|chunk| !held.contains(&chunk.blob))
        .count();
    assert!(old.chunks.len() > 4 && (1..=2).contains(&fresh));

    let queue = sdk.queue().unwrap();
    let encodings: Vec<MaterialChunk> = queue
        .encoded_body(identity)
        .unwrap()
        .into_iter()
        .map(|(_, descriptor)| MaterialChunk {
            blob: descriptor.hash.to_string(),
            length: descriptor.decoded_len,
        })
        .collect();
    assert_eq!(encodings, new.chunks);
    let mut bytes = Vec::new();
    queue
        .body(identity)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes == second, "the spooled chunks read back as the file");
}
