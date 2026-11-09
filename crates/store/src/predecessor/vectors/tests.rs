use super::*;
use crate::predecessor::fixture::{self, VectorKey};

fn store(vectors: Vec<(VectorKey, Vec<f32>)>) -> (tempfile::TempDir, VectorStore) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("embeddings.db");
    fixture::vector_store(&path, "voyage-4", &vectors, 7_000);
    let store = VectorStore::open(&path, dir.path()).unwrap();
    (dir, store)
}

#[test]
fn an_addressed_store_yields_every_vector_from_its_chunk() {
    let (_dir, store) = store(vec![
        (VectorKey::Address([1; 32]), vec![1.0, 0.0, 0.0, 0.0]),
        (VectorKey::Address([2; 32]), vec![0.0, 1.0, 0.0, 0.0]),
        (VectorKey::Address([3; 32]), vec![0.0, 0.0, 0.5, -0.5]),
    ]);
    assert_eq!(store.layout(), Layout::Addressed);
    let mut identities = store.addressed().unwrap();
    identities.sort_by_key(|vector| vector.address);
    assert_eq!(identities.len(), 3);
    assert_eq!(
        identities[2],
        Addressed {
            address: [3; 32],
            model: "voyage-4".into(),
            dims: 4,
            embedded_at: Some(7_000),
        }
    );
    let mut seen = Vec::new();
    store
        .for_each(4, |id, values| {
            let floats: Vec<f32> = values
                .chunks(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            seen.push((id.to_owned(), floats));
            Ok(())
        })
        .unwrap();
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[2].0, "03".repeat(32));
    assert_eq!(seen[2].1, [0.0, 0.0, 0.5, -0.5]);
}

#[test]
fn a_store_by_message_is_counted_by_origin() {
    let (_dir, store) = store(vec![
        (
            VectorKey::Message("codex-session:s1:m1".into()),
            vec![1.0, 0.0],
        ),
        (
            VectorKey::Message("codex-session:s1:m2".into()),
            vec![0.0, 1.0],
        ),
        (
            VectorKey::Message("chatgpt-export:c1:m1".into()),
            vec![1.0, 1.0],
        ),
    ]);
    assert_eq!(store.layout(), Layout::ByMessage);
    assert_eq!(
        store.by_message_counts().unwrap(),
        BTreeMap::from([
            ("chatgpt-export".to_owned(), 1),
            ("codex-session".to_owned(), 2)
        ])
    );
    assert!(store.addressed().is_err());
}

#[test]
fn a_vector_whose_slot_is_not_set_is_an_error_not_a_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("embeddings.db");
    fixture::vector_store(
        &path,
        "voyage-4",
        &[(VectorKey::Address([1; 32]), vec![1.0, 0.0])],
        0,
    );
    Connection::open(&path)
        .unwrap()
        .execute("UPDATE message_embeddings_chunks SET validity = x'00'", [])
        .unwrap();
    let store = VectorStore::open(&path, dir.path()).unwrap();
    let error = store.for_each(2, |_, _| Ok(())).unwrap_err().to_string();
    assert!(error.contains("not held at its recorded place"), "{error}");
}

#[test]
fn a_database_that_is_not_a_vector_store_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("other.db");
    Connection::open(&path)
        .unwrap()
        .execute_batch("CREATE TABLE message_embeddings_meta (message_id TEXT)")
        .unwrap();
    let error = VectorStore::open(&path, dir.path())
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("not one of the predecessor's vector stores"),
        "{error}"
    );
}
