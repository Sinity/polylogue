use super::*;

#[test]
fn saved_hashes_equal_blake3_at_chunk_and_tree_boundaries() {
    let bytes: Vec<_> = (0..2_000_000).map(|n| (n % 251) as u8).collect();
    let mut hash = PrefixHash::default();
    let mut previous = 0;
    for length in [
        0, 1, 1023, 1024, 1025, 2048, 3072, 4096, 8191, 8192, 8193, 65535, 65536, 65537, 131072,
        199999, 200000, 262144, 524287, 524288, 524289, 1_000_000, 2_000_000,
    ] {
        hash.update(&bytes[previous..length]);
        assert!(hash.valid());
        assert_eq!(hash.len(), length as u64);
        assert_eq!(hash.finalize(), blake3::hash(&bytes[..length]), "{length}");
        hash = serde_json::from_slice(&serde_json::to_vec(&hash).unwrap()).unwrap();
        previous = length;
    }
}
