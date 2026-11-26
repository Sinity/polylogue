use super::*;

#[test]
fn counts_blob_lists_each_class_once() {
    let class = EntryClass::from_bits(1).unwrap();
    let mut counts = ItemCounts {
        tokens_full: 5,
        ..ItemCounts::default()
    };
    counts.classes.insert(class, 3);
    let encoded = encode_counts(&counts);
    assert_eq!(decode_counts(&encoded).unwrap().classes[&class], 3);
    // The same class entry appended again.
    let mut repeated = encoded.clone();
    repeated.extend_from_slice(&encoded[1..]);
    assert!(matches!(
        decode_counts(&repeated),
        Err(StoreError::StoredValue { column: None, .. })
    ));
}
