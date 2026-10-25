use super::*;

#[test]
fn sorted_json_counts_and_membership_preserve_duplicates_and_opaque_ids() {
    let mut selected = SelectedMessages::default();
    assert_eq!(selected.count().unwrap(), 0);
    assert!(!selected.contains("absent").unwrap());
    assert_eq!(serde_json::to_string(&selected).unwrap(), "[]");
    let mut expected = vec!["z", "a\0", "é", "a:b:c", "a", "a\0", "🦀", ""];
    for id in &expected {
        selected.insert(id).unwrap();
    }
    expected.sort_unstable();
    assert_eq!(selected.count().unwrap(), expected.len() as u64);
    assert_eq!(
        serde_json::to_string(&selected).unwrap(),
        serde_json::to_string(&expected).unwrap()
    );
    for id in &expected {
        assert!(selected.contains(id).unwrap());
    }
    for id in ["a\0z", "a:b", "message:a", "éx"] {
        assert!(!selected.contains(id).unwrap());
    }
}
