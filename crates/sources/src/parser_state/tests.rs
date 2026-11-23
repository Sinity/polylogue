use super::*;

fn map() -> Map<Vec<u32>> {
    let mut map = Map::default();
    map.bind(Handle::default(), "numbers".into());
    map
}

#[test]
fn a_removed_key_reads_as_absent_and_returns_its_last_value() {
    let map = map();
    map.insert("a".into(), vec![1]);
    assert_eq!(map.remove("a"), Some(vec![1]));
    assert_eq!(map.get("a"), None);
    assert_eq!(map.remove("a"), None);
}

#[test]
fn a_changed_entry_is_written_back_when_it_goes_out_of_scope() {
    let map = map();
    map.insert("a".into(), vec![1]);
    map.get_mut("a").unwrap().push(2);
    assert_eq!(map.get("a"), Some(vec![1, 2]));
    map.get_or_insert("b".into(), Vec::new).push(3);
    assert_eq!(map.get("b"), Some(vec![3]));
    assert!(map.get_mut("missing").is_none());
}

#[test]
fn clearing_a_map_forgets_its_keys() {
    let mut map = map();
    map.insert("a".into(), vec![1]);
    map.clear();
    assert_eq!(map.get("a"), None);
    assert!(map.is_bound());
}

#[test]
fn a_set_accepts_each_value_once() {
    let mut set = Set::<(u32, String)>::default();
    set.bind(Handle::default(), "pairs".into());
    assert!(set.insert((1, "x".into())));
    assert!(!set.insert((1, "x".into())));
    assert!(set.insert((2, "x".into())));
}
