use super::*;

fn key(n: u64) -> [u8; 32] {
    *blake3::hash(&n.to_le_bytes()).as_bytes()
}

#[test]
fn spilled_updates_and_rewrites_keep_each_keys_latest_value() {
    let mut rows = KeyRows::new(64 * 1024).unwrap();
    for n in 0..1_000 {
        rows.set(key(n), n as i64 + 1).unwrap();
    }
    for n in 0..1_000 {
        assert_eq!(rows.get(&key(n)).unwrap(), Some(n as i64 + 1));
    }
    for n in 1_000..1_100 {
        assert_eq!(rows.get(&key(n)).unwrap(), None);
    }
    rows.transform_values(|value| -value).unwrap();
    for n in 0..1_000 {
        assert_eq!(rows.get(&key(n)).unwrap(), Some(-(n as i64 + 1)));
    }
    for n in (0..1_000).step_by(2) {
        rows.set(key(n), n as i64 + 1).unwrap();
    }
    for n in 0..1_000 {
        let expected = (n as i64 + 1) * if n % 2 == 0 { 1 } else { -1 };
        assert_eq!(rows.get(&key(n)).unwrap(), Some(expected));
    }
    rows.clear();
    for n in 0..1_000 {
        assert_eq!(rows.get(&key(n)).unwrap(), None);
    }
}
