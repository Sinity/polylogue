use super::*;
use serde_json::json;

#[test]
fn poisoned_locks_keep_their_values() {
    let mutex = Mutex::new(0);
    let _ = std::panic::catch_unwind(|| {
        let mut value = mutex.lock().unwrap();
        *value = 7;
        panic!("poison");
    });
    assert_eq!(*mutex.locked(), 7);
    let lock = RwLock::new(0);
    let _ = std::panic::catch_unwind(|| {
        let mut value = lock.write().unwrap();
        *value = 9;
        panic!("poison");
    });
    assert_eq!(*lock.locked(), 9);
}

#[test]
fn json_fields_keep_their_types() {
    let value =
        json!({"text":"yes", "unsigned":42, "signed":-7, "flag":true, "list":[1], "map":{"a":2}});
    assert_eq!(value.str("text"), Some("yes"));
    assert_eq!(value.u64("unsigned"), Some(42));
    assert_eq!(value.i64("signed"), Some(-7));
    assert_eq!(value.bool("flag"), Some(true));
    assert_eq!(value.array("list").unwrap(), &vec![json!(1)]);
    assert_eq!(value.object("map").unwrap().get("a"), Some(&json!(2)));
    assert_eq!(value.str("unsigned"), None);
    assert_eq!(value.u64("signed"), None);
    assert_eq!(value.bool("missing"), None);
}

#[test]
fn hex_encodes_all_byte_values() {
    let bytes: Vec<_> = (0..=u8::MAX).collect();
    assert_eq!(
        hex(&bytes),
        concat!(
            "000102030405060708090a0b0c0d0e0f",
            "101112131415161718191a1b1c1d1e1f",
            "202122232425262728292a2b2c2d2e2f",
            "303132333435363738393a3b3c3d3e3f",
            "404142434445464748494a4b4c4d4e4f",
            "505152535455565758595a5b5c5d5e5f",
            "606162636465666768696a6b6c6d6e6f",
            "707172737475767778797a7b7c7d7e7f",
            "808182838485868788898a8b8c8d8e8f",
            "909192939495969798999a9b9c9d9e9f",
            "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf",
            "b0b1b2b3b4b5b6b7b8b9babbbcbdbebf",
            "c0c1c2c3c4c5c6c7c8c9cacbcccdcecf",
            "d0d1d2d3d4d5d6d7d8d9dadbdcdddedf",
            "e0e1e2e3e4e5e6e7e8e9eaebecedeeef",
            "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
        )
    );
    assert_eq!(hex(&[]), "");
}

#[test]
fn thread_runs_with_its_name() {
    let thread = spawn_named("helper-test", || {
        (std::thread::current().name().unwrap().to_owned(), 7)
    })
    .unwrap();
    assert_eq!(thread.join().unwrap(), ("helper-test".to_owned(), 7));
}

#[test]
fn large_numbers_in_opaque_json_are_lossless() {
    let value: crate::Json =
        serde_json::from_str(r#"{"n":123456789012345678901234567890}"#).unwrap();
    assert_eq!(
        serde_json::to_string(&value).unwrap(),
        r#"{"n":123456789012345678901234567890}"#
    );
}

#[test]
fn poisoned_condition_wait_keeps_the_stored_value() {
    let mutex = Mutex::new(0);
    let _ = std::panic::catch_unwind(|| {
        let mut value = mutex.lock().unwrap();
        *value = 7;
        panic!("poison");
    });
    let condition = std::sync::Condvar::new();
    let held = mutex
        .lock()
        .expect_err("the fixture mutex is poisoned")
        .into_inner();
    let (value, _) = unpoisoned(condition.wait_timeout(held, std::time::Duration::ZERO));
    assert_eq!(*value, 7);
}
