use super::*;
use serde_json::json;

#[test]
fn poisoned_locks_keep_their_values() {
    let mutex = Mutex::new(0);
    let _ = std::panic::catch_unwind(|| {
        *mutex.lock().unwrap() = 7;
        panic!("poison");
    });
    assert_eq!(*mutex.locked(), 7);
    let lock = RwLock::new(0);
    let _ = std::panic::catch_unwind(|| {
        *lock.write().unwrap() = 9;
        panic!("poison");
    });
    assert_eq!(*lock.locked(), 9);
    *lock.write_locked() = 10;
    assert_eq!(*lock.locked(), 10);
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
    assert_eq!(hex(&[0, 1, 15, 16, 127, 128, 254, 255]), "00010f107f80feff");
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
        *mutex.lock().unwrap() = 7;
        panic!("poison");
    });
    let condition = std::sync::Condvar::new();
    let (value, _) = unpoisoned(condition.wait_timeout(mutex.locked(), std::time::Duration::ZERO));
    assert_eq!(*value, 7);
}
