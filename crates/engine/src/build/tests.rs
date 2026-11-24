use super::*;

/// Four large inputs in flight at once, each holding a range's share: the
/// readings held together never exceed the allowance of two ranges, and
/// every input is read. A share above the allowance runs alone.
#[test]
fn pool_admits_by_prepared_memory() {
    let range = 64 << 20;
    let admission = Admission::new(2 * range);
    let most = Mutex::new(0);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                let admitted = admission.admit(range, || false).unwrap();
                let held = admission.held();
                let mut most = most.lock().unwrap();
                *most = (*most).max(held);
                drop(most);
                std::thread::sleep(std::time::Duration::from_millis(20));
                drop(admitted);
            });
        }
    });
    let most = *most.lock().unwrap();
    assert!((range..=2 * range).contains(&most), "{most}");
    assert_eq!(admission.held(), 0);
    let alone = admission.admit(10 * range, || false).unwrap();
    assert_eq!(admission.held(), 2 * range);
    assert!(matches!(
        admission.admit(1, || true),
        Err(EngineError::Stopped)
    ));
    drop(alone);
}
