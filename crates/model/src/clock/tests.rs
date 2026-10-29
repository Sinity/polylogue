use super::WallClock;
use std::sync::{
    Arc,
    atomic::{AtomicI64, Ordering},
};
#[test]
fn cloned_clock_uses_one_explicit_sampler() {
    let seconds = Arc::new(AtomicI64::new(100));
    let source = seconds.clone();
    let clock = WallClock::new(move || {
        jiff::Timestamp::from_second(source.load(Ordering::SeqCst)).unwrap()
    });
    let copy = clock.clone();
    assert_eq!(clock.now().as_second(), 100);
    seconds.store(200, Ordering::SeqCst);
    assert_eq!(copy.now().as_second(), 200);
}
#[test]
fn fixed_clock_has_no_environment_or_day_default() {
    let instant = "2026-03-29T01:00:00Z".parse().unwrap();
    assert_eq!(WallClock::fixed(instant).now(), instant);
}
