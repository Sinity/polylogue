//! The logical day comes from the operator's settings.

use super::*;

#[test]
fn the_logical_day_follows_its_settings() {
    let mut w = morrow();
    let mut a = ask("since:2026-10-05");
    a.day = None;
    let machine_default = run_q(&w, a).unwrap();
    w.fx.setting("day.zone", "Pacific/Kiritimati").unwrap();
    w.fx.setting("day.logical_start", "00:00").unwrap();
    let pool = w.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    let day = logical_day(&reader).unwrap();
    assert_eq!(day.zone_name(), Some("Pacific/Kiritimati"));
    assert_eq!(day.start_minutes, 0);
    drop(reader);
    let mut a = ask("since:2026-10-05");
    a.day = None;
    let resolved = run_q(&w, a).unwrap();
    assert_eq!(resolved["resolved"]["zone"], "Pacific/Kiritimati");
    assert_eq!(resolved["resolved"]["day_start_minutes"], 0);
    assert_ne!(resolved["resolved"], machine_default["resolved"]);
    w.fx.setting("day.zone", "Mars/Olympus").unwrap();
    let mut a = ask("kind:*");
    a.day = None;
    let refused = run_q(&w, a).unwrap_err();
    assert!(refused.message.contains("day.zone"), "{}", refused.message);
}
