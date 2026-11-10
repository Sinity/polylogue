//! A session's messages read in source-time order across routes and in
//! source position within each input (docs/sessions.md, messages).
use super::*;
use proptest::prelude::*;

fn turn(id: &str, ms: Option<i64>) -> Message {
    let mut message = text_message(id, "s", Role::User, id, None);
    message.at = ms.and_then(Timestamp::from_unix_ms);
    if message.at.is_none() {
        message.at_quality = polylogue_model::TimeQuality::Unknown;
        message.at_precision = None;
    }
    message
}

/// The bytes of one step of an input: an imported input is named by them.
fn bytes(input: &str, start: u64) -> Vec<u8> {
    format!("{input}@{start};")
        .bytes()
        .cycle()
        .take(100)
        .collect()
}

/// One step of an input: its first byte and its turns, each at its own offset.
fn step(input: &str, route: Route, start: u64, turns: &[(&str, Option<i64>)]) -> Batch {
    let rows = turns
        .iter()
        .enumerate()
        .map(|(n, (id, ms))| row(start + 10 * n as u64, turn(id, *ms)))
        .collect();
    let mut batch = batch(input, route, start, rows, &["s"]);
    if route.is_imported() {
        batch.input.id = polylogue_model::BlobHash::of(&bytes(input, 0)).to_string();
    }
    batch
}

fn store(steps: &[&Batch]) -> TestArchive {
    let w = test_archive();
    for batch in steps {
        let start = match batch.change {
            Change::Appended { from } => from,
            _ => 0,
        };
        let input = batch
            .observation
            .as_ref()
            .unwrap()
            .path
            .trim_start_matches("/p/");
        w.store_reading((*batch).clone(), Some(&bytes(input, start)))
            .unwrap();
        let report = w.writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }
    w
}

fn order(w: &TestArchive) -> Vec<String> {
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader()
        .unwrap()
        .messages(&sid("s"), false)
        .unwrap()
        .iter()
        .map(|m| m.id.native().as_str().to_owned())
        .collect()
}

#[test]
fn export_and_capture_turns_read_in_time_order() {
    // The export holds an older turn the capture no longer has; the capture
    // holds a newer turn the export does not have yet. The capture's input ID
    // sorts before the export's, so ordering by input would put the capture's
    // turns first. Untimed turns stay after the turn before them in their
    // input, including across an append.
    let export = step(
        "export",
        Route::Export,
        0,
        &[
            ("older", Some(1_000)),
            ("after-older", None),
            ("shared", Some(2_000)),
        ],
    );
    let capture = step(
        "0-capture",
        Route::Capture,
        0,
        &[("shared", Some(2_000)), ("newer", Some(3_000))],
    );
    let appended = step(
        "0-capture",
        Route::Capture,
        100,
        &[("after-newer", None), ("newest", Some(4_000))],
    );
    // A rebuild reads the capture's whole chain in one step.
    let whole = step(
        "0-capture",
        Route::Capture,
        0,
        &[
            ("shared", Some(2_000)),
            ("newer", Some(3_000)),
            ("after-newer", None),
            ("newest", Some(4_000)),
        ],
    );
    assert!(capture.input.id < export.input.id);
    let expected = [
        "older",
        "after-older",
        "shared",
        "newer",
        "after-newer",
        "newest",
    ];
    for steps in [
        vec![&export, &capture, &appended],
        vec![&capture, &export, &appended],
        vec![&capture, &appended, &export],
        vec![&export, &whole],
        vec![&whole, &export],
    ] {
        assert_eq!(order(&store(&steps)), expected, "{} steps", steps.len());
    }
}

#[derive(Debug, Clone)]
struct Timeline {
    /// Each turn's time when it has one, and whether input A and input B
    /// hold it; times rise along the timeline.
    turns: Vec<(Option<i64>, bool, bool)>,
    /// How many of A's turns its first step holds; the rest are appended.
    split: usize,
}

fn timeline() -> impl Strategy<Value = Timeline> {
    (
        prop::collection::vec(
            (any::<bool>(), 0u8..3).prop_map(|(timed, held)| (timed, held != 1, held != 0)),
            2..7,
        ),
        0usize..7,
    )
        .prop_map(|(turns, split)| Timeline {
            turns: turns
                .into_iter()
                .enumerate()
                .map(|(n, (timed, a, b))| (timed.then_some(1_000 * (n as i64 + 1)), a, b))
                .collect(),
            split,
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Every arrival order of two inputs' readings, one of them appended in
    /// two steps, gives one session order, with timed turns in time order.
    #[test]
    fn arrival_order_gives_one_session_order(timeline in timeline()) {
        let names: Vec<String> = (0..timeline.turns.len()).map(|n| format!("t{n}")).collect();
        let held = |pick: fn(&(Option<i64>, bool, bool)) -> bool| -> Vec<(&str, Option<i64>)> {
            timeline
                .turns
                .iter()
                .zip(&names)
                .filter(|(turn, _)| pick(turn))
                .map(|(turn, name)| (name.as_str(), turn.0))
                .collect()
        };
        let a = held(|t| t.1);
        let b = held(|t| t.2);
        let split = timeline.split.min(a.len());
        let a_first = step("a", Route::Live, 0, &a[..split]);
        let a_appended = step("a", Route::Live, 100, &a[split..]);
        let b_whole = step("b", Route::Capture, 0, &b);
        let expected = order(&store(&[&a_first, &a_appended, &b_whole]));
        prop_assert_eq!(&order(&store(&[&a_first, &b_whole, &a_appended])), &expected);
        prop_assert_eq!(&order(&store(&[&b_whole, &a_first, &a_appended])), &expected);
        let times: std::collections::BTreeMap<&str, i64> = names
            .iter()
            .zip(&timeline.turns)
            .filter_map(|(name, turn)| Some((name.as_str(), turn.0?)))
            .collect();
        let timed: Vec<i64> = expected.iter().filter_map(|id| times.get(id.as_str()).copied()).collect();
        prop_assert!(timed.is_sorted(), "{expected:?}");
    }
}
