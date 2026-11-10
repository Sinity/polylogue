//! One conversation through several routes: the shown message is composed
//! field group by field group by route precedence (docs/sessions.md).
use super::*;
use proptest::prelude::*;

fn answer(text: &str) -> Message {
    let mut message = text_message("r", "s", Role::Assistant, text, None);
    message.at = None;
    message.at_quality = polylogue_model::TimeQuality::Unknown;
    message.at_precision = None;
    message
}

fn at(ms: i64) -> (Option<Timestamp>, polylogue_model::TimeQuality) {
    (
        Timestamp::from_unix_ms(ms),
        polylogue_model::TimeQuality::Intrinsic,
    )
}

fn timed(mut message: Message, ms: i64) -> Message {
    (message.at, message.at_quality) = at(ms);
    message.at_precision = Some(polylogue_model::Precision::Instant);
    message
}

/// Each input's bytes: an imported input is named by them.
fn bytes(input: &str) -> Vec<u8> {
    input.bytes().cycle().take(100).collect()
}

fn input_id(input: &str, route: Route) -> String {
    if route.is_imported() {
        polylogue_model::BlobHash::of(&bytes(input)).to_string()
    } else {
        input.to_owned()
    }
}

struct Reading {
    batch: Batch,
    bytes: Vec<u8>,
}

fn reading(input: &str, route: Route, message: Message, page_text: bool) -> Reading {
    let mut row = row(0, message);
    row.page_text = page_text;
    let mut batch = batch(input, route, 0, vec![row], &["s"]);
    batch.input.id = input_id(input, route);
    Reading {
        batch,
        bytes: bytes(input),
    }
}

/// Stores the readings in the given order, each through the writer.
fn store(readings: &[Reading], order: &[usize]) -> TestArchive {
    let w = test_archive();
    for &n in order {
        w.store_reading(readings[n].batch.clone(), Some(&readings[n].bytes))
            .unwrap();
        let report = w.writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }
    w
}

fn shown(w: &TestArchive) -> Message {
    let pool = ReadPool::open(&w.archive).unwrap();
    let mut messages = pool.reader().unwrap().messages(&sid("s"), false).unwrap();
    assert_eq!(messages.len(), 1);
    messages.remove(0)
}

/// The main route's input, then each other field group and its input.
fn routes(w: &TestArchive) -> (String, Vec<(String, String)>) {
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let main = conn
        .query_row(
            "SELECT i.id FROM message m JOIN input_ref i ON i.pk=m.input_pk",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let groups = conn
        .prepare(
            "SELECT r.field_group,i.id FROM message_route r JOIN input_ref i ON i.pk=r.input_pk
             ORDER BY r.field_group",
        )
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    (main, groups)
}

#[test]
fn routes_merge_by_field() {
    // The capture has the provider's text and stop reason; the export adds
    // the time, model and usage the capture lacks.
    let mut captured = answer("captured text");
    captured.stop_reason = Some("end_turn".into());
    let mut exported = timed(answer("exported text"), 5_000);
    exported.model = Some("model-x".into());
    exported.usage.output = Some(7);
    let readings = [
        reading("capture", Route::Capture, captured.clone(), false),
        reading("export", Route::Export, exported.clone(), false),
    ];
    let mut expected = captured;
    expected.at = exported.at;
    expected.at_quality = exported.at_quality;
    expected.at_precision = exported.at_precision;
    expected.model = exported.model.clone();
    expected.usage = exported.usage.clone();
    for order in [[0, 1], [1, 0]] {
        let w = store(&readings, &order);
        assert_eq!(shown(&w), expected, "order {order:?}");
        assert_eq!(
            routes(&w),
            (
                "capture".into(),
                vec![
                    ("model".into(), input_id("export", Route::Export)),
                    ("time".into(), input_id("export", Route::Export)),
                    ("usage".into(), input_id("export", Route::Export)),
                ]
            ),
            "order {order:?}"
        );
    }
}

#[test]
fn newer_export_keeps_dropped_field() {
    // The newer export no longer carries the model the older one recorded.
    let mut older = timed(answer("text"), 5_000);
    older.model = Some("model-x".into());
    older.stop_reason = Some("end_turn".into());
    let mut newer = older.clone();
    newer.model = None;
    let readings = [
        reading("export-older", Route::Export, older.clone(), false),
        reading("export-newer", Route::Export, newer, false),
    ];
    for order in [[0, 1], [1, 0]] {
        let w = store(&readings, &order);
        assert_eq!(shown(&w), older, "order {order:?}");
    }
}

#[test]
fn page_capture_never_overrides_a_field() {
    // Page text outranks nothing, though the capture route outranks exports;
    // it supplies only a field no other route has.
    let mut page = timed(answer("page text"), 9_000);
    page.model = Some("model-from-page".into());
    let mut exported = timed(answer("exported text"), 5_000);
    exported.stop_reason = Some("end_turn".into());
    let readings = [
        reading("page", Route::Capture, page, true),
        reading("export", Route::Export, exported.clone(), false),
    ];
    let mut expected = exported;
    expected.model = Some("model-from-page".into());
    for order in [[0, 1], [1, 0]] {
        let w = store(&readings, &order);
        assert_eq!(shown(&w), expected, "order {order:?}");
        assert_eq!(
            routes(&w),
            (
                input_id("export", Route::Export),
                vec![("model".into(), "page".into())]
            ),
            "order {order:?}"
        );
    }
}

/// Every row the composition writes, named by natural keys.
fn composed_rows(w: &TestArchive) -> Vec<String> {
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let mut rows = Vec::new();
    for sql in [
        "SELECT m.id,s.id,i.id,m.source_chain,m.source_start,m.source_end,m.session_order,
                m.role,m.written_by,m.prompt_kind,m.model,m.at,hex(m.at_time),m.parent_id,m.in_progress,
                m.input_tokens,m.output_tokens,m.stop_reason,m.attrs_json,m.words,m.cost_nusd
         FROM message m JOIN session s ON s.pk=m.session_pk JOIN input_ref i ON i.pk=m.input_pk",
        "SELECT r.field_group,i.id,r.chain FROM message_route r JOIN input_ref i ON i.pk=r.input_pk",
        "SELECT b.idx,b.kind,hex(b.body),t.body IS NOT NULL FROM block b LEFT JOIN text t ON t.id=b.text_id",
        "SELECT i.id,s.chain,s.route,s.completion,s.main,hex(s.version_hash) FROM message_source s
         JOIN input_ref i ON i.pk=s.input_pk",
    ] {
        let mut statement = conn.prepare(sql).unwrap();
        let columns = statement.column_count();
        let mut found: Vec<String> = statement
            .query_map([], |r| {
                (0..columns)
                    .map(|n| r.get::<_, rusqlite::types::Value>(n).map(|v| format!("{v:?}")))
                    .collect::<rusqlite::Result<Vec<_>>>()
                    .map(|values| values.join("|"))
            })
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        found.sort();
        rows.extend(found);
    }
    rows
}

#[derive(Debug, Clone)]
struct Generated {
    route: Route,
    page_text: bool,
    text: u8,
    at: Option<i64>,
    model: Option<u8>,
    output: Option<u64>,
    stop: bool,
}

fn generated() -> impl Strategy<Value = Generated> {
    (
        prop::sample::select(vec![
            Route::Live,
            Route::Capture,
            Route::Export,
            Route::Copy,
        ]),
        any::<bool>(),
        0u8..3,
        prop::option::of(1_000i64..1_003),
        prop::option::of(0u8..2),
        prop::option::of(1u64..3),
        any::<bool>(),
    )
        .prop_map(
            |(route, page_text, text, at, model, output, stop)| Generated {
                route,
                page_text,
                text,
                at,
                model,
                output,
                stop,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Readings arriving in any order produce identical rows.
    #[test]
    fn arrival_order_property_identical_rows(
        readings in prop::collection::vec(generated(), 2..4),
        rotation in 0usize..4,
    ) {
        let batches: Vec<_> = readings
            .iter()
            .enumerate()
            .map(|(n, r)| {
                let mut message = answer(&format!("text {}", r.text));
                if let Some(ms) = r.at {
                    message = timed(message, ms);
                }
                message.model = r.model.map(|m| format!("model-{m}"));
                message.usage.output = r.output;
                message.stop_reason = r.stop.then(|| "end_turn".into());
                reading(&format!("input-{n}"), r.route, message, r.page_text)
            })
            .collect();
        let forward: Vec<usize> = (0..batches.len()).collect();
        let reversed: Vec<usize> = forward.iter().rev().copied().collect();
        let mut rotated = forward.clone();
        rotated.rotate_left(rotation % batches.len());
        let expected = composed_rows(&store(&batches, &forward));
        prop_assert_eq!(&composed_rows(&store(&batches, &reversed)), &expected);
        prop_assert_eq!(&composed_rows(&store(&batches, &rotated)), &expected);
    }
}
