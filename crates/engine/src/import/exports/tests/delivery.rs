use super::*;
use polylogue_model::batch::{ExportCursor, ExportDelivery, ExportDeliverySource};

fn source() -> Vec<u8> {
    let mut newer = conversation("new", 2, &[("new-message", "new body")]);
    newer["current_node"] = json!("new-message");
    let mut older = conversation("old", 1, &[("old-message", "old body")]);
    older["current_node"] = json!("old-message");
    serde_json::to_vec(&vec![newer, older]).unwrap()
}

fn plan(home: &ClaudeHome, bytes: &[u8]) -> (Engine, polylogue_store::ArchiveLock, Plan) {
    let path = home.dir.path().join("conversations.json");
    fs::write(&path, bytes).unwrap();
    let (mut engine, lock) = engine(home, 16);
    let captured = Captured::stream(&engine.archive, &mut fs::File::open(&path).unwrap()).unwrap();
    let origin = polylogue_sources::origin::by_name("chatgpt").unwrap();
    let plan = engine
        .import_plan(&path, origin, captured, None, &mut ImportReport::default())
        .unwrap()
        .unwrap();
    (engine, lock, plan)
}

fn parser(plan: &Plan) -> Box<dyn ExportParser> {
    let Some(polylogue_sources::Parse::Export(parser)) = (plan.origin.open)(&plan.path) else {
        panic!("native export parser missing")
    };
    parser
}

fn collect(engine: &Engine, plan: &Plan) -> Vec<ParseStep> {
    let mut steps = Vec::new();
    parse_export(plan, &mut *parser(plan), &engine.blobs, |item| {
        let (mut step, cursor) = item.into_parse(plan, &engine.blobs, &engine.stopped_attempt)?;
        step.prepare();
        steps.push(step);
        Ok(cursor)
    })
    .unwrap();
    steps
}

#[test]
fn semantic_delivery_order_keeps_nonmonotone_physical_item_ranges() {
    let home = ClaudeHome::new();
    let raw = source();
    let (engine, _lock, plan) = plan(&home, &raw);
    let steps = collect(&engine, &plan);
    let mut cursor: Option<ExportCursor> = None;
    let mut positions = Vec::new();
    let mut texts = Vec::new();
    let mut items = 0;
    for (ordinal, step) in steps.iter().enumerate() {
        step.validate_export().unwrap();
        let delivery = step.delivery.unwrap();
        assert_eq!(delivery.ordinal, ordinal as u64);
        cursor = Some(match cursor {
            None => ExportCursor::from_first(&delivery).unwrap(),
            Some(cursor) => cursor.advance(&delivery).unwrap(),
        });
        match delivery.source {
            ExportDeliverySource::ContainerHeader => {
                assert_eq!(ordinal, 0);
                assert_eq!((step.start, step.parsed_upto), (0, 0));
            }
            ExportDeliverySource::Item { start, end } => {
                let value: serde_json::Value =
                    serde_json::from_slice(&raw[start as usize..end as usize]).unwrap();
                assert!(value["mapping"].is_object());
                assert_eq!((step.start, step.parsed_upto), (0, 0));
                positions.push(start);
                texts.extend(
                    step.complete_output()
                        .unwrap()
                        .messages
                        .iter()
                        .map(|row| row.message.text()),
                );
                items += 1;
            }
            ExportDeliverySource::Complete {
                length,
                item_count,
                delivery_count,
            } => {
                assert_eq!(ordinal, steps.len() - 1);
                assert_eq!(length, raw.len() as u64);
                assert_eq!(item_count, items);
                assert_eq!(delivery_count, ordinal as u64);
                assert_eq!(step.parsed_upto, raw.len() as u64);
                assert!(step.progress.reached_end());
                assert!(
                    step.checkpoint
                        .as_ref()
                        .unwrap()
                        .matches(&plan.origin.reading_version(), raw.len() as u64)
                );
            }
        }
    }
    assert_eq!(items, 2);
    assert!(positions.windows(2).any(|pair| pair[0] > pair[1]));
    assert_eq!(texts, ["old body", "new body"]);
    assert_ne!(cursor.unwrap().prefix_hash, ExportDelivery::seed());
}

#[test]
fn interrupted_delivery_replays_the_same_sealed_prefix_from_archived_bytes() {
    let home = ClaudeHome::new();
    let raw = source();
    let (engine, _lock, plan) = plan(&home, &raw);
    let mut prefix = Vec::new();
    let mut first_item = true;
    let result = parse_export(&plan, &mut *parser(&plan), &engine.blobs, |item| {
        let (mut step, cursor) = item.into_parse(&plan, &engine.blobs, &engine.stopped_attempt)?;
        step.prepare();
        let delivery = step.delivery.unwrap();
        if matches!(delivery.source, ExportDeliverySource::Item { .. }) {
            if !first_item {
                return Err(std::io::Error::other("synthetic interrupted callback").into());
            }
            first_item = false;
        }
        prefix.push(delivery);
        Ok(cursor)
    });
    assert!(result.is_err());
    assert!(
        prefix
            .iter()
            .all(|delivery| !matches!(delivery.source, ExportDeliverySource::Complete { .. }))
    );
    fs::remove_file(&plan.path).unwrap();
    let replay = collect(&engine, &plan);
    let deliveries: Vec<_> = replay.iter().map(|step| step.delivery.unwrap()).collect();
    assert_eq!(&deliveries[..prefix.len()], prefix.as_slice());
    assert!(matches!(
        deliveries.last().unwrap().source,
        ExportDeliverySource::Complete { item_count: 2, .. }
    ));
}

fn assert_refusal(step: &ParseStep) {
    assert_eq!(
        step.stopped_attempt,
        Some(polylogue_testkit::stopped_attempt())
    );
    assert!(matches!(
        step.delivery.unwrap().source,
        ExportDeliverySource::ContainerHeader
    ));
    step.validate_export().unwrap();
    assert_eq!((step.start, step.parsed_upto), (0, 0));
    assert!(!step.progress.reached_end());
    assert!(step.checkpoint.is_none());
    assert!(step.complete_output().unwrap().messages.is_empty());
    assert!(step.complete_output().unwrap().sessions.is_empty());
    assert_eq!(step.counts.records, 1);
    assert_eq!(step.counts.unreadable.values().sum::<u64>(), 1);
}

#[test]
fn malformed_scan_publishes_one_unreadable_refusal_without_a_checkpoint() {
    for suffix in [",", ",]", "] false"] {
        let home = ClaudeHome::new();
        let raw =
            format!("[{{\"id\":\"valid-prefix\",\"mapping\":{{}},\"current_node\":null}}{suffix}");
        let (mut engine, _lock, mut plan) = plan(&home, raw.as_bytes());
        let steps = collect(&engine, &plan);
        assert_eq!(steps.len(), 1);
        assert_refusal(&steps[0]);
        let mut pass = Pass::default();
        let mut export = parser(&plan);
        engine
            .execute_export(&mut plan, &mut *export, &mut pass)
            .unwrap();
        assert_eq!(plan.parse_pos, 0);
        assert!(plan.failed && plan.failure_recorded);
        assert_eq!(
            home.count(
                "SELECT count(*) AS n FROM parse_state WHERE status='refused' AND reason IS NOT NULL AND at_length IS NOT NULL"
            ),
            1
        );
        assert_eq!(home.count("SELECT count(*) AS n FROM message"), 0);
        assert_eq!(home.count("SELECT parsed_upto AS n FROM parse_state"), 0);
        assert_eq!(
            home.count("SELECT sum(unreadable) AS n FROM parse_state"),
            1
        );
        assert_eq!(
            home.count("SELECT count(*) AS n FROM parse_state WHERE checkpoint IS NOT NULL"),
            0
        );
        assert_eq!(
            home.count("SELECT count(*) AS n FROM wire_reading WHERE json_extract(state,'$.refused_export')=1 AND json_extract(state,'$.export_cursor') IS NULL AND json_type(state,'$.progress.refused')='object' AND json_extract(state,'$.last.footprint.source.kind')='container_header'"),
            1
        );
        let wire = engine
            .reads
            .reader()
            .unwrap()
            .wire_shapes(polylogue_store::wire_shapes::Scope::current(None))
            .unwrap();
        assert_eq!(wire.coverage.len(), 1);
        // The refusal is the chain's parse state (docs/archive.md), which
        // coverage names over the source's unreadable record.
        assert_eq!(
            wire.coverage[0].status(),
            polylogue_store::wire_shapes::CoverageStatus::Refused
        );
        fs::remove_file(&plan.path).unwrap();
        let (progress, stored) = {
            let reader = engine.reads.reader().unwrap();
            let progress = reader.input_progress(&plan.input.id).unwrap().unwrap();
            let stored = reader
                .stored_chain(&plan.input.id, progress.chain, progress.length)
                .unwrap();
            (progress, stored)
        };
        assert_eq!(progress.length, raw.len() as u64);
        let mut retained = Vec::new();
        std::io::Read::read_to_end(&mut stored.reader(Arc::clone(&engine.blobs)), &mut retained)
            .unwrap();
        assert_eq!(retained, raw.as_bytes());
        let custody = [
            home.count("SELECT count(*) AS n FROM a.revision"),
            home.count("SELECT count(*) AS n FROM a.blob"),
            home.count("SELECT sum(length) AS n FROM a.extent"),
        ];
        // A replacement is planned from the archive's actual immutable chain,
        // independently of the original import plan's acquisition disposition.
        let replacement = Plan::stored(
            plan.path,
            plan.origin,
            plan.input,
            Change::Reparse {
                chain: progress.chain,
            },
            None,
            progress.length,
            0,
            pool::Bytes::Stored(Arc::new(stored)),
            engine.config.chunk,
            false,
        );
        let mut replay = Pass::default();
        engine
            .replace_readings(vec![replacement], &mut replay)
            .unwrap();
        assert!(replay.errors.is_empty(), "{replay:?}");
        assert_eq!(replay.bytes_stored, 0);
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert_eq!(
            [
                home.count("SELECT count(*) AS n FROM a.revision"),
                home.count("SELECT count(*) AS n FROM a.blob"),
                home.count("SELECT sum(length) AS n FROM a.extent"),
            ],
            custody
        );
        assert_eq!(home.count("SELECT parsed_upto AS n FROM parse_state"), 0);
        assert_eq!(
            home.count("SELECT sum(unreadable) AS n FROM parse_state"),
            1
        );
        assert_eq!(home.count("SELECT sum(records) AS n FROM parse_state"), 1);
        assert_eq!(
            home.count("SELECT count(*) AS n FROM parse_state WHERE checkpoint IS NOT NULL"),
            0
        );
        assert_eq!(
            home.count("SELECT count(*) AS n FROM wire_reading WHERE json_extract(state,'$.refused_export')=1 AND json_extract(state,'$.export_cursor') IS NULL AND json_type(state,'$.progress.refused')='object' AND json_extract(state,'$.last.footprint.source.kind')='container_header'"),
            1
        );
    }
}

struct ShortSuccess {
    rewind: bool,
}
impl ExportParser for ShortSuccess {
    fn scan(
        &mut self,
        _: &str,
        reader: &mut dyn export::Input,
        sink: &mut dyn export::ItemSink,
    ) -> Result<export::ScanStats, export::ScanError> {
        if self.rewind {
            std::io::copy(&mut *reader, &mut std::io::sink())
                .map_err(|error| export::ScanError::Input(serde_json::Error::io(error)))?;
            reader
                .rewind()
                .map_err(|error| export::ScanError::Input(serde_json::Error::io(error)))?;
        }
        let mut byte = [0];
        std::io::Read::read_exact(reader, &mut byte)
            .map_err(|error| export::ScanError::Input(serde_json::Error::io(error)))?;
        // A successful callback and staged item cannot certify the unscanned suffix.
        sink.push(0..1, Vec::new(), None, b"{}")
            .map_err(export::ScanError::Callback)?;
        Ok(export::ScanStats {
            elements: 1,
            peak_element_bytes: 2,
        })
    }

    fn item(
        &mut self,
        _: &[u8],
        _: &mut polylogue_sources::document::Sink<'_>,
        _: &mut dyn export::ExportState,
    ) -> Result<(), String> {
        panic!("items from an incomplete scan must not be delivered")
    }
}

#[test]
fn successful_short_scans_cannot_complete_even_after_a_full_rewound_pass() {
    for rewind in [false, true] {
        let home = ClaudeHome::new();
        let raw = source();
        let (engine, _lock, plan) = plan(&home, &raw);
        let mut steps = Vec::new();
        let read = parse_export(&plan, &mut ShortSuccess { rewind }, &engine.blobs, |item| {
            let (mut step, cursor) =
                item.into_parse(&plan, &engine.blobs, &engine.stopped_attempt)?;
            step.prepare();
            steps.push(step);
            Ok(cursor)
        })
        .unwrap();
        assert!(!read.complete);
        assert_eq!(read.bytes_read, u64::from(rewind) * raw.len() as u64 + 1);
        assert_eq!(steps.len(), 1);
        assert_refusal(&steps[0]);
    }
}

#[test]
fn trailing_whitespace_is_consumed_before_complete() {
    let home = ClaudeHome::new();
    let mut raw = source();
    raw.extend_from_slice(b" \n\t");
    let (engine, _lock, plan) = plan(&home, &raw);
    let steps = collect(&engine, &plan);
    assert!(matches!(steps.last().unwrap().delivery.unwrap().source,
        ExportDeliverySource::Complete { length, .. } if length == raw.len() as u64));
}

#[cfg(feature = "faults")]
#[test]
fn changed_executable_extends_the_committed_export_prefix_without_recounting() {
    use polylogue_store::fault::{self, Action};
    let home = ClaudeHome::new();
    let path = home.dir.path().join("conversations.json");
    let raw = source();
    fs::write(&path, &raw).unwrap();
    let old_message = polylogue_model::MessageId::parse("chatgpt:old-message").unwrap();
    let old_admission = {
        let (mut engine, _lock) = engine(&home, 16);
        fault::clear();
        fault::arm("parse", &path.to_string_lossy(), Action::Fail, 1, 1);
        let report = engine.import(std::slice::from_ref(&path)).unwrap();
        fault::clear();
        assert!(report.refused.is_empty(), "{report:?}");
        assert_eq!(report.pass.errors.len(), 1, "{report:?}");
        assert!(
            report.pass.errors[0].contains("injected fault at parse"),
            "{report:?}"
        );
        assert_eq!(texts(&home, "chatgpt:old"), ["old body"]);
        assert!(texts(&home, "chatgpt:new").is_empty());
        assert_eq!(home.count("SELECT sum(records) AS n FROM parse_state"), 2);
        assert_eq!(home.count("SELECT parsed_upto AS n FROM parse_state"), 0);
        engine
            .reads
            .reader()
            .unwrap()
            .message_admission(&old_message)
            .unwrap()
            .unwrap()
    };
    fs::remove_file(&path).unwrap();
    let (mut engine, _lock) = engine(&home, 16);
    let unchanged = engine.run_pass().unwrap();
    assert!(unchanged.errors.is_empty(), "{unchanged:?}");
    assert_eq!(unchanged.bytes_read, 0, "{unchanged:?}");
    assert_eq!(texts(&home, "chatgpt:old"), ["old body"]);
    assert!(texts(&home, "chatgpt:new").is_empty());
    assert_eq!(home.count("SELECT sum(records) AS n FROM parse_state"), 2);
    let config = engine.config.clone();
    let archive = engine.archive.clone();
    let mut attempt = engine.stopped_attempt;
    attempt.executable = polylogue_model::BlobHash::of(b"changed export reader executable");
    drop(engine);
    let mut engine = Engine::new(config, archive, attempt).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pass.bytes_stored, 0, "{pass:?}");
    assert_eq!(texts(&home, "chatgpt:old"), ["old body"]);
    assert_eq!(texts(&home, "chatgpt:new"), ["new body"]);
    assert_eq!(home.count("SELECT sum(records) AS n FROM parse_state"), 4);
    assert_eq!(
        home.count("SELECT parsed_upto AS n FROM parse_state"),
        raw.len() as i64
    );
    assert_eq!(home.count("SELECT count(*) AS n FROM a.extent"), 1);
    assert_eq!(
        engine
            .reads
            .reader()
            .unwrap()
            .message_admission(&old_message)
            .unwrap(),
        Some(old_admission)
    );
    assert!(engine.run_pass().unwrap().errors.is_empty());
    assert_eq!(home.count("SELECT sum(records) AS n FROM parse_state"), 4);
}
