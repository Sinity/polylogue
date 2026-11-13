use super::*;
use polylogue_model::Json;
use polylogue_sources::{Arrival, Capture, OriginSpec, Parse, Root, Step, StepParser};

fn source_rows(engine: &Engine, input: &str) -> Vec<Json> {
    engine.reads.reader().unwrap().query(
        "SELECT 'revision' AS row,number AS n,chain,length,produced AS detail FROM a.revision WHERE input=?1
         UNION ALL SELECT 'extent',offset,chain,length,hex(blob) FROM a.extent WHERE input=?1 ORDER BY row,n",
        &[polylogue_store::SqlValue::Text(input.to_owned())],
    ).unwrap()
}

#[test]
fn a_completed_scope_reset_reloads_confirmed_state_without_recapturing() {
    let home = ClaudeHome::new();
    let text = claude_user("one", "scope", "original confirmed body");
    home.append("scope.jsonl", &text);
    let path = home.file("scope.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let input = engine.inputs[&path].input_id.clone();
    let chain = engine.inputs[&path].chain.unwrap();
    assert!(engine.inputs[&path].parser_state.is_some());
    let captured = source_rows(&engine, &input);
    let checkpoint = engine
        .reads
        .reader()
        .unwrap()
        .parser_checkpoint(
            &input,
            chain,
            &engine.inputs[&path].origin.reading_version(),
        )
        .unwrap();
    let mut admitted_state = checkpoint
        .as_ref()
        .and_then(|stored| stored.checkpoint.as_ref())
        .and_then(|checkpoint| checkpoint.parser_state.as_ref())
        .expect("Claude continuation is admitted with its cursor")
        .clone();
    // The kept state leaves out the keyed context's changes, which the
    // index holds (docs/architecture.md, pipeline).
    admitted_state
        .as_object_mut()
        .unwrap()
        .remove(polylogue_model::parser_context::FIELD);
    assert_eq!(
        engine.inputs[&path].parser_state.as_ref(),
        Some(&admitted_state)
    );
    // Fault only the process-local cache, leaving the admitted continuation
    // untouched. Exact equality after reset distinguishes restoration from reuse.
    engine
        .inputs
        .get_mut(&path)
        .unwrap()
        .parser_state
        .as_mut()
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert(
            "_scope_test_stale_cache".into(),
            Json::String("previous scope".into()),
        );
    assert_ne!(
        engine.inputs[&path].parser_state.as_ref(),
        Some(&admitted_state)
    );
    engine.archive.advance_generation_for_test().unwrap();
    let (resumed, _) = engine.one_pass(false, &Default::default()).unwrap();
    assert!(resumed.errors.is_empty(), "{resumed:?}");
    assert_eq!(resumed.full_scans, 1, "the new scope repeats discovery");
    assert_eq!((resumed.bytes_read, resumed.bytes_stored), (0, 0));
    assert_eq!(engine.inputs[&path].input_id, input);
    assert_eq!(engine.inputs[&path].chain, Some(chain));
    assert_eq!(engine.inputs[&path].parsed_upto, text.len() as u64);
    assert_eq!(
        engine.inputs[&path].parser_state.as_ref(),
        Some(&admitted_state),
        "the new scope restores admitted state instead of reusing the old cache"
    );
    assert_eq!(source_rows(&engine, &input), captured);
    assert_eq!(
        engine
            .reads
            .reader()
            .unwrap()
            .parser_checkpoint(
                &input,
                chain,
                &engine.inputs[&path].origin.reading_version()
            )
            .unwrap(),
        checkpoint
    );
}

#[test]
fn a_cached_scope_cannot_be_rebound_after_its_entry_check() {
    let home = ClaudeHome::new();
    home.append(
        "scope-race.jsonl",
        &claude_user("one", "scope-race", "admitted before the reset"),
    );
    let path = home.file("scope-race.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let input = engine.inputs[&path].input_id.clone();
    let captured = source_rows(&engine, &input);
    // This is the valid entry check, followed by a complete reset before
    // the shared load/turn token-acquisition seam runs.
    engine.read_scope.as_ref().unwrap().check().unwrap();
    engine.archive.advance_generation_for_test().unwrap();
    engine.archive.scope_token().unwrap().check().unwrap();
    assert!(matches!(
        engine.cached_scope(),
        Err(EngineError::Store(
            polylogue_store::StoreError::ScopeChanged
        ))
    ));
    assert!(matches!(
        engine.read_scope.as_ref().unwrap().check(),
        Err(polylogue_store::StoreError::ScopeChanged)
    ));
    let resumed = engine.run_pass().unwrap();
    assert!(resumed.errors.is_empty(), "{resumed:?}");
    assert_eq!((resumed.bytes_read, resumed.bytes_stored), (0, 0));
    assert_eq!(source_rows(&engine, &input), captured);
    assert_eq!(engine.inputs[&path].input_id, input);
    engine.read_scope.as_ref().unwrap().check().unwrap();
}

struct Control {
    started: Sender<()>,
    release: Receiver<()>,
}
static CONTROL: Mutex<Option<Control>> = Mutex::new(None);
struct Held(Box<dyn StepParser>);
impl StepParser for Held {
    fn step(
        &self,
        state: Option<&Json>,
        bytes: &[u8],
        base: u64,
        finished: bool,
        context: &polylogue_model::parser_context::Handle,
    ) -> Result<(Step, Json), polylogue_sources::ParseError> {
        if let Some(control) = CONTROL.lock().unwrap().take() {
            control.started.send(()).unwrap();
            control.release.recv().unwrap();
        }
        self.0.step(state, bytes, base, finished, context)
    }
}
fn held(path: &Path) -> Option<Parse> {
    let Parse::Lines(parser) = (polylogue_sources::origin::by_name("claude-code")
        .unwrap()
        .open)(path)?
    else {
        panic!("Claude line parser")
    };
    Some(Parse::Lines(Box::new(Held(parser))))
}
fn held_origin() -> &'static OriginSpec {
    let native = polylogue_sources::origin::by_name("claude-code").unwrap();
    Box::leak(Box::new(OriginSpec {
        name: native.name,
        harness_prefixes: native.harness_prefixes,
        modules: native.modules,
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: polylogue_model::Route::Live,
        input_identity: polylogue_sources::InputIdentity::Location,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: native.recognize,
        open: held,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    }))
}

#[test]
fn a_scope_reset_during_preparation_cannot_complete_with_stale_state_or_capture_twice() {
    let home = ClaudeHome::new();
    let first = claude_user("one", "scope-session", "first confirmed body");
    home.append("scope-session.jsonl", &first);
    let path = home.file("scope-session.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.live_every = Duration::ZERO;
    assert!(engine.run_pass().unwrap().errors.is_empty());
    // Pause the actual Claude parser without changing its version or output.
    engine.inputs.get_mut(&path).unwrap().origin = held_origin();
    let input = engine.inputs[&path].input_id.clone();
    let chain = engine.inputs[&path].chain.unwrap();
    let second = claude_user("two", "scope-session", "prepared suffix");
    let expected = format!("{first}{second}");
    std::fs::write(&path, &expected).unwrap();
    let archive = engine.archive.clone();
    let (started, waiting) = channel();
    let (release, continue_reading) = channel();
    *CONTROL.lock().unwrap() = Some(Control {
        started,
        release: continue_reading,
    });
    let (mut engine, interrupted) = std::thread::scope(|threads| {
        let work = threads.spawn(move || {
            let (tx, rx) = channel();
            let status = Arc::new(Mutex::new(LoopStatus::default()));
            let result = engine.run(tx, rx, &status, Lifetime::UntilIdle, || {});
            (engine, result)
        });
        waiting
            .recv_timeout(Duration::from_secs(10))
            .expect("actual parser entered after acquiring the suffix");
        archive.advance_generation_for_test().unwrap();
        release.send(()).unwrap();
        work.join().unwrap()
    });
    let summary = interrupted.unwrap();
    assert!(
        summary.passes >= 2,
        "the interrupted turn must not establish idle: {summary:?}"
    );
    assert!(summary.work.errors.is_empty(), "{summary:?}");
    let captured = source_rows(&engine, &input);
    assert_eq!(
        captured
            .iter()
            .filter(|row| row["row"] == "revision")
            .count(),
        2
    );
    assert_eq!(
        captured.iter().filter(|row| row["row"] == "extent").count(),
        2
    );
    let (resumed, _) = engine.one_pass(false, &Default::default()).unwrap();
    assert!(resumed.errors.is_empty(), "{resumed:?}");
    assert_eq!((resumed.bytes_read, resumed.bytes_stored), (0, 0));
    assert_eq!(source_rows(&engine, &input), captured);
    assert_eq!(engine.inputs[&path].input_id, input);
    assert_eq!(engine.inputs[&path].chain, Some(chain));
    assert_eq!(engine.inputs[&path].parsed_upto, expected.len() as u64);
    let reader = engine.reads.reader().unwrap();
    let stored = reader
        .stored_chain(&input, chain, expected.len() as u64)
        .unwrap();
    assert_eq!(
        stored
            .read_range(&engine.blobs, 0, expected.len() as u64)
            .unwrap(),
        expected.as_bytes()
    );
    assert_eq!(reader.query("SELECT count(*) AS n FROM message m JOIN session s ON s.pk=m.session_pk WHERE s.id='claude-code:scope-session'", &[]).unwrap()[0]["n"], 2);
}
