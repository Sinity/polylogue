use super::*;
use polylogue_sources::origin::{Arrival, Capture, ParserModule, Root};
use polylogue_sources::parse::{Parse, ParseError, Step, StepParser};
use polylogue_testkit::{ClaudeHome, claude_user};
use std::sync::{
    Barrier,
    atomic::{AtomicUsize, Ordering},
};

static TOGETHER: Barrier = Barrier::new(2);
static STARTED: AtomicUsize = AtomicUsize::new(0);
static PARSED: Mutex<Option<Sender<()>>> = Mutex::new(None);

struct Together(Box<dyn StepParser>);
impl StepParser for Together {
    fn step(
        &self,
        state: Option<&polylogue_model::Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Step, polylogue_model::Json), ParseError> {
        if STARTED.fetch_add(1, Ordering::SeqCst) < 2 {
            TOGETHER.wait();
        }
        let result = self.0.step(state, bytes, base);
        PARSED.lock().unwrap().as_ref().unwrap().send(()).unwrap();
        result
    }
}

const fn origin(open: fn(&Path) -> Option<Parse>) -> OriginSpec {
    OriginSpec {
        modules: &[ParserModule {
            name: "parallel-test",
            version: 1,
        }],
        name: "claude-code",
        harness_prefixes: &[],
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: |_, _| true,
        open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    }
}

fn lines(path: &Path) -> Option<Box<dyn StepParser>> {
    match (polylogue_sources::origin::by_name("claude-code")
        .unwrap()
        .open)(path)?
    {
        Parse::Lines(parser) => Some(parser),
        _ => None,
    }
}

static ORIGIN: OriginSpec = origin(|path| Some(Parse::Lines(Box::new(Together(lines(path)?)))));

fn engine(w: &ClaudeHome) -> (Engine, polylogue_store::ArchiveLock) {
    let (archive, _lock) = Archive::open_or_create(&w.root).unwrap();
    let config = Config {
        home: w.home.clone(),
        tool_dirs: Default::default(),
        workers: 2,
        reader_memory_bytes: 128 << 20,
        bytes_in_flight: 1 << 20,
        machine: "test-machine".into(),
        host: "test".into(),
        chunk: 64 << 20,
        rescan_every: Duration::from_secs(60),
        live_every: Duration::from_secs(2),
    };
    let engine = Engine::new(config, archive).unwrap();
    (engine, _lock)
}

#[test]
fn chains_parse_together_and_continue_while_the_writer_is_blocked() {
    STARTED.store(0, Ordering::SeqCst);
    let w = ClaudeHome::new();
    for name in ["a", "b", "c"] {
        w.append(
            &format!("{name}.jsonl"),
            &claude_user(name, name, "stored text"),
        );
    }
    let (mut engine, _lock) = engine(&w);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let behind = engine
        .reads
        .reader()
        .unwrap()
        .chains_behind_index(&[("claude-code".into(), "before".into())])
        .unwrap();
    let watched: HashMap<_, _> = engine
        .inputs
        .iter()
        .map(|(path, state)| (state.input_id.clone(), path.clone()))
        .collect();
    let plans = behind
        .into_iter()
        .map(|chain| {
            Plan::stored(
                watched[&chain.input].clone(),
                &ORIGIN,
                InputKey {
                    id: chain.input,
                    format: "claude-code".into(),
                    arrival_rank: 0,
                },
                Change::Reparse {
                    chain: chain.progress.chain,
                },
                None,
                chain.progress.length,
                0,
                pool::Bytes::Unloaded {
                    chain: chain.progress.chain,
                    length: chain.progress.length,
                    decoded: true,
                },
                64 << 20,
                false,
            )
        })
        .collect();
    let (parsed, received) = channel();
    *PARSED.lock().unwrap() = Some(parsed);
    let (locked, ready) = channel();
    let index = w.root.join("index.db");
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut db = rusqlite::Connection::open(index).unwrap();
            let tx = db
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                .unwrap();
            locked.send(()).unwrap();
            // Two workers must finish all three chains before any commit can proceed.
            for _ in 0..3 {
                received.recv().unwrap();
            }
            tx.commit().unwrap();
        });
        ready.recv().unwrap();
        let mut pass = Pass::default();
        engine.replace_readings(plans, &mut pass).unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert_eq!(pass.messages_written, 3);
    });
    *PARSED.lock().unwrap() = None;
    for name in ["a", "b", "c"] {
        assert_eq!(w.texts(name), ["stored text"]);
    }
    w.append("a.jsonl", &claude_user("later", "a", "later text"));
    let live = engine
        .live_plans(&HashSet::new(), &mut Pass::default())
        .unwrap();
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].path, w.file("a.jsonl"));
}

#[test]
fn a_nested_live_scan_does_not_reschedule_a_reading_in_progress() {
    let w = ClaudeHome::new();
    w.append("live.jsonl", &claude_user("one", "live", "first"));
    let (mut engine, _lock) = engine(&w);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let path = w.file("live.jsonl");
    engine.reparsing.insert(path.clone(), 1);
    w.append("live.jsonl", &claude_user("two", "live", "second"));
    assert!(
        engine
            .live_plans(&HashSet::new(), &mut Pass::default())
            .unwrap()
            .is_empty()
    );
    engine.reparsing.remove(&path);
    let live = engine
        .live_plans(&HashSet::new(), &mut Pass::default())
        .unwrap();
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].path, path);
}

static OLD_READING: Mutex<Option<Sender<()>>> = Mutex::new(None);
static LIVE_APPEND: Mutex<Option<Sender<()>>> = Mutex::new(None);
static RESUME_SECOND: Mutex<Option<Receiver<()>>> = Mutex::new(None);

struct LiveParser {
    parser: Box<dyn StepParser>,
    second: bool,
}

impl StepParser for LiveParser {
    fn step(
        &self,
        state: Option<&polylogue_model::Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Step, polylogue_model::Json), ParseError> {
        if self.second {
            let resume = RESUME_SECOND.lock().unwrap().take().unwrap();
            resume.recv().unwrap();
        }
        let result = self.parser.step(state, bytes, base);
        if !self.second {
            let new = String::from_utf8_lossy(bytes).contains("\"later\"");
            let signal = if new { &LIVE_APPEND } else { &OLD_READING };
            signal.lock().unwrap().as_ref().unwrap().send(()).unwrap();
        }
        result
    }
}

static LIVE_ORIGIN: OriginSpec = origin(|path| {
    Some(Parse::Lines(Box::new(LiveParser {
        parser: lines(path)?,
        second: path.file_name().is_some_and(|name| name == "b.jsonl"),
    })))
});

#[test]
fn a_committed_live_path_can_append_while_another_chain_is_still_parsing() {
    let w = ClaudeHome::new();
    for name in ["a", "b"] {
        w.append(&format!("{name}.jsonl"), &claude_user(name, name, "first"));
    }
    let (mut engine, _lock) = engine(&w);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let behind = engine
        .reads
        .reader()
        .unwrap()
        .chains_behind_index(&[("claude-code".into(), "before".into())])
        .unwrap();
    let watched: HashMap<_, _> = engine
        .inputs
        .iter_mut()
        .map(|(path, state)| {
            state.origin = &LIVE_ORIGIN;
            (state.input_id.clone(), (path.clone(), state.facts))
        })
        .collect();
    let plans = behind
        .into_iter()
        .map(|chain| {
            let (path, facts) = &watched[&chain.input];
            Plan::stored(
                path.clone(),
                &LIVE_ORIGIN,
                InputKey {
                    id: chain.input,
                    format: "claude-code".into(),
                    arrival_rank: 0,
                },
                Change::Reparse {
                    chain: chain.progress.chain,
                },
                *facts,
                chain.progress.length,
                0,
                pool::Bytes::Unloaded {
                    chain: chain.progress.chain,
                    length: chain.progress.length,
                    decoded: true,
                },
                64 << 20,
                true,
            )
        })
        .collect();
    let (old, old_ready) = channel();
    let (new, new_ready) = channel();
    let (resume, resumed) = channel();
    *OLD_READING.lock().unwrap() = Some(old);
    *LIVE_APPEND.lock().unwrap() = Some(new);
    *RESUME_SECOND.lock().unwrap() = Some(resumed);
    std::thread::scope(|scope| {
        let w = &w;
        scope.spawn(move || {
            old_ready.recv().unwrap();
            w.append("a.jsonl", &claude_user("later", "a", "later text"));
            new_ready.recv().unwrap();
            resume.send(()).unwrap();
        });
        let mut pass = Pass::default();
        engine.replace_readings(plans, &mut pass).unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
    });
    *OLD_READING.lock().unwrap() = None;
    *LIVE_APPEND.lock().unwrap() = None;
    assert_eq!(w.texts("a"), ["first", "later text"]);
    assert_eq!(w.texts("b"), ["first"]);
}
