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

static ORIGIN: OriginSpec = OriginSpec {
    modules: &[ParserModule {
        name: "parallel-test",
        version: 1,
    }],
    name: "claude-code",
    support: Support::Parsed,
    arrival: Arrival::Live,
    root: Root::HOME,
    locations: &[],
    capture: Capture::Bytes,
    recognize: |_, _| true,
    open: |path| match (polylogue_sources::origin::by_name("claude-code")
        .unwrap()
        .open)(path)?
    {
        Parse::Lines(parser) => Some(Parse::Lines(Box::new(Together(parser)))),
        _ => None,
    },
    resume: None,
    bare_uuid_ids: false,
    quota: None,
    agent_env: &[],
    web: None,
    assets: None,
    collector: None,
};

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
    let mut engine = Engine::new(config, archive).unwrap();
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
}
