use super::*;
use polylogue_sources::origin::{Arrival, Capture, ParserModule, Root};
use polylogue_sources::parse::{Parse, ParseError, Step, StepParser};
use polylogue_testkit::{ClaudeHome, claude_user};
use std::sync::mpsc::Sender;
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
        finished: bool,
    ) -> Result<(Step, polylogue_model::Json), ParseError> {
        if STARTED.fetch_add(1, Ordering::SeqCst) < 2 {
            TOGETHER.wait();
        }
        let result = self.0.step(state, bytes, base, finished);
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
        import: polylogue_sources::origin::Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: polylogue_model::Route::Live,
        input_identity: polylogue_sources::InputIdentity::Location,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
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
        clock: polylogue_model::WallClock::fixed("2026-10-07T10:00:00Z".parse().unwrap()),
        home: w.home.clone(),
        tool_dirs: Default::default(),
        workers: 3,
        memory_budget: Default::default(),
        reader_memory_bytes: 128 << 20,
        bytes_in_flight: 1 << 20,
        machine: "test-machine".into(),
        host: "test".into(),
        chunk: 64 << 20,
        live_every: Duration::from_secs(2),
        reads_sources: true,
    };
    let engine = Engine::new(config, archive, polylogue_testkit::stopped_attempt()).unwrap();
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
        .chains_behind_index(
            &[("claude-code".into(), "before".into())],
            polylogue_store::StoppedChains::Finished,
            polylogue_store::RevisionChains::All,
            None,
        )
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
                    route: polylogue_model::Route::Live,
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
        engine
            .execute_pool(Vec::new(), plans, &[], &mut pass)
            .unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert_eq!(STARTED.load(Ordering::SeqCst), 3);
        assert_eq!(
            pass.messages_written, 0,
            "the same stored rows are unchanged"
        );
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
        finished: bool,
    ) -> Result<(Step, polylogue_model::Json), ParseError> {
        if self.second {
            let resume = RESUME_SECOND.lock().unwrap().take().unwrap();
            resume.recv().unwrap();
        }
        let result = self.parser.step(state, bytes, base, finished);
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
        .chains_behind_index(
            &[("claude-code".into(), "before".into())],
            polylogue_store::StoppedChains::Finished,
            polylogue_store::RevisionChains::All,
            None,
        )
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
                    route: polylogue_model::Route::Live,
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
    let (result, pass, appended) = std::thread::scope(|scope| {
        let w = &w;
        let append = scope.spawn(move || {
            let appended = old_ready
                .recv_timeout(Duration::from_secs(10))
                .and_then(|()| {
                    w.append("a.jsonl", &claude_user("later", "a", "later text"));
                    new_ready.recv_timeout(Duration::from_secs(10))
                });
            // Release the other parser even when live work failed, so the
            // production pass can return its actual errors to the annotation.
            let _ = resume.send(());
            appended
        });
        let mut pass = Pass::default();
        let result = engine.execute_pool(Vec::new(), plans, &[], &mut pass);
        (result, pass, append.join().unwrap())
    });
    assert!(result.is_ok(), "replacement result: {result:?}; {pass:?}");
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(
        appended.is_ok(),
        "live append signal: {appended:?}; {pass:?}"
    );
    *OLD_READING.lock().unwrap() = None;
    *LIVE_APPEND.lock().unwrap() = None;
    assert_eq!(w.texts("a"), ["first", "later text"]);
    assert_eq!(w.texts("b"), ["first"]);
}

struct PausedChain(Box<dyn StepParser>);

struct ChainPause {
    started: Sender<()>,
    resume: Arc<(Mutex<bool>, std::sync::Condvar)>,
}

static CHAIN_PAUSE: Mutex<Option<ChainPause>> = Mutex::new(None);

impl StepParser for PausedChain {
    fn step(
        &self,
        state: Option<&polylogue_model::Json>,
        bytes: &[u8],
        base: u64,
        finished: bool,
    ) -> Result<(Step, polylogue_model::Json), ParseError> {
        let (started, resume) = {
            let pause = CHAIN_PAUSE.lock().unwrap();
            let pause = pause.as_ref().unwrap();
            (pause.started.clone(), Arc::clone(&pause.resume))
        };
        let _ = started.send(());
        let (resumed, changed) = &*resume;
        let mut resumed = resumed.lock().unwrap();
        while !*resumed {
            resumed = changed.wait(resumed).unwrap();
        }
        self.0.step(state, bytes, base, finished)
    }
}

static PAUSED_ORIGIN: OriginSpec =
    origin(|path| Some(Parse::Lines(Box::new(PausedChain(lines(path)?)))));

#[test]
fn live_append_commits_before_waiting_chain_replacements() {
    let w = ClaudeHome::new();
    for name in ["chain-a", "chain-b", "live"] {
        w.append(&format!("{name}.jsonl"), &claude_user(name, name, "first"));
    }
    let (mut engine, _lock) = engine(&w);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    engine.config.workers = 2;
    engine.config.live_every = Duration::ZERO;
    let behind = engine
        .reads
        .reader()
        .unwrap()
        .chains_behind_index(
            &[("claude-code".into(), "before".into())],
            polylogue_store::StoppedChains::Finished,
            polylogue_store::RevisionChains::All,
            None,
        )
        .unwrap();
    let watched: HashMap<_, _> = engine
        .inputs
        .iter_mut()
        .filter(|(path, _)| *path != &w.file("live.jsonl"))
        .map(|(path, state)| {
            state.origin = &PAUSED_ORIGIN;
            (state.input_id.clone(), (path.clone(), state.facts))
        })
        .collect();
    let chains = behind
        .into_iter()
        .filter_map(|chain| {
            let (path, facts) = watched.get(&chain.input)?;
            Some(Plan::stored(
                path.clone(),
                &PAUSED_ORIGIN,
                InputKey {
                    id: chain.input,
                    format: "claude-code".into(),
                    route: polylogue_model::Route::Live,
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
            ))
        })
        .collect();
    let (started, entered) = channel();
    let resume = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    *CHAIN_PAUSE.lock().unwrap() = Some(ChainPause {
        started,
        resume: Arc::clone(&resume),
    });
    let commits = engine.writer.completion_commits();
    let archive = engine.archive.clone();
    let live_input = engine.location_id(&w.file("live.jsonl"));
    let (result, pass, live_committed) = std::thread::scope(|scope| {
        let w = &w;
        let observed = scope.spawn(move || {
            let committed = (|| {
                entered.recv_timeout(Duration::from_secs(10)).ok()?;
                w.append("live.jsonl", &claude_user("later", "live", "live append"));
                let length = std::fs::metadata(w.file("live.jsonl")).unwrap().len();
                let reads = ReadPool::open(&archive).unwrap();
                loop {
                    commits.recv_timeout(Duration::from_secs(10)).ok()?;
                    if reads
                        .reader()
                        .unwrap()
                        .input_progress(&live_input)
                        .unwrap()
                        .is_some_and(|progress| progress.parsed_upto == length)
                    {
                        return Some(());
                    }
                }
            })()
            .is_some();
            let (resumed, changed) = &*resume;
            *resumed.lock().unwrap() = true;
            changed.notify_all();
            committed
        });
        let mut pass = Pass::default();
        let result = engine.execute_pool(Vec::new(), chains, &[], &mut pass);
        (result, pass, observed.join().unwrap())
    });
    *CHAIN_PAUSE.lock().unwrap() = None;
    assert!(result.is_ok(), "replacement result: {result:?}; {pass:?}");
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(
        live_committed,
        "the live append commits while both replacement parsers are held"
    );
    assert_eq!(w.texts("live"), ["first", "live append"]);
    assert_eq!(w.texts("chain-a"), ["first"]);
    assert_eq!(w.texts("chain-b"), ["first"]);
}

#[cfg(feature = "faults")]
#[test]
fn live_append_commits_while_an_oversized_chain_waits_for_its_reply() {
    use polylogue_store::fault::{self, Action};

    let w = ClaudeHome::new();
    w.append(
        "chain.jsonl",
        &claude_user("old", "chain", &"x".repeat(4096)),
    );
    w.append("live.jsonl", &claude_user("first", "live", "first"));
    let (mut engine, _lock) = engine(&w);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let path = w.file("chain.jsonl");
    let state = &engine.inputs[&path];
    let input = InputKey {
        id: state.input_id.clone(),
        format: "claude-code".into(),
        route: polylogue_model::Route::Live,
    };
    let progress = engine
        .reads
        .reader()
        .unwrap()
        .input_progress(&input.id)
        .unwrap()
        .unwrap();
    let budget = 1024;
    assert!(progress.length > budget);
    let plan = Plan::stored(
        path,
        state.origin,
        input.clone(),
        Change::Reparse {
            chain: progress.chain,
        },
        state.facts,
        progress.length,
        0,
        pool::Bytes::Unloaded {
            chain: progress.chain,
            length: progress.length,
            decoded: true,
        },
        1024,
        false,
    );
    let (done, finished) = channel();
    let pool = pool::Pool::start(
        2,
        budget,
        engine.writer.sender(),
        Arc::clone(&engine.blobs),
        engine.archive.clone(),
        done,
        pool::ChainSettings {
            stopped_attempt: engine.stopped_attempt,
            read_memory: engine.config.reader_memory_bytes,
            yielding: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        },
    );
    fault::arm("pool.chain_reply", &input.id, Action::Hold, 0, 1);
    assert!(pool.submit_chain(1, plan));
    let until = Instant::now() + Duration::from_secs(10);
    while !fault::held() && Instant::now() < until {
        std::thread::yield_now();
    }
    let paused = fault::held();
    let commits = engine.writer.completion_commits();
    w.append("live.jsonl", &claude_user("later", "live", "live append"));
    let mut live = engine
        .live_plans(&HashSet::new(), &mut Pass::default())
        .unwrap();
    assert_eq!(live.len(), 1);
    let mut live = live.pop().unwrap();
    let id = live.input.id.clone();
    let length = live.size;
    assert!(engine.load_stored(&mut live).unwrap());
    let job = live.next_job(&engine.stopped_attempt);
    assert!(job.end - job.start <= budget);
    let room = pool.has_room(job.end - job.start);
    let committed = if paused && room {
        assert!(pool.submit(job));
        loop {
            if engine
                .reads
                .reader()
                .unwrap()
                .input_progress(&id)
                .unwrap()
                .is_some_and(|progress| progress.parsed_upto == length)
            {
                break true;
            }
            if commits.recv_timeout(Duration::from_secs(10)).is_err() {
                break false;
            }
        }
    } else {
        false
    };
    fault::release();
    drop(pool);
    fault::clear();
    for completed in finished {
        match completed {
            pool::Finished::Chain(done) => assert!(done.written.is_ok(), "{:?}", done.written),
            pool::Finished::Range(done) => assert!(done.error.is_none(), "{:?}", done.error),
            pool::Finished::Captured(_) => panic!("no rollout was captured"),
        }
    }
    assert!(paused, "the chain reaches the paused reply");
    assert!(
        room,
        "a staged chain waiting for its reply leaves room for a live range"
    );
    assert!(
        committed,
        "the live range commits before the chain reply resumes"
    );
    assert_eq!(w.texts("live"), ["first", "live append"]);
}
