use super::*;

fn authored_bundle() -> Vec<u8> {
    // Independent wire literals: the manifest's claimed operator is unrelated
    // to the current program accepting this historical setting.
    let manifest = json!({"record":"manifest","data":{
        "format":"polylogue-bundle","version":1,"query":format!("query:{}","0".repeat(64)),"query_text":"synthetic",
        "ran_at":"2026-01-01T00:00:00Z","read_at":{"generation":"synthetic","index":100,"archive":100},"caught_up":true,"currency":{"mode":"standalone"},
        "counts":{"sessions":1,"messages":0,"documents":0,"events":0,"links":0,"authored":1,"assertions":0,"text_fragments":0,"variants":0,"variant_parts":0,"blobs":0},
        "flags":{"format":"jsonl","with_blobs":false,"read":{"view":"transcript","inherited":false,"assertions":true}},"enabled_variants":[],"author":"operator"}});
    let session = json!({"record":"session","data":{"ref":"session:claude-code:bundle-session","commit":1,"session":{"id":"claude-code:bundle-session","kind":"main","cwd":[]}}});
    let authored = json!({"record":"authored","data":{"ref":"request:setting-request","commit":2,"prepared":{
        "request":{"id":"setting-request","machine":"exported-machine","at":"2026-01-01T00:00:00Z","authorship":{"author":"operator","via":"cli"},"write":{"write":"set_setting","key":"browser.actions","value":"on","machine":null}},
        "outcome":{"id":"setting-request","repeat":false,"pending":false},
        "effect":{"assertions":[{"id":"setting-request","root":"setting-request","kind":"setting","key":"browser.actions","value":{"machine":null,"value":{"type":"boolean","value":true}},"author":"operator","via":"cli","machine":"exported-machine","created_at":"2026-01-01T00:00:00Z","valid_until":null,"links":[]}],"variants":{"variants":[],"variant_parts":[]},"kept_results":[],"deliveries":[],"chats":[],"backfills":[]}}}});
    [manifest, session, authored]
        .into_iter()
        .flat_map(|record| {
            let mut raw = serde_json::to_vec(&record).unwrap();
            raw.push(b'\n');
            raw
        })
        .collect()
}

#[test]
fn canonical_bundle_keeps_historical_authorship_and_separate_import_observation() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("records.jsonl");
    fs::write(&path, authored_bundle()).unwrap();
    let (mut engine, _lock) = engine(&w, 64);
    let report = engine.import(std::slice::from_ref(&path)).unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert_eq!(report.packages[0].sessions, 1);
    assert_eq!(
        (report.user_state.applied, report.user_state.pending),
        (1, 0)
    );
    assert_eq!(w.count("SELECT count(*) AS n FROM a.client_request"), 1);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.assertion WHERE id='setting-request' AND kind='setting' AND key='browser.actions' AND json_extract(value,'$.value.value')=1 AND author='operator'"),1);
    assert_eq!(import_actors(&engine), ["script:polylogue-import"]);
    let again = engine.import(std::slice::from_ref(&path)).unwrap();
    assert!(again.pass.errors.is_empty(), "{again:?}");
    assert_eq!((again.user_state.applied, again.user_state.present), (0, 1));
    assert_eq!(w.count("SELECT count(*) AS n FROM a.client_request"), 1);
}

#[test]
fn malformed_canonical_suffix_and_retired_own_exports_are_refused_before_writes() {
    for bytes in [
        [
            authored_bundle(),
            b"{\"record\":\"session\",\"data\":false}\n".to_vec(),
        ]
        .concat(),
        br#"{"format":"polylogue-export","version":4,"sessions":[],"user_state":[]}"#.to_vec(),
    ] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join("input.jsonl");
        fs::write(&path, bytes).unwrap();
        let (mut engine, _lock) = engine(&w, 64);
        let report = engine.import(&[path]).unwrap();
        assert_eq!(report.refused.len(), 1, "{report:?}");
        assert_eq!(w.count("SELECT count(*) AS n FROM a.input"), 0);
        assert_eq!(w.count("SELECT count(*) AS n FROM a.client_request"), 0);
    }
}

#[cfg(feature = "faults")]
#[test]
fn authored_crash_child() {
    let Some(base) = std::env::var_os("POLYLOGUE_BUNDLE_CRASH_BASE") else {
        return;
    };
    let base = std::path::PathBuf::from(base);
    let point = std::env::var("POLYLOGUE_BUNDLE_CRASH_POINT").unwrap();
    let (archive, _lock) = Archive::open_or_create(&base.join("archive")).unwrap();
    let config = Config {
        clock: polylogue_model::WallClock::fixed("2026-01-01T00:00:00Z".parse().unwrap()),
        home: base.join("home"),
        tool_dirs: Default::default(),
        workers: 1,
        reader_memory_bytes: 8 << 20,
        bytes_in_flight: 1 << 20,
        machine: "test-machine".into(),
        host: "test".into(),
        chunk: 64,
        rescan_every: std::time::Duration::from_secs(60),
        live_every: std::time::Duration::from_secs(2),
        reads_sources: true,
    };
    let mut engine = Engine::new(config, archive).unwrap();
    polylogue_store::fault::arm(
        &point,
        "",
        polylogue_store::fault::Action::StopProcess,
        0,
        1,
    );
    let report = engine.import(&[base.join("records.jsonl")]).unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    panic!("the configured custody boundary was not reached");
}

#[cfg(feature = "faults")]
#[allow(unsafe_code)]
fn interrupt_authored_import(w: &ClaudeHome, point: &str) {
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "import::exports::tests::polylogue_bundle::authored_crash_child",
            "--nocapture",
        ])
        .env("POLYLOGUE_BUNDLE_CRASH_BASE", w.dir.path())
        .env("POLYLOGUE_BUNDLE_CRASH_POINT", point)
        .spawn()
        .unwrap();
    let mut status = 0;
    assert_eq!(
        unsafe { libc::waitpid(child.id() as libc::pid_t, &mut status, libc::WUNTRACED) },
        child.id() as libc::pid_t
    );
    assert!(
        libc::WIFSTOPPED(status),
        "child exited before {point}: {status}"
    );
    child.kill().unwrap();
    assert!(!child.wait().unwrap().success());
}

#[cfg(feature = "faults")]
#[test]
fn ordinary_open_replays_stored_authored_state_after_real_interruption() {
    for point in [
        "import.before_authored_restore",
        "import.after_authored_restore",
    ] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join("records.jsonl");
        fs::write(&path, authored_bundle()).unwrap();
        interrupt_authored_import(&w, point);
        fs::remove_file(&path).unwrap();
        let (mut engine, _lock) = engine(&w, 64);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert_eq!(w.count("SELECT count(*) AS n FROM a.client_request"), 1);
        assert_eq!(
            w.count(
                "SELECT count(*) AS n FROM a.assertion WHERE id='setting-request' AND kind='setting' AND json_extract(value,'$.value.value')=1"
            ),
            1
        );
        assert_eq!(import_actors(&engine), ["script:polylogue-import"]);
    }
}

#[cfg(feature = "faults")]
#[test]
fn explicit_receiver_after_interruption_does_not_rewrite_historical_authorship() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("records.jsonl");
    fs::write(&path, authored_bundle()).unwrap();
    interrupt_authored_import(&w, "import.after_authored_restore");
    let (mut engine, _lock) = engine(&w, 64);
    let report = engine
        .import_with_options(
            &[path],
            &ImportOptions {
                authorship: Some(polylogue_model::authorship::Authorship {
                    author: polylogue_model::Author::Script("receiver-b".into()),
                    via: polylogue_model::authorship::Channel::Import,
                }),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert_eq!(report.user_state.present, 1);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.client_request"), 1);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.assertion WHERE id='setting-request' AND kind='setting' AND author='operator' AND json_extract(value,'$.value.value')=1"), 1);
    assert_eq!(
        import_actors(&engine),
        ["script:polylogue-import", "script:receiver-b"]
    );
    assert!(engine.run_pass().unwrap().errors.is_empty());
    assert_eq!(w.count("SELECT count(*) AS n FROM a.client_request"), 1);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.assertion WHERE kind='setting'"),
        1
    );
}

fn import_actors(engine: &Engine) -> Vec<String> {
    engine.reads.reader().unwrap().query(
        "SELECT DISTINCT json_extract(details_json,'$.import.authorship.author') AS actor FROM a.observation WHERE class='imported' ORDER BY actor", &[]
    ).unwrap().into_iter().map(|row| row["actor"].as_str().unwrap().to_owned()).collect()
}
