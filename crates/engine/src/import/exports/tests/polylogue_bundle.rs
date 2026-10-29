use super::*;

fn authored_bundle() -> Vec<u8> {
    // Independent wire literals: the manifest's claimed operator is unrelated
    // to the current program accepting this historical setting.
    let manifest = json!({"record":"manifest","data":{
        "format":"polylogue-bundle","version":1,"query":format!("query:{}","0".repeat(64)),"query_text":"synthetic",
        "ran_at":"2026-01-01T00:00:00Z","read_at":{"generation":"synthetic","index":100,"archive":100},"caught_up":true,
        "counts":{"sessions":1,"messages":0,"documents":0,"events":0,"links":0,"authored":1,"variants":0,"variant_parts":0,"blobs":0},
        "flags":{"format":"jsonl","view":"full","with_blobs":false,"inherited":false,"assertions":true},"enabled_variants":[],"author":"operator"}});
    let session = json!({"record":"session","data":{"ref":"session:claude-code:bundle-session","commit":1,"session":{"id":"claude-code:bundle-session","kind":"main","cwd":[]}}});
    let authored = json!({"record":"authored","data":{"ref":"request:setting-request","commit":2,"prepared":{
        "request":{"id":"setting-request","machine":"exported-machine","at":"2026-01-01T00:00:00Z","write":{"write":"set_setting","key":"day.zone","value":"UTC","author":"operator"}},
        "outcome":{"id":"setting-record","repeat":false,"pending":false},
        "effect":{"variants":{"variants":[],"variant_parts":[]},"notes":[],"withdrawals":[],"settings":[{"id":"setting-record","machine":"*","key":"day.zone","value":"UTC","author":"operator","set_at":"2026-01-01T00:00:00Z"}],"kept_results":[],"deliveries":[],"chats":[],"backfills":[]}},"imports":[]}});
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
fn canonical_bundle_restores_historical_state_with_separate_local_ingress() {
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
    assert_eq!(w.count("SELECT count(*) AS n FROM a.setting WHERE id='setting-record' AND value='UTC' AND author='operator'"),1);
    let reads = engine.reads.reader().unwrap();
    let provenance = reads
        .query(
            "SELECT imports FROM source_client_request WHERE id='setting-request'",
            &[],
        )
        .unwrap();
    let imports: serde_json::Value =
        serde_json::from_str(provenance[0]["imports"].as_str().unwrap()).unwrap();
    assert_eq!(imports["observed"][0]["actor"], "script:polylogue-import");
    drop(reads);
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
    };
    let mut engine = Engine::new(config, archive).unwrap();
    polylogue_store::fault::arm(
        &point,
        "",
        polylogue_store::fault::Action::StopProcess,
        0,
        1,
    );
    engine.import(&[base.join("records.jsonl")]).unwrap();
    panic!("the configured custody boundary was not reached");
}

#[cfg(feature = "faults")]
#[test]
#[allow(unsafe_code)]
fn ordinary_open_replays_stored_authored_state_after_real_interruption() {
    for point in [
        "import.before_authored_restore",
        "import.after_authored_restore",
    ] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join("records.jsonl");
        fs::write(&path, authored_bundle()).unwrap();
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
        fs::remove_file(&path).unwrap();
        let (mut engine, _lock) = engine(&w, 64);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert_eq!(w.count("SELECT count(*) AS n FROM a.client_request"), 1);
        assert_eq!(
            w.count(
                "SELECT count(*) AS n FROM a.setting WHERE id='setting-record' AND value='UTC'"
            ),
            1
        );
        let reader = engine.reads.reader().unwrap();
        let provenance = reader
            .query(
                "SELECT imports FROM source_client_request WHERE id='setting-request'",
                &[],
            )
            .unwrap();
        let imports: serde_json::Value =
            serde_json::from_str(provenance[0]["imports"].as_str().unwrap()).unwrap();
        assert_eq!(imports["observed"].as_array().unwrap().len(), 1);
        assert_eq!(imports["observed"][0]["actor"], "script:polylogue-import");
    }
}
