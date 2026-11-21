//! Native binary transition over one seeded database, bus and original store.
use super::*;
use serde_json::{Value, json};
use sinex_model::EventPayload;
use std::process::Command;

fn reply(running: &Running, body: Value) -> Value {
    let started = Instant::now();
    loop {
        match real_sinex::request(&running.socket, "read", "/v1", body.clone()) {
            Ok(answer) => return answer,
            Err(error) => {
                assert!(
                    started.elapsed() < Duration::from_secs(20),
                    "Sinex API readiness: {error}"
                );
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

fn originals(running: &Running) -> Value {
    serde_json::from_str(&running.real_sinex.sql("SELECT json_build_object(
      'events',(SELECT json_agg(json_build_array(id,event_type,material_id,record_offset,payload) ORDER BY id)
         FROM core.events WHERE source='synthetic-evidence'),
      'materials',(SELECT json_agg(json_build_array(id,blob,length) ORDER BY id) FROM raw.materials
         WHERE id IN (SELECT material_id FROM core.events WHERE source='synthetic-evidence')))")).unwrap()
}

fn seed(running: &Running, frame: &str) {
    send_captured_events(
        running,
        frame,
        &[(
            sinex_model::event_types::machine::FileChanged::declaration().kind,
            json!({"received_at":"2026-03-01T10:00:01Z","change":"written",
            "path":format!("/home/op/fixture/{frame}.txt"),"path_hex":null,"to":null,
            "is_dir":false,"note":null}),
        )],
    );
}

fn broad_rows(running: &Running) -> Vec<Value> {
    let answer = reply(
        running,
        json!({"method":"query","query":"events | fields id, event_type, source, payload","page-size":1000}),
    );
    assert_eq!(answer["outcome"], "ok", "{answer}");
    answer["result"]["rows"]
        .as_array()
        .expect("broad event rows")
        .clone()
}

fn apply(binary: &Path, running: &Running) {
    assert!(
        Command::new(binary)
            .args([
                "admin",
                "schema",
                "--database",
                &running.real_sinex.pg.url("sinex", None)
            ])
            .status()
            .unwrap()
            .success(),
        "normal startup schema step"
    );
}

#[test]
#[ignore = "needs signed SINEXD and SINEXD_BEFORE binaries with their exact SINEXD_REVISION and SINEXD_BEFORE_REVISION"]
fn a_new_binary_recovers_originals_and_admission_after_the_old_binary_refuses_its_declaration() {
    let next = PathBuf::from(std::env::var_os("SINEXD").expect("new signed daemon"));
    let before = PathBuf::from(std::env::var_os("SINEXD_BEFORE").expect("old signed daemon"));
    for (binary, key) in [
        (&before, "SINEXD_BEFORE_REVISION"),
        (&next, "SINEXD_REVISION"),
    ] {
        let wanted = std::env::var(key).expect("exact signed runtime revision");
        let version = Command::new(binary).arg("--version").output().unwrap();
        assert!(version.status.success());
        let version = String::from_utf8(version.stdout).unwrap();
        assert!(
            version.contains(&wanted) && version.contains("dirty=false"),
            "{version}"
        );
        eprintln!("binary recovery runtime: {version}");
    }
    assert_eq!(sinex_protocol::PRODUCER_VERSION, 7);
    let root = tempfile::tempdir().unwrap();
    let real_sinex = real_sinex::RealSinex::start_using(
        root.path(),
        vec![serde_json::to_value(polylogue_sinex::kinds::source_declaration()).unwrap()],
        true,
        &before,
        true,
    );
    let mut running = Running {
        config: Config {
            mode: Mode::Primary,
            nats: real_sinex.nats.url().into(),
            password_file: None,
            subject_root: "rebuild".into(),
            producer: "polylogue".into(),
            store: real_sinex.store.clone(),
            upload_root: None,
        },
        socket: real_sinex.socket.clone(),
        real_sinex,
    };
    assert_eq!(
        running
            .real_sinex
            .sql("SELECT max(version) FROM sinex_schema.applied"),
        "47"
    );
    assert_eq!(running.real_sinex.sql("SELECT current_setting('fsync'),current_setting('synchronous_commit'),current_setting('full_page_writes')"),"on|on|on");
    drop(connect(&running.config)); // SDK producer hello over the same protocol-7 bus.
    seed(&running, "before-transition");
    let original = originals(&running);
    assert_eq!(original["events"].as_array().unwrap().len(), 1);
    assert_eq!(original["materials"].as_array().unwrap().len(), 1);
    let expected = broad_rows(&running);
    assert_eq!(expected.len(), 1);
    running.real_sinex.stop_cleanly();

    apply(&next, &running);
    running.real_sinex.restart_using(&next);
    let kinds = reply(&running, json!({"method":"kinds","kind":"state.interval"}));
    assert_eq!(kinds["outcome"], "ok", "{kinds}");
    assert!(
        kinds["result"]["kinds"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["declaration"]["time_rule"]["rule"] == "inherited"),
        "{kinds}"
    );
    assert_eq!(broad_rows(&running), expected);
    assert_eq!(originals(&running), original);
    running.real_sinex.stop_cleanly();

    apply(&before, &running);
    running.real_sinex.restart_using(&before);
    let refused = reply(
        &running,
        json!({"method":"query","query":"events | fields id, event_type, source, payload"}),
    );
    assert_ne!(
        refused["outcome"], "ok",
        "old binary must diagnose unsupported declaration: {refused}"
    );
    assert!(
        refused.to_string().contains("inherited"),
        "unexpected old-binary refusal: {refused}"
    );
    eprintln!("older binary declaration refusal: {refused}");
    assert_eq!(originals(&running), original);
    running.real_sinex.stop_cleanly();

    apply(&next, &running);
    running.real_sinex.restart_using(&next);
    assert_eq!(broad_rows(&running), expected);
    let kinds = reply(&running, json!({"method":"kinds","kind":"state.interval"}));
    assert_eq!(kinds["outcome"], "ok", "{kinds}");
    drop(connect(&running.config)); // A new SDK session must complete producer hello after recovery.
    seed(&running, "after-recovery");
    let recovered = originals(&running);
    for name in ["events", "materials"] {
        let rows = recovered[name].as_array().unwrap();
        assert_eq!(rows.len(), 2, "{name}: {recovered}");
        for row in original[name].as_array().unwrap() {
            assert!(rows.contains(row), "original {name} changed: {recovered}");
        }
    }
    assert_eq!(broad_rows(&running).len(), 2);
    assert_eq!(running.real_sinex.sql("SELECT count(*) FROM core.ingress_frames WHERE producer='evidence' AND refusal IS NOT NULL"),"0");
    eprintln!("recovered producer acknowledgments: {}",running.real_sinex.sql("SELECT json_agg(json_build_object('frame',frame,'commit',commit_seq,'admitted',admitted,'refusal',refusal) ORDER BY number) FROM core.ingress_frames WHERE producer='evidence' AND frame LIKE 'after-recovery/%'"));
    running.real_sinex.stop_cleanly();
    eprintln!(
        "binary recovery: original1event/1material unchanged; recovered2events/2materials; protocol7 admission; schema47; normal shutdowns"
    );
}
